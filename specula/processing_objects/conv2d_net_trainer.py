import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json

from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula.lib.efficient_u_net import UNetRegressor

import os

# ==============================
#   Loss Functions
# ==============================

class WeightedHuberLoss(nn.Module):
    """Robust Huber loss with per-dimension weights"""
    def __init__(self, weights, delta=1.0, device=None):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32).to(device))
        self.delta = delta
    
    def forward(self, preds, targets):
        diff = preds - targets
        abs_diff = torch.abs(diff)
        quadratic = torch.clamp(abs_diff, max=self.delta)
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic**2 + self.delta * linear
        return (self.weights * loss).mean()


class WeightedMSELoss(nn.Module):
    """Element-wise weighted MSE loss"""
    def __init__(self, weights, device):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32).to(device))

    def forward(self, preds, targets):
        diff = preds - targets
        weighted_sq = self.weights * (diff ** 2)
        return weighted_sq.mean()


# ==============================
#   Early Stopping (fixed)
# ==============================

class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        # --- FIX ---
        # Proper comparison: stop if no improvement
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


# ==============================
#   Main Training Class (patched)
# ==============================

class Conv2dNetTrainer(BaseProcessingObj):
    def __init__(self,
                 network_filename,
                 nmodes=20,
                 epoch_len=20,
                 dropout=0.01,
                 patience=600,
                 channels=32,
                 load_from_file=False,
                 conv_block_type=0,
                 depth=5,
                 val_split=0.2,
                 target_device_idx=None,
                 precision=None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.channels = channels
        self.patience = patience
        self.dropout = dropout
        self.epoch_len = epoch_len
        self.grad_clip_value = 1.0
        self.val_split = val_split
        self.verbose = True


        # --- FIX: robust filename handling ---
        base, ext = os.path.splitext(network_filename)
        if ext == '':
            ext = '.pth'

        # Detect if filename already has parameter suffix
        if any(tag in base for tag in ['_cvtype', '_m', '_ch', '_dp']):
            # Assume user passed a full model path with params already
            self.network_filename = base + ext
        else:
            params_string = f'_cvtype{conv_block_type}_m{nmodes}_ch{self.channels}_dp{self.dropout:.3f}'
            self.network_filename = f"{base}{params_string}{ext}"

        # Always use the same stats file base name
        self.stats_filename = self.network_filename.replace('.pth', '_stats.json')

#        params_string = f'_cvtype{conv_block_type}_m{nmodes}_ch{self.channels}_dp{self.dropout:.3f}'
#        self.network_filename = network_filename.replace('.pth', params_string + '.pth')
        self.nmodes = nmodes

        # --- FIX --- Main device and GPU setup
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.multi_gpu = torch.cuda.device_count() > 1
        if self.verbose:
            print(f"[{self.name}] Using {torch.cuda.device_count()} GPUs")
            if self.multi_gpu:
                print(f"[{self.name}] Main device: {self.device}")

        # =========================
        #   Load or create model
        # =========================
        if load_from_file:
            try:
                checkpoint = torch.load(self.network_filename, map_location='cpu')
                model = UNetRegressor(
                    input_channels=1,
                    output_size=nmodes,
                    base_channels=self.channels,
                    input_size=(160, 160),
                    dropout_level=self.dropout,
                    conv_block_type=conv_block_type,
                    depth=depth
                )

                # --- FIX --- Handle "module." keys
                def _clean_state_dict(sd):
                    return {k.replace('module.', ''): v for k, v in sd.items()}

                if isinstance(checkpoint, dict):
                    model.load_state_dict(_clean_state_dict(checkpoint), strict=False)
                elif hasattr(checkpoint, 'state_dict'):
                    model.load_state_dict(_clean_state_dict(checkpoint.state_dict()), strict=False)
                else:
                    model = checkpoint

                self.model = model
                self.model.eval()

                # Load normalization stats
                self.stats_filename = self.network_filename.replace('.pth', '_stats.json')
                try:
                    with open(self.stats_filename, 'r') as f:
                        stats = json.load(f)
                    self.meanp = stats['meanp']
                    self.stdp = stats['stdp']
                    self.meanmodes = torch.tensor(stats['meanmodes'])
                    self.stdmodes = torch.tensor(stats['stdmodes'])
                    if self.verbose:
                        print(f"[{self.name}] ✓ Model loaded from {self.network_filename}")
                        print(f"[{self.name}] ✓ Stats loaded from {self.stats_filename}")
                except FileNotFoundError:
                    print(f"[{self.name}] WARNING: Stats file not found.")
                    self.meanp = None
                    self.stdp = None
                    self.meanmodes = None
                    self.stdmodes = None

            except Exception as e:
                print(f"[{self.name}] WARNING: Failed to load model: {e}. Creating new model.")
                load_from_file = False

        if not load_from_file:
            self.model = UNetRegressor(
                input_channels=1,
                output_size=nmodes,
                base_channels=self.channels,
                input_size=(160, 160),
                dropout_level=self.dropout,
                conv_block_type=conv_block_type,
                depth=depth
            )
            self.meanp = None
            self.stdp = None
            self.meanmodes = None
            self.stdmodes = None

        self.model = self.model.cpu()

        if self.multi_gpu and self.device.type == 'cuda':
            gpu_ids = list(range(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model, device_ids=gpu_ids)
            self.model = self.model.to(self.device)
            if self.verbose:
                print(f"[{self.name}] Model wrapped with DataParallel")
        else:
            self.model = self.model.to(self.device)
            if self.verbose:
                print(f"[{self.name}] Model using single device: {self.device}")

        self._verify_device_consistency()

        # =========================
        #   Loss and optim setup
        # =========================
        ww = [0.05] * nmodes
        ww[0] = 2.0
        ww[1:6] = [0.02, 0.02, 0.01, 0.01, 0.01]

        self.loss_fn = WeightedMSELoss(weights=ww, device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0)
        self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.8, patience=self.patience // 10,
            verbose=True, min_lr=0.5e-5
        )

        self.early_stopping = EarlyStopping(patience=self.patience, min_delta=1e-6)
        self.enable_early_stopping = True

        self.inputs['input_2d_batch'] = InputValue(type=BaseValue)
        self.inputs['labels'] = InputValue(type=BaseValue)
        self.outputs["loss"] = BaseValue(target_device_idx=target_device_idx)

        self.X = self.y = None
        self.min_loss = 1e9
        self.firstTrigger = True
        self.step_count = 0
        self.should_stop = False
        self.val_inputs = self.val_targets = None
        self.val_initialized = False
        self.stats_filename = self.network_filename.replace('.pth', '_stats.json')

    # --- FIX --- Safe numpy/cupy → torch converter
    def _to_torch(self, arr, device, dtype=torch.float32):
        xp = getattr(self, 'xp', __import__('numpy'))
        if xp is not __import__('numpy'):
            arr = xp.asnumpy(arr)
        return torch.from_numpy(arr).to(device=device, dtype=dtype)

    def _verify_device_consistency(self):
        if hasattr(self.model, 'module'):
            params = list(self.model.module.parameters())
        else:
            params = list(self.model.parameters())
        if not params:
            print(f"[{self.name}] WARNING: Model has no parameters")
            return
        devices = {p.device for p in params}
        if len(devices) > 1:
            print(f"[{self.name}] ERROR: Model parameters span devices: {devices}")
            self.model = self.model.to(self.device)
        else:
            if self.verbose:
                print(f"[{self.name}] All model parameters on {next(iter(devices))}")

    def update_loss_weights(self, modes):
        mode_stds = self.xp.std(modes, axis=0)
        weights = 1.0 / (mode_stds + 1e-8)
        weights = weights / weights.sum() * self.nmodes

        w_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            self.loss_fn.weights.data.copy_(w_tensor)

        if self.verbose and self.step_count % 100 == 0:
            print(f"[{self.name}] Updated loss weights: {weights}")

    def trigger(self):
        if self.should_stop:
            if self.verbose:
                print(f"[{self.name}] Training stopped (should_stop=True)")
            return

        self.X = self.local_inputs['input_2d_batch']
        self.y = self.local_inputs['labels']
        if self.X is None or self.y is None:
            return

        try:
            ph = self.X.get_value()[:, 1] * self.X.get_value()[:, 0]
            modes = self.y.get_value()[:, 1:self.nmodes + 1]
        except Exception as e:
            print(f"[{self.name}] ERROR extracting data: {e}")
            return

        try:
            if self.firstTrigger:
                self.meanp = self.xp.mean(ph)
                self.stdp = self.xp.std(ph) + 1e-8
                # --- FIX --- per-mode stats
                self.meanmodes = self.xp.mean(modes, axis=0)
                self.stdmodes = self.xp.std(modes, axis=0) + 1e-8
                self.firstTrigger = False
                if self.verbose:
                    print(f"[{self.name}] Normalization initialized.")
            else:
                alpha = 0.001
                self.meanp = (1 - alpha) * self.meanp + alpha * self.xp.mean(ph)
                self.stdp = (1 - alpha) * self.stdp + alpha * (self.xp.std(ph) + 1e-8)
                self.meanmodes = (1 - alpha) * self.meanmodes + alpha * self.xp.mean(modes, axis=0)
                self.stdmodes = (1 - alpha) * self.stdmodes + alpha * (self.xp.std(modes, axis=0) + 1e-8)

            ph = (ph - self.meanp) / self.stdp
            modes = (modes - self.meanmodes) / self.stdmodes
            ph = ph[:, self.xp.newaxis, :, :]

            batch_size = ph.shape[0]
            n_val = max(1, int(batch_size * self.val_split))
            n_train = batch_size - n_val

            # --- FIX --- reproducible, CPU-safe split
            perm = torch.randperm(batch_size).numpy()
            train_idx = perm[:n_train]
            val_idx = perm[n_train:]

            ph_train, modes_train = ph[train_idx], modes[train_idx]
            ph_val, modes_val = ph[val_idx], modes[val_idx]

            params = list(self.model.parameters())
            if not params:
                raise RuntimeError("Model has no parameters")
            model_device = params[0].device

            inputs = self._to_torch(ph_train, model_device)
            targets = self._to_torch(modes_train, model_device)
            val_inputs_new = self._to_torch(ph_val, model_device)
            val_targets_new = self._to_torch(modes_val, model_device)

            if not self.val_initialized:
                self.val_inputs = val_inputs_new.detach()
                self.val_targets = val_targets_new.detach()
                self.val_initialized = True
                if self.verbose:
                    print(f"[{self.name}] Validation set initialized ({n_val} samples)")
            else:
                max_val = 1000
                self.val_inputs = torch.cat([self.val_inputs, val_inputs_new.detach()], 0)[-max_val:]
                self.val_targets = torch.cat([self.val_targets, val_targets_new.detach()], 0)[-max_val:]

        except Exception as e:
            print(f"[{self.name}] ERROR preprocessing: {e}")
            import traceback; traceback.print_exc()
            return

        # Training
        try:
            self.model.train()
            for _ in range(self.epoch_len):
                self.optimizer.zero_grad()
                preds = self.model(inputs)
                loss = self.loss_fn(preds, targets)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[{self.name}] NaN/Inf loss, skipping batch.")
                    return

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
                self.optimizer.step()

            self.loss = loss
            self.step_count += 1

        except Exception as e:
            print(f"[{self.name}] ERROR training step: {e}")
            import traceback; traceback.print_exc()
            return

        # Validation
        try:
            if not self.val_initialized:
                return
            self.model.eval()
            with torch.no_grad():
                preds_val = self.model(self.val_inputs)
                eval_loss = self.loss_fn(preds_val, self.val_targets).item()

            self.plateau_scheduler.step(eval_loss)

            if eval_loss < self.min_loss:
                self.min_loss = eval_loss
                # --- FIX --- always save best model
                model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                torch.save(model_to_save.state_dict(), self.network_filename)
                stats = {
                    'meanp': float(self.meanp),
                    'stdp': float(self.stdp),
                    'meanmodes': self.meanmodes.tolist(),
                    'stdmodes': self.stdmodes.tolist(),
                    'nmodes': self.nmodes
                }
                with open(self.stats_filename, 'w') as f:
                    json.dump(stats, f, indent=2)
                if self.verbose:
                    print(f"[{self.name}] ✓ Model saved to {self.network_filename}")

            if self.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"[{self.name}] Step {self.step_count} | LR {lr:.2e} | "
                      f"Train {self.loss.item():.6f} | Val {eval_loss:.6f} | Best {self.min_loss:.6f}")

            if self.enable_early_stopping and self.early_stopping(eval_loss):
                self.should_stop = True
                print(f"[{self.name}] Early stopping triggered at step {self.step_count}")

        except Exception as e:
            print(f"[{self.name}] ERROR evaluation: {e}")
            import traceback; traceback.print_exc()
            return

    def post_trigger(self):
        super().post_trigger()
        if hasattr(self, 'loss'):
            if hasattr(self.outputs["loss"], 'set_value'):
                self.outputs["loss"].set_value(self.loss.item())
            else:
                self.outputs["loss"] = BaseValue(self.loss.item(), target_device_idx=self.outputs["loss"].target_device_idx)

    def finalize(self):
        print(f'[{self.name}] Training complete!')
        print(f'  Steps: {self.step_count}')
        print(f'  Best loss: {self.min_loss:.6f}')
        print(f'  Stats saved: {self.stats_filename}')
