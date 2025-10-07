import os
import torch
import torch.nn as nn
import torch.optim as optim
from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.base_value import BaseValue
import torch.nn.functional as F
import json

default_dropout = 0.01
default_dropout_2d = 0.03

class UNetRegressor(nn.Module):
    def __init__(self, input_channels=1, output_size=5, base_channels=32, input_size=(64, 64)):
        super().__init__()

        # Encoder
        self.enc1 = EnhancedConvBlock(input_channels, base_channels)
        self.pool1 = nn.AvgPool2d(2)
        self.enc2 = EnhancedConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.AvgPool2d(2)
        self.enc3 = EnhancedConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.AvgPool2d(2)
        self.enc4 = EnhancedConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.AvgPool2d(2)
        self.enc5 = EnhancedConvBlock(base_channels * 8, base_channels * 16)
        self.pool5 = nn.AvgPool2d(2)

        self.bottleneck = EnhancedConvBlock(base_channels * 16, base_channels * 32)

        # Decoder
        self.up5 = nn.ConvTranspose2d(base_channels * 32, base_channels * 16, kernel_size=2, stride=2)
        self.att5 = AttentionGate(base_channels * 16, base_channels * 16, base_channels * 8)
        self.dec5 = EnhancedConvBlock(base_channels * 32, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.att4 = AttentionGate(base_channels * 8, base_channels * 8, base_channels * 4)
        self.dec4 = EnhancedConvBlock(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate(base_channels * 4, base_channels * 4, base_channels * 2)
        self.dec3 = EnhancedConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(base_channels * 2, base_channels * 2, base_channels)
        self.dec2 = EnhancedConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.att1 = AttentionGate(base_channels, base_channels, base_channels // 2)
        self.dec1 = EnhancedConvBlock(base_channels * 2, base_channels)

        # Multi-scale fusion
        self.ms_bottleneck = MultiScaleFeatureExtractor(base_channels * 32)
        self.ms_dec5 = MultiScaleFeatureExtractor(base_channels * 16)
        self.ms_dec4 = MultiScaleFeatureExtractor(base_channels * 8)
        self.ms_dec3 = MultiScaleFeatureExtractor(base_channels * 4)
        self.ms_dec2 = MultiScaleFeatureExtractor(base_channels * 2)
        self.ms_dec1 = MultiScaleFeatureExtractor(base_channels)

        # Dynamically infer feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_size)
            features = self._forward_features(dummy)
            feature_dim = features.shape[1]

        # Improved fully connected regressor with more capacity
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(default_dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(default_dropout),
            nn.Linear(128, output_size)
        )

    def _forward_features(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        e5 = self.enc5(self.pool4(e4))

        b = self.bottleneck(self.pool5(e5))

        d5 = self.up5(b)
        d5 = torch.cat([self.att5(d5, e5), d5], dim=1)
        d5 = self.dec5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat([self.att4(d4, e4), d4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([self.att3(d3, e3), d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([self.att2(d2, e2), d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([self.att1(d1, e1), d1], dim=1)
        d1 = self.dec1(d1)

        return torch.cat([
            self.ms_bottleneck(b),
            self.ms_dec5(d5),
            self.ms_dec4(d4),
            self.ms_dec3(d3),
            self.ms_dec2(d2),
            self.ms_dec1(d1)
        ], dim=1)

    def forward(self, x):
        features = self._forward_features(x)
        return self.fc(features)


# ==============================
#   Attention and Squeeze Blocks
# ==============================

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ==============================
#   Core Convolutional Blocks
# ==============================

class EnhancedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=default_dropout_2d, use_residual=True):
        super().__init__()
        self.use_residual = use_residual

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)

        # residual adapter if channels differ
        self.residual_adapter = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else None
        )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)

        out = self.se(out)

        if self.use_residual:
            if self.residual_adapter is not None:
                residual = self.residual_adapter(residual)
            out += residual

        return self.relu(out)


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.pool3 = nn.AdaptiveAvgPool2d(2)
        self.pool4 = nn.AdaptiveAvgPool2d(4)
        self.pool5 = nn.AdaptiveAvgPool2d(8)

    def forward(self, x):
        B, C, H, W = x.shape
        features = []
        for pool in [self.pool1, self.pool2, self.pool3, self.pool4, self.pool5]:
            pooled = pool(x)
            if pooled.shape[2] <= H and pooled.shape[3] <= W:
                features.append(pooled.view(B, -1))
        return torch.cat(features, dim=1)


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
#   Early Stopping
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
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False


# ==============================
#   Main Training Class
# ==============================

class Conv2dNetTrainer(BaseProcessingObj):
    def __init__(self,
                 network_filename,
                 nmodes=20,
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.grad_clip_value = 0.5

        self.verbose = True
        self.network_filename = network_filename
        self.nmodes = nmodes
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
        self.model = UNetRegressor(
            input_channels=1,    
            output_size=nmodes,
            base_channels=32,  # Reduced from 128 for better generalization
            input_size=(160, 160)
        ).to(self.device)
        
        # Huber loss for robustness to outliers
        # Initialize with equal weights, will be updated based on data statistics
        ww = [0.05] * nmodes
        ww[0] = 4.0
        ww[1] = 3
        ww[2] = 3
        ww[3] = 0.5
        ww[4] = 0.5
        ww[5] = 0.5
        #ww[6] = 0.25
        #ww[7] = 0.25
        #ww[8] = 0.25
        #ww[9] = 0.25
        #ww[10] = 0.125
        #ww[11] = 0.125
        #ww[12] = 0.125
        #ww[13] = 0.125
        #ww[14] = 0.125
        self.loss_fn = WeightedMSELoss(
            weights=ww,
            device=self.device
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        
        # Learning rate warmup scheduler
        #warmup_steps = 50  # ~few epochs
        #def warmup_lambda(step):
        #    return min(1.0, 0.1 + (step + 1) / warmup_steps)
        #self.warmup_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup_lambda)
                
        self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.9,
            patience=40,
            verbose=True,
            min_lr=1e-6
        )

        # Early stopping
        self.early_stopping = EarlyStopping(patience=800, min_delta=1e-6)
        self.enable_early_stopping = True  # Disabled by default for safety

        # Gradient clipping threshold
        self.grad_clip_value = 1.0
        
        self.inputs['input_2d_batch'] = InputValue(type=BaseValue)
        self.inputs['labels'] = InputValue(type=BaseValue)

        self.outputs["loss"] = None
        self.X = None
        self.y = None
        self.min_loss = 1e9
        self.firstTrigger = True
        
        # Normalization statistics (will be computed from first batch)
        self.meanp = None
        self.stdp = None
        self.meanmodes = None
        self.stdmodes = None
        
        # Training statistics
        self.step_count = 0
        self.should_stop = False
        
        # Statistics file for saving normalization params
        self.stats_filename = network_filename.replace('.pth', '_stats.json')
    
    def update_loss_weights(self, modes):
        """Update loss weights based on target variance"""
        mode_stds = self.xp.std(modes, axis=0)
        # Inverse variance weighting
        weights = 1.0 / (mode_stds + 1e-8)
        # Normalize so they sum to nmodes (keep average weight = 1)
        weights = weights / weights.sum() * self.nmodes
        
        # Update loss function weights
        self.loss_fn.weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
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
            if self.verbose:
                print(f"[{self.name}] Skipping: X or y is None")
            return
        if self.X.generation_time < self.current_time or self.y.generation_time < self.current_time:
            if self.verbose:
                print(f"[{self.name}] Skipping: stale data")
            return

        try:
            ph = self.X.get_value()[:, 1] * self.X.get_value()[:, 0]
            modes = self.y.get_value()[:, 1:self.nmodes+1]
        except Exception as e:
            print(f"[{self.name}] ERROR extracting data: {e}")
            return
        
        # Initialize normalization statistics from first batch
        try:
            if self.firstTrigger:            
                self.meanp = self.xp.mean(ph)
                self.stdp = self.xp.std(ph) + 1e-8
                self.meanmodes = self.xp.mean(modes)
                self.stdmodes = self.xp.std(modes) + 1e-8

                # Update loss weights based on initial statistics
                # self.update_loss_weights(modes)
                
                self.firstTrigger = False
                
                if self.verbose:
                    print(f"[{self.name}] Initialized normalization:")
                    print(f"  Phase: mean={self.meanp:.4f}, std={self.stdp:.4f}")
                    print(f"  Modes: mean={self.meanmodes}, std={self.stdmodes}")

            else:
                alpha = 0.05  # update rate
                self.meanp = (1 - alpha) * self.meanp + alpha * self.xp.mean(ph)
                self.stdp = (1 - alpha) * self.stdp + alpha * (self.xp.std(ph) + 1e-8)
                self.meanmodes = (1 - alpha) * self.meanmodes + alpha * self.xp.mean(modes, axis=0)
                self.stdmodes = (1 - alpha) * self.stdmodes + alpha * (self.xp.std(modes, axis=0) + 1e-8)

        except Exception as e:
            print(f"[{self.name}] ERROR during initialization: {e}")
            return                

        try:
            # Apply normalization
            ph = (ph - self.meanp) / self.stdp
            modes = (modes - self.meanmodes) / self.stdmodes

            # Apply data augmentation BEFORE adding channel dimension
            # ph = self.apply_data_augmentation(ph)
            
            # Add channel dimension: (B, H, W) -> (B, 1, H, W)
            ph = ph[:, self.xp.newaxis, :, :]

            inputs = torch.tensor(ph, dtype=torch.float32, device=self.device)
            targets = torch.tensor(modes, dtype=torch.float32, device=self.device)
        except Exception as e:
            print(f"[{self.name}] ERROR during preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # === Training step ===
        try:
            self.model.train()
            for i in range(5):
                self.optimizer.zero_grad()
                preds = self.model(inputs)
                self.loss = self.loss_fn(preds, targets)
                
                # Check for NaN/Inf in loss
                if torch.isnan(self.loss) or torch.isinf(self.loss):
                    print(f"[{self.name}] ERROR: Loss is NaN or Inf! Skipping this batch.")
                    print(f"  Inputs stats: min={inputs.min():.4f}, max={inputs.max():.4f}, mean={inputs.mean():.4f}")
                    print(f"  Targets stats: min={targets.min():.4f}, max={targets.max():.4f}, mean={targets.mean():.4f}")
                    return
                        
                self.loss.backward()        
                
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)

                self.optimizer.step()

                # Update learning rate (warmup for first few epochs, then plateau)            
                # self.warmup_scheduler.step()            
                
            self.step_count += 1
        except Exception as e:
            print(f"[{self.name}] ERROR during training step: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # === Evaluation step ===
        try:
            self.model.eval()
            with torch.no_grad():
                eval_preds = self.model(inputs)
                eval_loss = self.loss_fn(eval_preds, targets).item()
            
            # Update plateau scheduler based on eval loss
            self.plateau_scheduler.step(eval_loss)
            
            # Save best model
            if eval_loss < self.min_loss:
                self.min_loss = eval_loss
                if eval_loss<0.01:
                    torch.save(self.model, self.network_filename)
                
                    # Save normalization statistics
                    stats = {
                        'meanp': float(self.meanp),
                        'stdp': float(self.stdp),
                        'meanmodes': self.meanmodes.tolist() if hasattr(self.meanmodes, 'tolist') else [float(x) for x in self.meanmodes],
                        'stdmodes': self.stdmodes.tolist() if hasattr(self.stdmodes, 'tolist') else [float(x) for x in self.stdmodes],
                        'nmodes': self.nmodes
                    }
                    with open(self.stats_filename, 'w') as f:
                        json.dump(stats, f, indent=2)
                    
                    if self.verbose:
                        print(f"[{self.name}] ✓ Model saved to {self.network_filename}")
            
            if self.verbose:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"[{self.name}] Step {self.step_count} | LR: {current_lr:.2e} | "
                      f"Train loss: {self.loss.item():.6f} | Eval loss: {eval_loss:.6f} | "
                      f"Best: {self.min_loss:.6f}")
            
            # Check early stopping (only if enabled)
            if self.enable_early_stopping and self.early_stopping(eval_loss):
                self.should_stop = True
                if self.verbose:
                    print(f"[{self.name}] Early stopping triggered after {self.step_count} steps")
                    print(f"[{self.name}] To disable early stopping, set trainer.enable_early_stopping = False")


            total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            if total_norm > 1.0 and self.verbose:
                print(f"[{self.name}] Gradient norm: {total_norm:.2f} (clipped)")

        except Exception as e:
            print(f"[{self.name}] ERROR during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return

    def post_trigger(self):
        super().post_trigger()
        if hasattr(self, 'loss'):
            self.outputs["loss"] = self.loss.item()

    def finalize(self):
        print(f'[{self.name}] Training complete!')
        print(f'  Total steps: {self.step_count}')
        print(f'  Best loss: {self.min_loss:.6f}')
        print(f'  Normalization saved to: {self.stats_filename}')