import os
import torch
import torch.nn as nn
from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula import cpuArray, np
import json


class Conv2dNetTester(BaseProcessingObj):
    def __init__(self,
                 network_filename,
                 nmodes=20,
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.verbose = True
        self.network_filename = network_filename
        self.nmodes = nmodes

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        if os.path.isfile(self.network_filename):
            self.model = torch.load(self.network_filename, map_location=self.device)
            self.model.eval()
            if self.verbose:
                print(f"[{self.name}] Loaded model from {self.network_filename}")
        else:
            raise FileNotFoundError(f"Model file not found at {self.network_filename}")
        
        # Load normalization statistics
        self.stats_filename = network_filename.replace('.pth', '_stats.json')
        if os.path.isfile(self.stats_filename):
            with open(self.stats_filename, 'r') as f:
                stats = json.load(f)
            self.meanp = stats['meanp']
            self.stdp = stats['stdp']
            self.meanmodes = self.xp.array(stats['meanmodes'])
            self.stdmodes = self.xp.array(stats['stdmodes'])
            if self.verbose:
                print(f"[{self.name}] Loaded normalization statistics from {self.stats_filename}")
                print(f"  Phase: mean={self.meanp:.4f}, std={self.stdp:.4f}")
                print(f"  Modes: mean shape={self.meanmodes.shape}, std shape={self.stdmodes.shape}")
        else:
            raise FileNotFoundError(f"Statistics file not found at {self.stats_filename}. "
                                   f"Make sure to train the model first!")
        
        # Loss function for evaluation
        self.loss_fn = nn.MSELoss()
        
        self.inputs['input_2d_batch'] = InputValue(type=BaseValue)
        self.inputs['labels'] = InputValue(type=BaseValue)
        
        self.outputs["loss"] = None
        self.outputs["prediction"] = None
        self.outputs["targets"] = None
        self.outputs["error"] = None

        # Statistics tracking
        self.total_abs_error = None
        self.total_squared_error = None
        self.count = 0
        self.all_errors = []

    def trigger(self):
        x_in = self.local_inputs['input_2d_batch']
        y_in = self.local_inputs['labels']

        if x_in is None or y_in is None:
            return

        # Extract phase and modes
        ph = x_in.get_value()[:, 1] * x_in.get_value()[:, 0]
        modes = y_in.get_value()[:, 1:self.nmodes+1]

        if self.verbose:
            print(f"[{self.name}] Input phase shape: {ph.shape}")
            print(f"[{self.name}] Target modes shape: {modes.shape}")
            print(f"[{self.name}] Target modes (original): {modes}")

        # Apply the SAME normalization as training
        ph = (ph - self.meanp) / self.stdp
        modes_normalized = (modes - self.meanmodes) / self.stdmodes

        ph = ph[:, self.xp.newaxis, :, :]  # shape (B, 1, H, W)

        # Convert to torch tensors
        inputs = torch.tensor(ph, dtype=torch.float32, device=self.device)
        targets_normalized = torch.tensor(modes_normalized, dtype=torch.float32, device=self.device)

        # Inference
        with torch.no_grad():
            preds_normalized = self.model(inputs)
            
            # Compute loss on normalized values
            loss_normalized = self.loss_fn(preds_normalized, targets_normalized).item()

        # Denormalize predictions to original scale
        preds_normalized_np = self.xp.asarray(preds_normalized.cpu().numpy())
        preds = preds_normalized_np * (self.stdmodes + self.meanmodes)

        # Compute error in original units
        error = preds - modes
        
        # Update statistics
        if self.total_abs_error is None:
            self.total_abs_error = self.xp.abs(error)
            self.total_squared_error = error ** 2
        else:
            self.total_abs_error += self.xp.abs(error)
            self.total_squared_error += error ** 2
        
        self.count += ph.shape[0]  # Count batch size
        self.all_errors.append(error)

        # Store outputs
        self.outputs["loss"] = loss_normalized
        self.outputs["prediction"] = preds
        self.outputs["targets"] = modes
        self.outputs["error"] = error

        if self.verbose:
            print(f"[{self.name}] Predictions (denormalized): {preds}")
            print(f"[{self.name}] Targets (original): {modes}")
            print(f"[{self.name}] Error per mode: {error}")
            print(f"[{self.name}] Normalized loss: {loss_normalized:.6f}")
            print(f"[{self.name}] Mean absolute error: {self.xp.mean(self.xp.abs(error)):.6f}")
            print(f"[{self.name}] RMS error: {self.xp.sqrt(self.xp.mean(error**2)):.6f}")

    def post_trigger(self):
        super().post_trigger()

    def finalize(self):
        if self.count == 0:
            print(f"[{self.name}] No data processed!")
            return super().finalize()
        
        # Compute final statistics
        mean_abs_error = self.total_abs_error / self.count
        mean_squared_error = self.total_squared_error / self.count
        rms_error = self.xp.sqrt(mean_squared_error)
        print('mean_abs_error', mean_abs_error.shape)
        print('mean_squared_error', mean_squared_error.shape)
        print('rms_error', rms_error.shape)

        # Compute per-mode statistics
        all_errors_concat = self.xp.concatenate(self.all_errors, axis=0)
        std_error = self.xp.std(all_errors_concat, axis=0)
        
        print(f"\n{'='*80}")
        print(f"[{self.name}] FINAL TEST STATISTICS")
        print(f"{'='*80}")
        print(f"Total samples processed: {self.count}")
        print(f"\nPer-mode statistics:")
        print(f"{'Mode':<6} {'MAE':<12} {'RMSE':<12} {'StdErr':<12}")
        print(f"{'-'*48}")
        
        mean_abs_error = cpuArray(mean_abs_error)
        rms_error = cpuArray(rms_error)
        std_error = cpuArray(std_error)
        for i in range(self.nmodes):
            print(i, 'meanabs std rms', mean_abs_error[0,i], std_error[i], rms_error[0,i])
        
        print(f"\nOverall statistics:")
        print(f"  Mean Absolute Error (averaged across all modes): {self.xp.mean(mean_abs_error):.6f}")
        print(f"  Root Mean Squared Error (averaged across all modes): {self.xp.mean(rms_error):.6f}")
        print(f"  Maximum MAE across modes: {self.xp.max(mean_abs_error):.6f} (mode {np.argmax(mean_abs_error)})")
        print(f"  Minimum MAE across modes: {self.xp.min(mean_abs_error):.6f} (mode {np.argmin(mean_abs_error)})")
        print(f"{'='*80}\n")
        
        return super().finalize()