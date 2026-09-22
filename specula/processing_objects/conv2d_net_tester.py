import torch

from specula import cpuArray, np
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.lib.cnn_checkpoint import load_trained_network
from specula.lib.frame_stacker import FrameStacker
from specula.processing_objects.conv2d_net_trainer import network_input


class Conv2dNetTester(BaseProcessingObj):
    """
    Evaluates a network trained by Conv2dNetTrainer on buffered batches
    with known modes: no training, just the predictions ('prediction'
    output, and the batch's mean squared error as 'loss') and, at the end
    of the simulation, per-mode error statistics.

    The network parameters (nmodes, channels, dropout, conv_block_type,
    depth, head_type, input_channels, n_frames) and label_offset /
    baseline_offset must match the ones used for training (see
    Conv2dNetTrainer); n_frames and head_type are checked against the
    checkpoint's stats file.

    With the optional 'baseline' input connected, the network output is a
    residual: the baseline is added back to it before comparing with the
    labels. With the optional 'gain_mod' input connected, samples with
    gain_mod < gain_mod_threshold (loop open or at reduced gain, not
    representative of closed-loop operation) are left out of the statistics.
    """

    def __init__(self,
                 network_filename,
                 nmodes=20,
                 channels=32,
                 dropout=0.01,
                 conv_block_type=0,
                 depth=5,
                 head_type='pooled',
                 head_grid=32,
                 input_channels=1,
                 n_frames=1,
                 label_offset=1,
                 baseline_offset=0,
                 gain_mod_threshold=1.0,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.nmodes = nmodes
        self.input_channels = input_channels
        self.label_offset = label_offset
        self.baseline_offset = baseline_offset
        self.gain_mod_threshold = gain_mod_threshold
        self.stacker = FrameStacker(n_frames, self.xp)

        self.model, stats = load_trained_network(
            network_filename, nmodes=nmodes, input_channels=input_channels, n_frames=n_frames,
            channels=channels, depth=depth, dropout=dropout, conv_block_type=conv_block_type,
            head_type=head_type, head_grid=head_grid)
        self.meanp = stats['meanp']
        self.stdp = stats['stdp']
        self.meanmodes = self.xp.array(stats['meanmodes'])
        self.stdmodes = self.xp.array(stats['stdmodes'])

        self.inputs['input_2d_batch'] = InputValue(type=BaseValue)
        self.inputs['labels'] = InputValue(type=BaseValue)
        self.inputs['baseline'] = InputValue(type=BaseValue, optional=True)
        self.inputs['gain_mod'] = InputValue(type=BaseValue, optional=True)
        self.outputs['loss'] = BaseValue(target_device_idx=target_device_idx)
        self.outputs['prediction'] = BaseValue(target_device_idx=target_device_idx)

        self.count = 0
        self.all_errors = []
        self.all_targets = []

    def trigger(self):
        x_in = self.local_inputs['input_2d_batch']
        labels_in = self.local_inputs['labels']
        if x_in is None or labels_in is None:
            return

        # Stacked before any frame is dropped (see Conv2dNetTrainer).
        x = self.stacker(network_input(x_in.get_value(), self.input_channels))
        first = self.label_offset
        modes = labels_in.get_value()[:, first:first + self.nmodes]
        baseline = None
        baseline_in = self.local_inputs['baseline']
        if baseline_in is not None:
            b = self.baseline_offset
            baseline = baseline_in.get_value()[:, b:b + self.nmodes]

        gain_mod_in = self.local_inputs['gain_mod']
        if gain_mod_in is not None:
            keep = self.xp.asarray(gain_mod_in.get_value()).reshape(-1) >= self.gain_mod_threshold
            if not self.xp.any(keep):
                return
            x, modes = x[keep], modes[keep]
            if baseline is not None:
                baseline = baseline[keep]

        x = (x - self.meanp) / self.stdp
        if x.ndim == 3:     # single-channel frames: add the channel axis
            x = x[:, self.xp.newaxis]
        with torch.no_grad():
            preds_normalized = self.model(torch.from_numpy(np.ascontiguousarray(cpuArray(x), dtype=np.float32)))
        preds = self.xp.asarray(preds_normalized.numpy()) * self.stdmodes + self.meanmodes
        if baseline is not None:
            preds = preds + baseline

        error = preds - modes
        self.count += x.shape[0]
        self.all_errors.append(error)
        self.all_targets.append(modes)

        self.outputs['prediction'].value = preds
        self.outputs['prediction'].generation_time = self.current_time
        self.outputs['loss'].set_value(float(self.xp.mean(error ** 2)))
        self.outputs['loss'].generation_time = self.current_time

    def finalize(self):
        if self.count == 0:
            print(f'[{self.name}] No data processed!')
            return super().finalize()

        errors = cpuArray(self.xp.concatenate(self.all_errors, axis=0))
        targets = cpuArray(self.xp.concatenate(self.all_targets, axis=0))
        mae = np.mean(np.abs(errors), axis=0)
        rmse = np.sqrt(np.mean(errors ** 2, axis=0))
        std_error = np.std(errors, axis=0)
        # Symmetric mean absolute percentage error
        smape = 100.0 * np.mean(
            2.0 * np.abs(errors) / (np.abs(targets) + np.abs(errors + targets) + 1e-3), axis=0)

        print(f"\n{'=' * 80}")
        print(f'[{self.name}] FINAL TEST STATISTICS')
        print(f"{'=' * 80}")
        print(f'Total samples processed: {self.count}')
        print('\nPer-mode statistics:')
        print(f"{'Mode':<6} {'MAE':<12} {'RMSE':<12} {'SMAPE (%)':<12} {'StdErr':<12}")
        print(f"{'-' * 60}")
        for i in range(self.nmodes):
            print(f'{i:<6d} {mae[i]:<12.6f} {rmse[i]:<12.6f} {smape[i]:<12.2f} {std_error[i]:<12.6f}')
        print('\nOverall statistics:')
        print(f'  Mean Absolute Error (averaged): {np.mean(mae):.6f}')
        print(f'  Root Mean Squared Error (averaged): {np.mean(rmse):.6f}')
        print(f'  Mean SMAPE: {np.mean(smape):.2f}%')
        print(f'  Maximum MAE across modes: {np.max(mae):.6f} (mode {np.argmax(mae)})')
        print(f'  Minimum MAE across modes: {np.min(mae):.6f} (mode {np.argmin(mae)})')
        print(f"{'=' * 80}\n")

        self.mean_absolute_error = mae
        self.root_mean_squared_error = rmse
        self.smape = smape
        return super().finalize()
