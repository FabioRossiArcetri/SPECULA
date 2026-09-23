import torch

from specula import cpuArray, np
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.lib.cnn_checkpoint import calibration_factor, load_trained_network
from specula.lib.frame_stacker import FrameStacker
from specula.lib.nn_training_diagnostics import mode_groups
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

    With the optional 'reference' input connected -- another estimate of the
    same modes, typically the linear reconstructor's out_modes buffered
    alongside the labels -- its error is accumulated on exactly the same
    frames and reported next to the network's, which is the way to tell
    whether the network reconstructs better than the reconstructor it is
    meant to replace.
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
                 calibrate_gain=False,
                 input_channels=1,
                 n_frames=1,
                 label_offset=1,
                 baseline_offset=0,
                 reference_offset=0,
                 gain_mod_threshold=1.0,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.nmodes = nmodes
        self.input_channels = input_channels
        self.label_offset = label_offset
        self.baseline_offset = baseline_offset
        self.reference_offset = reference_offset
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
        # see Conv2dNetRec for what this is; scored here so the test matches
        # how the network is used in closed loop
        factor = calibration_factor(stats, nmodes) if calibrate_gain else None
        self.calibration = self.xp.array(factor) if factor is not None else None
        if calibrate_gain and self.calibration is None:
            raise ValueError(f'calibrate_gain is set, but {network_filename} has no per-mode gain '
                             f'measurement (it was trained before this existed)')

        self.inputs['input_2d_batch'] = InputValue(type=BaseValue)
        self.inputs['labels'] = InputValue(type=BaseValue)
        self.inputs['baseline'] = InputValue(type=BaseValue, optional=True)
        self.inputs['gain_mod'] = InputValue(type=BaseValue, optional=True)
        self.inputs['reference'] = InputValue(type=BaseValue, optional=True)
        self.outputs['loss'] = BaseValue(target_device_idx=target_device_idx)
        self.outputs['prediction'] = BaseValue(target_device_idx=target_device_idx)

        self.count = 0
        self.all_errors = []
        self.all_targets = []
        self.all_reference_errors = []

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
        reference = None
        reference_in = self.local_inputs['reference']
        if reference_in is not None:
            r = self.reference_offset
            reference = reference_in.get_value()[:, r:r + self.nmodes]

        gain_mod_in = self.local_inputs['gain_mod']
        if gain_mod_in is not None:
            keep = self.xp.asarray(gain_mod_in.get_value()).reshape(-1) >= self.gain_mod_threshold
            if not self.xp.any(keep):
                return
            x, modes = x[keep], modes[keep]
            if baseline is not None:
                baseline = baseline[keep]
            if reference is not None:
                reference = reference[keep]

        x = (x - self.meanp) / self.stdp
        if x.ndim == 3:     # single-channel frames: add the channel axis
            x = x[:, self.xp.newaxis]
        with torch.no_grad():
            preds_normalized = self.model(torch.from_numpy(np.ascontiguousarray(cpuArray(x), dtype=np.float32)))
        preds = self.xp.asarray(preds_normalized.numpy()) * self.stdmodes + self.meanmodes
        if self.calibration is not None:
            preds = (preds - self.meanmodes) * self.calibration + self.meanmodes
        if baseline is not None:
            preds = preds + baseline

        error = preds - modes
        self.count += x.shape[0]
        self.all_errors.append(error)
        self.all_targets.append(modes)
        if reference is not None:
            self.all_reference_errors.append(reference - modes)

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

        self._print_group_table(errors, targets)

        self.mean_absolute_error = mae
        self.root_mean_squared_error = rmse
        self.smape = smape
        return super().finalize()

    def _print_group_table(self, errors, targets):
        """Per mode group: how much of the residual each estimator explains,
        on exactly the same frames. FVU = 1 means no better than predicting
        the mean; the lower of the two columns is the better estimator."""
        reference = (cpuArray(self.xp.concatenate(self.all_reference_errors, axis=0))
                     if self.all_reference_errors else None)
        rms = lambda e, a, b: float(np.sqrt((e[:, a:b] ** 2).sum(axis=1).mean()))
        var = lambda a, b: float(((targets[:, a:b] - targets[:, a:b].mean(0)) ** 2).sum(axis=1).mean())
        header = f"{'modes':>9} {'label_rms':>10} {'cnn_err':>9} {'cnn_FVU':>8}"
        if reference is not None:
            header += f" | {'ref_err':>9} {'ref_FVU':>8}"
        print('\nPer mode group (RMS in quadrature over the group, in the labels units):')
        print(header)
        for a, b in mode_groups(self.nmodes):
            v = max(var(a, b), 1e-30)
            line = (f"{f'{a}-{b - 1}':>9} {np.sqrt(v):>10.2f} {rms(errors, a, b):>9.2f} "
                    f"{rms(errors, a, b) ** 2 / v:>8.3f}")
            if reference is not None:
                line += f" | {rms(reference, a, b):>9.2f} {rms(reference, a, b) ** 2 / v:>8.3f}"
            print(line)
        total = f"{'total':>9} {np.sqrt(max(var(0, self.nmodes), 1e-30)):>10.2f} {rms(errors, 0, self.nmodes):>9.2f}"
        print(total + (f" {'':>8} | {rms(reference, 0, self.nmodes):>9.2f}" if reference is not None else ''))
