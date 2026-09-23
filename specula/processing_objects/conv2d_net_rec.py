import numpy as np
import torch

from specula import cpuArray
from specula.base_processing_obj import InputDesc
from specula.processing_objects.base_modalrec import BaseModalrec
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula.lib.cnn_checkpoint import calibration_factor, load_trained_network
from specula.lib.frame_stacker import FrameStacker


class Conv2dNetRec(BaseModalrec):
    """
    Closed-loop-usable, deterministic inference reconstructor: loads a
    network trained by :class:`~specula.processing_objects.conv2d_net_trainer.Conv2dNetTrainer`
    and reconstructs modes from a 2D map built from the slopes, on every
    trigger. Same ``in_slopes -> out_modes`` contract as
    :class:`~specula.processing_objects.modalrec.Modalrec`, so it is a
    drop-in replacement for it in any existing SCAO YAML.

    This is the counterpart of
    :class:`~specula.processing_objects.conv2d_net_tester.Conv2dNetTester`
    for actual closed-loop use: no 'labels' input (there is no ground
    truth at runtime), and no evaluation-statistics bookkeeping -- just
    inference, every step.

    Note
    ----
    Only slopes-based 2D input (via `Slopes.get2d()`) is supported for
    now; pixel-based input (`Conv2dNetTrainer`'s `input_channels`
    generalization already supports it architecturally) is deliberately
    left for a follow-up, matching how it was flagged as future work
    rather than an immediate need.
    """

    def __init__(self,
                 network_filename,
                 baseline_offset=0,
                 calibrate_gain=False,
                 target_device_idx: int = None,
                 precision: int = None):
        """
        network_filename : str
            A checkpoint written by Conv2dNetTrainer. The network -- its
            architecture, nmodes, input_channels and n_frames -- is read from
            it, so nothing in the config can disagree with the weights.
            Until n_frames slope maps have arrived, the missing ones repeat
            the first.
        baseline_offset : int, optional
            Index of the first mode within the connected optional
            'baseline' input, if any. When connected, the network's
            (denormalized) output is treated as a residual and the
            baseline is added back to it to form the final out_modes --
            the closed-loop counterpart of Conv2dNetTrainer/Tester's
            residual-learning mode.
        calibrate_gain : bool, optional
            Undo the shrinkage of the network's predictions towards the mean,
            using the per-mode gain measured during training and saved in the
            checkpoint (see cnn_checkpoint.calibration_factor). The loop gains
            in a config then mean what they say: without it, a retrained
            network that shrinks less silently raises the effective loop gain.
            Modes the network barely predicts are left uncorrected, since
            amplifying them would mostly amplify noise.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.model, stats = load_trained_network(network_filename)
        network = stats['network']
        self.nmodes = nmodes = network['nmodes']
        self.input_channels = network['input_channels']
        self.n_frames = network['n_frames']
        self.baseline_offset = baseline_offset
        self.stacker = FrameStacker(self.n_frames, np)
        self.meanp = stats['meanp']
        self.stdp = stats['stdp']
        self.meanmodes = np.asarray(stats['meanmodes'], dtype=float)
        self.stdmodes = np.asarray(stats['stdmodes'], dtype=float)
        self.calibration = calibration_factor(stats, nmodes) if calibrate_gain else None
        if calibrate_gain and self.calibration is None:
            raise ValueError(f'calibrate_gain is set, but {network_filename} has no per-mode gain '
                             f'measurement (it was trained before this existed): retrain, or set '
                             f'calibrate_gain to false and fold the shrinkage into the loop gains')

        self.inputs['baseline'] = InputValue(type=BaseValue, optional=True)
        self.modes.value = self.xp.zeros(nmodes, dtype=self.dtype)

    @classmethod
    def input_names(cls):
        names = dict(BaseModalrec.input_names())
        names['baseline'] = InputDesc(
            BaseValue, 'Baseline reconstructor output added back to the network '
                      'output for residual learning; see Conv2dNetTrainer (optional)')
        return names

    def trigger_code(self):
        slopes_obj = self.local_inputs['in_slopes']
        map2d = cpuArray(slopes_obj.get2d())
        if map2d.ndim == 2:
            map2d = map2d[np.newaxis, ...]
        map2d = map2d.astype(np.float32)

        stacked = self.stacker(map2d[np.newaxis, ...])   # (1, input_channels * n_frames, H, W)
        ph = (stacked - self.meanp) / self.stdp
        inputs = torch.from_numpy(ph.astype(np.float32))

        with torch.no_grad():
            preds_normalized = self.model(inputs)

        preds = preds_normalized.cpu().numpy()[0] * self.stdmodes + self.meanmodes
        if self.calibration is not None:
            preds = (preds - self.meanmodes) * self.calibration + self.meanmodes

        baseline_in = self.local_inputs['baseline']
        if baseline_in is not None:
            baseline_modes = cpuArray(baseline_in.get_value())[
                self.baseline_offset:self.baseline_offset + self.nmodes]
            preds = preds + baseline_modes

        self.modes.value[:] = self.to_xp(preds, dtype=self.dtype)
        self.modes.generation_time = self.current_time
