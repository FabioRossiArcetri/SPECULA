import numpy as np
import torch

from specula import cpuArray
from specula.base_processing_obj import InputDesc
from specula.processing_objects.base_modalrec import BaseModalrec
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula.lib.cnn_checkpoint import load_trained_network
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
                 nmodes=20,
                 channels=32,
                 dropout=0.01,
                 conv_block_type=0,
                 depth=5,
                 input_channels=2,
                 baseline_offset=0,
                 n_frames=1,
                 head_type='pooled',
                 head_grid=32,
                 target_device_idx: int = None,
                 precision: int = None):
        """
        input_channels : int, optional
            Must match the value used to train the loaded network (see
            Conv2dNetTrainer). Default 2, matching the two channels
            (x slopes, y slopes) returned by Slopes.get2d() for an
            ordinary (non slopes-from-intensity) slope map; use 1 for a
            slopes-from-intensity map.
        baseline_offset : int, optional
            Index of the first mode within the connected optional
            'baseline' input, if any. When connected, the network's
            (denormalized) output is treated as a residual and the
            baseline is added back to it to form the final out_modes --
            the closed-loop counterpart of Conv2dNetTrainer/Tester's
            residual-learning mode.
        n_frames : int, optional
            Consecutive slope maps stacked as the network input (the current
            one and the n_frames - 1 before it); must match the value the
            network was trained with (see Conv2dNetTrainer; it is checked
            against the checkpoint's stats file). Until n_frames maps have
            arrived, the missing ones repeat the first.
        head_type : str, optional
            'pooled' (default) or 'spatial': the network's regression head
            (see UNetRegressor); must match the value the network was
            trained with (it is checked against the checkpoint's stats file).
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.nmodes = nmodes
        self.input_channels = input_channels
        self.baseline_offset = baseline_offset
        self.n_frames = n_frames
        self.stacker = FrameStacker(n_frames, np)
        self.model, stats = load_trained_network(
            network_filename, nmodes=nmodes, input_channels=input_channels, n_frames=n_frames,
            channels=channels, depth=depth, dropout=dropout, conv_block_type=conv_block_type,
            head_type=head_type, head_grid=head_grid)
        self.meanp = stats['meanp']
        self.stdp = stats['stdp']
        self.meanmodes = np.asarray(stats['meanmodes'], dtype=float)
        self.stdmodes = np.asarray(stats['stdmodes'], dtype=float)

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

        baseline_in = self.local_inputs['baseline']
        if baseline_in is not None:
            baseline_modes = cpuArray(baseline_in.get_value())[
                self.baseline_offset:self.baseline_offset + self.nmodes]
            preds = preds + baseline_modes

        self.modes.value[:] = self.to_xp(preds, dtype=self.dtype)
        self.modes.generation_time = self.current_time
