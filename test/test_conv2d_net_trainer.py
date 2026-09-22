import io
import os
import json
import tempfile
import unittest
import unittest.mock
import contextlib

import specula
specula.init(0)  # Default target device

import numpy as np

from specula import cpuArray
from specula.base_value import BaseValue

# torch is an optional dependency (see pyproject.toml's "nn" extra): skip
# every test in this module rather than failing collection when it's absent.
try:
    import torch
    from specula.lib.efficient_u_net import UNetRegressor, SpatialRegressionHead
    from specula.processing_objects.conv2d_net_trainer import (
        Conv2dNetTrainer,
        EarlyStopping,
        to_torch,
    )
    from specula.lib.nn_training_diagnostics import (
        TrainingDiagnostics,
        effective_dims,
    )
    from specula.processing_objects.conv2d_net_tester import Conv2dNetTester
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


NMODES = 20
CHANNELS = 4
DEPTH = 2
H = W = 32
BATCH = 8


def build_trainer(tmp_dir, network_name='testnet.pth', nmodes=NMODES, channels=CHANNELS,
                   depth=DEPTH, epoch_len=1, patience=600, val_split=0.25,
                   load_from_file=False, conv_block_type=0, input_channels=1,
                   norm_alpha=0.001, loss_delta=20.0, diag_interval=10, target_device_idx=-1,
                   **kwargs):
    network_filename = os.path.join(tmp_dir, network_name)
    return Conv2dNetTrainer(
        network_filename=network_filename,
        nmodes=nmodes,
        channels=channels,
        depth=depth,
        epoch_len=epoch_len,
        patience=patience,
        val_split=val_split,
        load_from_file=load_from_file,
        conv_block_type=conv_block_type,
        input_channels=input_channels,
        norm_alpha=norm_alpha,
        loss_delta=loss_delta,
        diag_interval=diag_interval,
        target_device_idx=target_device_idx,
        **kwargs,
    )


def feed_batch(trainer, batch=BATCH, nmodes=NMODES, h=H, w=W, seed=0):
    rng = np.random.default_rng(seed)
    x_val = BaseValue(value=rng.standard_normal((batch, 2, h, w)).astype(np.float32))
    y_val = BaseValue(value=rng.standard_normal((batch, nmodes + 1)).astype(np.float32))
    x_val.generation_time = x_val.seconds_to_t(1)
    y_val.generation_time = y_val.seconds_to_t(1)
    trainer.inputs['input_2d_batch'].set(x_val)
    trainer.inputs['labels'].set(y_val)
    trainer.check_ready(1)


def feed_batch_with_baseline(trainer, labels, baseline, batch=BATCH, h=H, w=W, seed=0):
    """Like feed_batch(), but with explicit labels/baseline arrays
    (shape (batch, n)) instead of random ones, and a 'baseline' input."""
    rng = np.random.default_rng(seed)
    x_val = BaseValue(value=rng.standard_normal((batch, 2, h, w)).astype(np.float32))
    y_val = BaseValue(value=labels.astype(np.float32))
    b_val = BaseValue(value=baseline.astype(np.float32))
    t = x_val.seconds_to_t(1)
    x_val.generation_time = t
    y_val.generation_time = t
    b_val.generation_time = t
    trainer.inputs['input_2d_batch'].set(x_val)
    trainer.inputs['labels'].set(y_val)
    trainer.inputs['baseline'].set(b_val)
    trainer.check_ready(1)


def feed_batch_with_gain_mod(trainer, gain_mod, batch=BATCH, nmodes=NMODES, h=H, w=W, seed=0):
    """Like feed_batch(), but with an explicit 'gain_mod' input (shape
    (batch,) or (batch, 1)) connected alongside input_2d_batch/labels."""
    rng = np.random.default_rng(seed)
    x_val = BaseValue(value=rng.standard_normal((batch, 2, h, w)).astype(np.float32))
    y_val = BaseValue(value=rng.standard_normal((batch, nmodes + 1)).astype(np.float32))
    g_val = BaseValue(value=np.asarray(gain_mod, dtype=np.float32).reshape(batch, 1))
    t = x_val.seconds_to_t(1)
    x_val.generation_time = t
    y_val.generation_time = t
    g_val.generation_time = t
    trainer.inputs['input_2d_batch'].set(x_val)
    trainer.inputs['labels'].set(y_val)
    trainer.inputs['gain_mod'].set(g_val)
    trainer.check_ready(1)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestNetworkFilenameHandling(unittest.TestCase):

    def test_pth_is_appended_without_extension(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, network_name='mynet', nmodes=3, depth=1)
            self.assertEqual(trainer.network_filename, os.path.join(d, 'mynet.pth'))
            self.assertEqual(trainer.stats_filename, os.path.join(d, 'mynet_stats.json'))

    def test_filename_is_used_verbatim(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, network_name='mynet.pt', nmodes=3, depth=1)
            self.assertEqual(trainer.network_filename, os.path.join(d, 'mynet.pt'))


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestConv2dNetTrainerLoading(unittest.TestCase):

    def _save_checkpoint(self, path, stats_path, nmodes=NMODES, channels=CHANNELS, depth=DEPTH):
        model = UNetRegressor(
            input_channels=1, output_size=nmodes, base_channels=channels,
            dropout_level=0.01, conv_block_type=0, depth=depth,
        )
        torch.save(model.state_dict(), path)
        stats = {
            'meanp': 0.5, 'stdp': 1.5,
            'meanmodes': [0.1] * nmodes, 'stdmodes': [1.1] * nmodes,
            'nmodes': nmodes,
        }
        with open(stats_path, 'w') as f:
            json.dump(stats, f)
        return model

    def test_missing_checkpoint_falls_back_to_new_model(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, network_name='nofile_m20_ch4_dp0.010.pth',
                                     load_from_file=True)
            # Falls back gracefully: a fresh model is created, no stats loaded.
            self.assertIsNone(trainer.meanp)
            self.assertIsNone(trainer.stdp)
            self.assertIsInstance(trainer.model, torch.nn.Module)

    def test_successful_load_restores_weights_and_stats(self):
        with tempfile.TemporaryDirectory() as d:
            # filename already tagged so Conv2dNetTrainer won't rewrite it
            net_path = os.path.join(d, 'existing_m20_ch4_dp0.010.pth')
            stats_path = net_path.replace('.pth', '_stats.json')
            source_model = self._save_checkpoint(net_path, stats_path)

            trainer = build_trainer(d, network_name='existing_m20_ch4_dp0.010.pth',
                                     load_from_file=True)

            self.assertEqual(trainer.meanp, 0.5)
            self.assertEqual(trainer.stdp, 1.5)
            # Regression: meanmodes/stdmodes must be plain xp arrays (matching
            # how trigger() uses them), not torch tensors -- a previous bug
            # stored them as torch.tensor(...), which this test's old
            # trainer.meanmodes.numpy() call happened to paper over.
            self.assertNotIsInstance(trainer.meanmodes, torch.Tensor)
            self.assertNotIsInstance(trainer.stdmodes, torch.Tensor)
            np.testing.assert_allclose(trainer.meanmodes, [0.1] * NMODES)
            np.testing.assert_allclose(trainer.stdmodes, [1.1] * NMODES)

            loaded_state = trainer._inner_model().state_dict()
            source_state = source_model.state_dict()
            for key in source_state:
                torch.testing.assert_close(loaded_state[key].cpu(), source_state[key].cpu())

    def test_successful_load_preserves_stats_across_first_trigger(self):
        # The loaded normalization stats must be updated via the same
        # slow EMA (alpha=0.001) as any other trigger(), not silently
        # replaced by a fresh from-scratch computation on just this batch.
        with tempfile.TemporaryDirectory() as d:
            net_path = os.path.join(d, 'existing_m20_ch4_dp0.010.pth')
            stats_path = net_path.replace('.pth', '_stats.json')
            self._save_checkpoint(net_path, stats_path)

            trainer = build_trainer(d, network_name='existing_m20_ch4_dp0.010.pth',
                                     load_from_file=True)
            loaded_meanmodes = np.array(trainer.meanmodes, copy=True)

            feed_batch(trainer, seed=7)
            trainer.trigger()

            # alpha=0.001 per trigger: one step can only nudge the mean by a
            # tiny amount, nowhere near a full from-scratch recompute (which,
            # for a standard-normal batch, would land near 0, i.e. a jump of
            # ~0.1 away from the loaded value -- two orders of magnitude
            # bigger than what one EMA step can produce).
            np.testing.assert_allclose(trainer.meanmodes, loaded_meanmodes, atol=0.01)

    def test_checkpoint_of_a_different_network_raises_instead_of_being_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            net_path = os.path.join(d, 'existing.pth')
            self._save_checkpoint(net_path, net_path.replace('.pth', '_stats.json'), depth=DEPTH + 1)
            with self.assertRaises(RuntimeError):
                build_trainer(d, network_name='existing.pth', load_from_file=True)

    def test_checkpoint_without_stats_file_warns_and_sets_none(self):
        with tempfile.TemporaryDirectory() as d:
            net_path = os.path.join(d, 'existing_m20_ch4_dp0.010.pth')
            model = UNetRegressor(
                input_channels=1, output_size=NMODES, base_channels=CHANNELS,
                dropout_level=0.01, conv_block_type=0, depth=DEPTH,
            )
            torch.save(model.state_dict(), net_path)
            # no stats file written on purpose

            trainer = build_trainer(d, network_name='existing_m20_ch4_dp0.010.pth',
                                     load_from_file=True)
            self.assertIsNone(trainer.meanp)
            self.assertIsNone(trainer.stdp)
            self.assertIsNone(trainer.meanmodes)
            self.assertIsNone(trainer.stdmodes)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestConv2dNetTrainerDevice(unittest.TestCase):

    def test_device_matches_cuda_availability(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d)
            expected = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.assertEqual(trainer.device.type, expected)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestNormAlpha(unittest.TestCase):
    """norm_alpha controls how fast meanp/stdp/meanmodes/stdmodes adapt on
    every trigger() after the first; it must default to the historical
    hardcoded 0.001, be validated, and actually be used in the EMA update."""

    def test_default_matches_historical_value(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d)
            self.assertEqual(trainer.norm_alpha, 0.001)

    def test_rejects_out_of_range_values(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                build_trainer(d, norm_alpha=0.0)
            with self.assertRaises(ValueError):
                build_trainer(d, norm_alpha=1.5)
            with self.assertRaises(ValueError):
                build_trainer(d, norm_alpha=-0.1)

    def test_accepts_boundary_value_one(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, norm_alpha=1.0)
            self.assertEqual(trainer.norm_alpha, 1.0)

    def test_custom_alpha_is_used_in_ema_update(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, norm_alpha=0.5)
            feed_batch(trainer, seed=1)
            trainer.trigger()  # first trigger: initializes stats directly
            meanmodes_after_first = np.array(trainer.meanmodes, copy=True)

            feed_batch(trainer, seed=2)
            trainer.trigger()  # second trigger: EMA update with alpha=0.5

            # Recompute the expected post-update value independently, using
            # the exact same batch the trainer just saw. feed_batch() draws
            # input_2d_batch before labels from the same RNG, so that draw
            # must be replicated here too to land on the same labels.
            rng = np.random.default_rng(2)
            _ = rng.standard_normal((BATCH, 2, H, W))
            labels = rng.standard_normal((BATCH, NMODES + 1)).astype(np.float32)
            fresh_batch_mean = np.mean(labels[:, 1:NMODES + 1], axis=0)
            expected = 0.5 * meanmodes_after_first + 0.5 * fresh_batch_mean

            np.testing.assert_allclose(trainer.meanmodes, expected, rtol=1e-4)

    def test_higher_alpha_adapts_faster_than_lower_alpha(self):
        with tempfile.TemporaryDirectory() as d:
            slow = build_trainer(d, network_name='slow.pth', norm_alpha=0.001)
            fast = build_trainer(d, network_name='fast.pth', norm_alpha=0.5)

            feed_batch(slow, seed=1)
            slow.trigger()
            feed_batch(fast, seed=1)
            fast.trigger()
            initial = np.array(slow.meanmodes, copy=True)
            np.testing.assert_allclose(slow.meanmodes, fast.meanmodes)  # same first batch

            feed_batch(slow, seed=2)
            slow.trigger()
            feed_batch(fast, seed=2)
            fast.trigger()

            slow_shift = np.abs(slow.meanmodes - initial).sum()
            fast_shift = np.abs(fast.meanmodes - initial).sum()
            self.assertGreater(fast_shift, slow_shift)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestLossDelta(unittest.TestCase):
    """loss_delta is the Huber-loss transition point (physical mode
    units); it must default to 20.0, be validated, and actually be passed
    through to the underlying nn.HuberLoss."""

    def test_default_matches_documented_value(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d)
            self.assertEqual(trainer.loss_delta, 20.0)

    def test_rejects_non_positive_values(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                build_trainer(d, loss_delta=0.0)
            with self.assertRaises(ValueError):
                build_trainer(d, loss_delta=-5.0)

    def test_custom_delta_is_used_by_loss_fn(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, loss_delta=5.0)
            self.assertIsInstance(trainer.loss_fn, torch.nn.HuberLoss)
            self.assertEqual(trainer.loss_fn.delta, 5.0)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestEarlyStopping(unittest.TestCase):

    def test_first_call_never_stops(self):
        es = EarlyStopping(patience=2, min_delta=1e-3)
        self.assertFalse(es(1.0))

    def test_improvement_resets_counter(self):
        es = EarlyStopping(patience=2, min_delta=1e-3)
        es(1.0)
        self.assertFalse(es(0.5))   # improvement
        self.assertFalse(es(0.9))   # worse, counter=1
        self.assertFalse(es(0.4))   # improvement, counter reset
        self.assertFalse(es(0.9))   # worse, counter=1

    def test_stops_after_patience_exceeded(self):
        es = EarlyStopping(patience=2, min_delta=1e-3)
        es(1.0)                     # baseline
        self.assertFalse(es(1.0))   # no improvement, counter=1
        self.assertTrue(es(1.0))    # no improvement, counter=2 >= patience


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestPhysicalSpaceLoss(unittest.TestCase):
    """The training loss must be a plain (unweighted) Huber loss computed
    in physical (de-normalized) mode units: preds*stdmodes+meanmodes vs.
    the true physical labels -- not the old per-mode-weighted MSE in
    normalized/z-scored space, and not a plain (outlier-sensitive) MSE in
    physical units either (see loss_delta's docstring)."""

    class _ConstantOutputModel(torch.nn.Module):
        """Always outputs zeros in normalized space, regardless of input,
        so the resulting physical-space loss can be computed by hand."""
        def __init__(self, nmodes):
            super().__init__()
            self.nmodes = nmodes
            # A dummy trainable parameter, multiplied by zero, so the
            # output still has a grad_fn (loss.backward() requires this)
            # while never actually changing the (constant) output value.
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            zeros = torch.zeros(x.shape[0], self.nmodes, dtype=x.dtype)
            return zeros + 0.0 * self.dummy

    def test_loss_is_plain_huber_in_physical_units(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1, val_split=0.25)
            trainer.model = self._ConstantOutputModel(NMODES)
            trainer.device = torch.device('cpu')

            rng = np.random.default_rng(0)
            x_val = BaseValue(value=rng.standard_normal((BATCH, 2, H, W)).astype(np.float32))
            y_val = BaseValue(value=rng.standard_normal((BATCH, NMODES + 1)).astype(np.float32))
            labels = cpuArray(y_val.value)[:, 1:NMODES + 1]  # label_offset=1 default
            t = x_val.seconds_to_t(1)
            x_val.generation_time = t
            y_val.generation_time = t
            trainer.inputs['input_2d_batch'].set(x_val)
            trainer.inputs['labels'].set(y_val)
            trainer.check_ready(1)

            torch.manual_seed(1234)
            trainer.trigger()

            # Reproduce the same train/val split trigger() computed, to
            # know exactly which rows the training loss was taken over.
            torch.manual_seed(1234)
            perm = torch.randperm(BATCH).numpy()
            n_val = max(1, int(BATCH * 0.25))
            n_train = BATCH - n_val
            train_idx = perm[:n_train]

            # First trigger: meanmodes/stdmodes are computed directly from
            # the full physical-space batch of labels.
            meanmodes = cpuArray(trainer.meanmodes)
            preds_physical = np.broadcast_to(meanmodes, (n_train, NMODES))
            diff = preds_physical - labels[train_idx]
            abs_diff = np.abs(diff)
            delta = trainer.loss_delta
            # nn.HuberLoss's own formula (mean reduction): quadratic below
            # delta, linear beyond it -- see loss_delta's docstring.
            per_entry = np.where(abs_diff < delta,
                                  0.5 * diff ** 2,
                                  delta * (abs_diff - 0.5 * delta))
            expected_loss = np.mean(per_entry)

            self.assertAlmostEqual(trainer.loss, expected_loss, places=4)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestConv2dNetTrainerTrigger(unittest.TestCase):

    def test_trigger_noop_when_inputs_not_set(self):
        # Exercise trigger()'s defensive "no data yet" branch directly:
        # going through check_ready()/setup() would raise, since the inputs
        # are declared as non-optional.
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d)
            trainer.local_inputs['input_2d_batch'] = None
            trainer.local_inputs['labels'] = None
            trainer.trigger()
            self.assertEqual(trainer.step_count, 0)

    def test_trigger_noop_when_should_stop(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d)
            feed_batch(trainer)
            trainer.should_stop = True
            trainer.trigger()
            self.assertEqual(trainer.step_count, 0)

    def test_input_channels_two_uses_raw_multichannel_input(self):
        # With input_channels=2, the buffered (B, 2, H, W) value must be fed
        # to the network as-is (two input channels), not collapsed into a
        # single-channel product as with the default input_channels=1.
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1, input_channels=2)
            self.assertEqual(trainer._inner_model().encoders[0].block[0].in_channels, 2)

            feed_batch(trainer)
            trainer.trigger()
            self.assertEqual(trainer.step_count, 1)

    def test_input_channels_one_takes_the_product_of_a_two_channel_value(self):
        # Historical convention (the petal configs): a (B, 2, H, W) buffered
        # value whose two channels (e.g. amplitude and phase) are multiplied
        # into the single-channel network input.
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1, input_channels=1)
            self.assertEqual(trainer._inner_model().encoders[0].block[0].in_channels, 1)

            rng = np.random.default_rng(0)
            value = rng.standard_normal((BATCH, 2, H, W)).astype(np.float32)
            expected = value[:, 1] * value[:, 0]
            x_val = BaseValue(value=value)
            y_val = BaseValue(value=rng.standard_normal((BATCH, NMODES + 1)).astype(np.float32))
            x_val.generation_time = y_val.generation_time = x_val.seconds_to_t(1)
            trainer.inputs['input_2d_batch'].set(x_val)
            trainer.inputs['labels'].set(y_val)
            trainer.check_ready(1)
            trainer.trigger()

            self.assertEqual(trainer.step_count, 1)
            self.assertAlmostEqual(float(trainer.meanp), float(np.mean(expected)), places=4)

    def test_input_channels_one_uses_a_single_map_per_frame_as_is(self):
        # A buffered value with one map per frame and no channel axis
        # (Slopes2D on slopes-from-intensity slopes): used unchanged, with
        # the channel axis added for the network.
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1, input_channels=1)
            rng = np.random.default_rng(0)
            value = rng.standard_normal((BATCH, H, W)).astype(np.float32)
            x_val = BaseValue(value=value)
            y_val = BaseValue(value=rng.standard_normal((BATCH, NMODES + 1)).astype(np.float32))
            x_val.generation_time = y_val.generation_time = x_val.seconds_to_t(1)
            trainer.inputs['input_2d_batch'].set(x_val)
            trainer.inputs['labels'].set(y_val)
            trainer.check_ready(1)
            trainer.trigger()

            self.assertEqual(trainer.step_count, 1)
            self.assertAlmostEqual(float(trainer.meanp), float(np.mean(value)), places=4)
            self.assertEqual(trainer.val_inputs.shape[1:], (1, H, W))

    def test_trigger_runs_one_training_step(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            feed_batch(trainer)
            trainer.trigger()

            self.assertEqual(trainer.step_count, 1)
            self.assertIsNotNone(trainer.val_inputs)
            self.assertIsNotNone(trainer.meanmodes)
            self.assertTrue(np.isfinite(trainer.loss))
            # Saving is periodic (every 10 steps), not gated on improvement
            # -- see test_model_saved_every_10_steps_not_on_improvement.
            self.assertFalse(os.path.isfile(trainer.network_filename))

    def test_model_saved_every_10_steps_not_on_improvement(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            for i in range(9):
                feed_batch(trainer, seed=i)
                trainer.trigger()
            self.assertEqual(trainer.step_count, 9)
            self.assertFalse(os.path.isfile(trainer.network_filename))

            feed_batch(trainer, seed=9)
            trainer.trigger()
            self.assertEqual(trainer.step_count, 10)
            self.assertTrue(os.path.isfile(trainer.network_filename))
            self.assertTrue(os.path.isfile(trainer.stats_filename))

    def test_saved_stats_min_loss_reflects_this_saves_eval_loss_not_best_ever(self):
        # Regression test: after switching from "save on improvement" to
        # "save every 10 steps", the persisted min_loss must reflect
        # *this save's* eval_loss (matching the weights actually being
        # written), not self.min_loss (best-ever seen, which can now
        # refer to a step other than the one just saved). Otherwise a
        # resumed run's first evaluation looks like an unexplained
        # regression against a number these exact weights never produced.
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            # Pretend an impossibly low loss was already seen (as if from
            # a different, no-longer-current set of weights) -- it must
            # not leak into what gets saved for the weights saved here.
            trainer.min_loss = -100.0

            for i in range(10):
                feed_batch(trainer, seed=i)
                trainer.trigger()

            with open(trainer.stats_filename) as f:
                stats = json.load(f)
            # A real eval_loss is an ordinary positive-ish number; -100
            # would only appear if self.min_loss leaked into the save.
            self.assertGreater(stats['min_loss'], -100.0)

    def test_trigger_creates_missing_parent_directory_before_saving(self):
        with tempfile.TemporaryDirectory() as d:
            nested_dir = os.path.join(d, 'does', 'not', 'exist', 'yet')
            trainer = build_trainer(nested_dir, epoch_len=1)
            self.assertFalse(os.path.isdir(nested_dir))

            for i in range(10):
                feed_batch(trainer, seed=i)
                trainer.trigger()

            self.assertTrue(os.path.isfile(trainer.network_filename))
            self.assertTrue(os.path.isfile(trainer.stats_filename))

    def test_second_trigger_accumulates_validation_set(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1, val_split=0.25)
            feed_batch(trainer, seed=1)
            trainer.trigger()
            trainer.post_trigger()
            n_val_after_first = trainer.val_inputs.shape[0]

            feed_batch(trainer, seed=2)
            trainer.trigger()
            trainer.post_trigger()
            n_val_after_second = trainer.val_inputs.shape[0]

            self.assertGreater(n_val_after_second, n_val_after_first)

    def test_post_trigger_sets_loss_output(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            feed_batch(trainer)
            trainer.trigger()
            trainer.post_trigger()
            self.assertAlmostEqual(
                trainer.outputs['loss'].value, trainer.loss, places=5)

    def test_post_trigger_before_any_trigger_does_not_raise(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d)
            # check_ready() marks inputs as changed; post_trigger() itself
            # must tolerate the missing 'loss' attribute (no trigger() yet).
            feed_batch(trainer)
            trainer.post_trigger()

    def test_finalize_runs_without_error(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            feed_batch(trainer)
            trainer.trigger()
            trainer.post_trigger()

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                trainer.finalize()
            self.assertIn('Training complete', buf.getvalue())

    def test_logs_train_val_loss_every_100_steps(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            feed_batch(trainer)
            trainer.step_count = 99  # this trigger() call increments it to 100

            with self.assertLogs(trainer.logger.logger, level='INFO') as cm:
                trainer.trigger()

            self.assertEqual(trainer.step_count, 100)
            self.assertTrue(any('modal_analysis ground truth' in msg for msg in cm.output))
            self.assertTrue(any('train=' in msg and 'val=' in msg for msg in cm.output))

    def test_does_not_log_loss_on_non_multiple_of_100_steps(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            feed_batch(trainer)
            trainer.step_count = 50  # this trigger() call increments it to 51

            with unittest.mock.patch.object(trainer.logger, 'info') as mock_info:
                trainer.trigger()

            mock_info.assert_not_called()


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestResidualLearning(unittest.TestCase):
    """When a 'baseline' input is connected, the network must be trained
    on labels - baseline instead of the raw labels."""

    def test_baseline_input_is_optional_and_unconnected_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d)
            self.assertTrue(trainer.inputs['baseline'].optional)

    def test_zero_residual_when_baseline_equals_labels(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            rng = np.random.default_rng(0)
            labels = rng.standard_normal((BATCH, NMODES + 1))
            # baseline equal to the true labels over the sliced mode range
            # (label_offset=1 by default): residual must be exactly zero.
            baseline = np.zeros((BATCH, NMODES))
            baseline[:] = labels[:, 1:NMODES + 1]

            feed_batch_with_baseline(trainer, labels, baseline)
            trainer.trigger()

            # meanmodes/stdmodes are computed directly from the (residual)
            # training target, independent of the network itself.
            np.testing.assert_allclose(trainer.meanmodes, np.zeros(NMODES), atol=1e-6)
            np.testing.assert_allclose(trainer.stdmodes, np.full(NMODES, 1e-8), atol=1e-9)

    def test_nonzero_residual_matches_labels_minus_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            rng = np.random.default_rng(1)
            labels = rng.standard_normal((BATCH, NMODES + 1))
            baseline = rng.standard_normal((BATCH, NMODES))

            feed_batch_with_baseline(trainer, labels, baseline)
            trainer.trigger()

            expected_residual = labels[:, 1:NMODES + 1] - baseline
            np.testing.assert_allclose(
                trainer.meanmodes, np.mean(expected_residual, axis=0), atol=1e-5)

    def test_baseline_offset_selects_correct_columns(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1, nmodes=3)
            trainer.baseline_offset = 2
            rng = np.random.default_rng(2)
            labels = rng.standard_normal((BATCH, 4))
            # baseline has extra leading columns that must be skipped
            baseline_full = rng.standard_normal((BATCH, 5))

            feed_batch_with_baseline(trainer, labels, baseline_full, batch=BATCH)
            trainer.trigger()

            expected_residual = labels[:, 1:4] - baseline_full[:, 2:5]
            np.testing.assert_allclose(
                trainer.meanmodes, np.mean(expected_residual, axis=0), atol=1e-5)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestGainModFiltering(unittest.TestCase):
    """When a 'gain_mod' input is connected, samples where the loop wasn't
    at full gain must be dropped from the batch before training (mirrors
    Conv2dNetTester's gain_mod_threshold)."""

    def test_gain_mod_filters_out_below_threshold_samples(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            feed_batch_with_gain_mod(trainer, gain_mod=[1.0, 1.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
            trainer.trigger()

            # Default gain_mod_threshold=1.0: only the 6 full-gain samples
            # are kept; the zero and the partial-gain one are dropped.
            self.assertIsNotNone(trainer.meanmodes)
            self.assertEqual(trainer.meanmodes.shape, (NMODES,))

    def test_gain_mod_all_below_threshold_skips_trigger(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            feed_batch_with_gain_mod(trainer, gain_mod=[0.0] * BATCH)
            trainer.trigger()

            self.assertEqual(trainer.step_count, 0)
            self.assertIsNone(trainer.meanmodes)

    def test_gain_mod_unconnected_keeps_all_samples(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            feed_batch(trainer)
            trainer.trigger()

            self.assertEqual(trainer.step_count, 1)

    def test_gain_mod_custom_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            trainer.gain_mod_threshold = 0.5
            feed_batch_with_gain_mod(trainer, gain_mod=[1.0, 0.5, 0.4, 0.0, 1.0, 0.5, 0.4, 0.0])
            trainer.trigger()

            # 1.0 and 0.5 both pass >= 0.5 -> 4 of the 8 samples kept.
            self.assertEqual(trainer.step_count, 1)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestTrainingDiagnostics(unittest.TestCase):

    def _run_steps(self, trainer, n, batch=32):
        # 32 per batch (24 training samples each) so that after 10 steps the
        # ridge baseline has enough samples to be fitted.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for i in range(n):
                feed_batch(trainer, batch=batch, seed=i)
                trainer.trigger()
        return buf.getvalue()

    def test_disabled_when_interval_is_zero(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, diag_interval=0)
            self.assertIsNone(trainer.diag)
            out = self._run_steps(trainer, 10)
            self.assertNotIn('[diag]', out)
            self.assertFalse(any(f.endswith('_diag.jsonl') for f in os.listdir(d)))

    def test_writes_one_record_per_interval_without_errors(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d)
            out = self._run_steps(trainer, 10)
            self.assertNotIn('[diag] ERROR', out)
            self.assertIn('linear ridge baseline', out)
            self.assertIn('forward in time', out)

            with open(trainer.diag.jsonl_filename) as f:
                records = [json.loads(line) for line in f]
            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertEqual(rec['step'], 10)
            self.assertEqual(len(rec['per_mode']['cnn_fvu']), NMODES)
            for key in ('cnn_fvu_total', 'groups', 'label_dims_99', 'cnn_pred_dims_99',
                        'grad_clipped_fraction', 'window_fresh_loss', 'hints'):
                self.assertIn(key, rec)

    def test_diagnostics_do_not_change_training(self):
        with tempfile.TemporaryDirectory() as d:
            torch.manual_seed(0)
            with_diag = build_trainer(d, network_name='a.pth')
            torch.manual_seed(0)
            without_diag = build_trainer(d, network_name='b.pth', diag_interval=0)
            for trainer in (with_diag, without_diag):
                torch.manual_seed(1)
                self._run_steps(trainer, 10)
            self.assertAlmostEqual(with_diag.loss, without_diag.loss, places=5)

    def test_out_of_band_label_residual_is_tracked(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d)
            rng = np.random.default_rng(0)
            # label_offset=1: columns 1..NMODES are trained on, the 5 extra
            # ones after them are the out-of-band residual.
            x_val = BaseValue(value=rng.standard_normal((BATCH, 2, H, W)).astype(np.float32))
            y_val = BaseValue(value=rng.standard_normal((BATCH, NMODES + 6)).astype(np.float32))
            t = x_val.seconds_to_t(1)
            x_val.generation_time = y_val.generation_time = t
            trainer.inputs['input_2d_batch'].set(x_val)
            trainer.inputs['labels'].set(y_val)
            trainer.check_ready(1)
            trainer.trigger()

            expected = np.sqrt(np.sum(cpuArray(y_val.value)[:, NMODES + 1:] ** 2, axis=1))
            self.assertEqual(trainer.diag.val_oob.shape[0], trainer.val_targets.shape[0])
            for v in trainer.diag.val_oob.numpy():
                self.assertTrue(np.any(np.isclose(expected, v, rtol=1e-5)))

    def test_ridge_baseline_recovers_a_linear_map(self):
        rng = np.random.default_rng(0)
        d_in, k = 40, 6
        w = torch.from_numpy(rng.standard_normal((d_in, k)))
        diag = TrainingDiagnostics('t', nmodes=k, max_val=1000, ridge_samples=4000,
                                   jsonl_filename=os.devnull,
                                   clip_value=1.0, device=torch.device('cpu'))
        x_train = torch.from_numpy(rng.standard_normal((500, d_in))).float()
        x_val = torch.from_numpy(rng.standard_normal((100, d_in))).float()
        y_train = (x_train.double() @ w).float()
        y_val = (x_val.double() @ w).float()
        diag.add_batch(x_train, y_train, x_val, torch.zeros(100),
                       torch.ones(500), torch.zeros(0))
        mse, _ = diag._ridge_val(y_val)
        fvu = mse.sum() / y_val.double().var(0, unbiased=False).sum()
        self.assertLess(float(fvu), 1e-3)

        # Forward in time: fitted only on the first batch, tested on a
        # second, later one.
        x_next = torch.from_numpy(rng.standard_normal((200, d_in))).float()
        y_next = (x_next.double() @ w).float()
        diag.add_batch(x_next, y_next, x_val[:0], torch.zeros(0),
                       torch.ones(200), torch.zeros(0))
        fwd_mse, fwd_t = diag._ridge_forward()
        self.assertEqual(fwd_t.shape[0], 200)
        fwd_fvu = fwd_mse.sum() / fwd_t.var(0, unbiased=False).sum()
        self.assertLess(float(fwd_fvu), 1e-3)

    def test_effective_dims_counts_rank(self):
        rng = np.random.default_rng(0)
        x = torch.from_numpy(rng.standard_normal((200, 3)) @ rng.standard_normal((3, 10)))
        self.assertEqual(effective_dims(x), 3)

def linear_batch(rng, w, n, side=8):
    """Batch whose labels (after a placeholder column, label_offset=1) are an
    exact linear function of the in-pupil pixels of a (n, 2, side, side)
    input."""
    yy, xx = np.mgrid[:side, :side] - (side - 1) / 2
    pupil = (xx ** 2 + yy ** 2) < (side / 2) ** 2
    x = np.zeros((n, 2, side, side), np.float32)
    x[:, :, pupil] = rng.standard_normal((n, 2, int(pupil.sum())))
    modes = x[:, :, pupil].reshape(n, -1) @ w + 3.0
    labels = np.concatenate([np.zeros((n, 1)), modes], axis=1).astype(np.float32)
    return x, labels


def feed_arrays(trainer, x, labels, t_seconds=1):
    x_val = BaseValue(value=x)
    y_val = BaseValue(value=labels)
    t = x_val.seconds_to_t(t_seconds)
    x_val.generation_time = y_val.generation_time = t
    trainer.inputs['input_2d_batch'].set(x_val)
    trainer.inputs['labels'].set(y_val)
    trainer.check_ready(t)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestInitSamplesAndFrozenNorm(unittest.TestCase):

    def _build(self, d, **kwargs):
        opts = dict(input_channels=2, freeze_norm=True, init_samples=400, diag_interval=0)
        opts.update(kwargs)
        return build_trainer(d, **opts)

    def _pupil_weights(self, rng, side=8):
        yy, xx = np.mgrid[:side, :side] - (side - 1) / 2
        n_pix = 2 * int(((xx ** 2 + yy ** 2) < (side / 2) ** 2).sum())
        return rng.standard_normal((n_pix, NMODES)) * 10

    def test_init_phase_collects_samples_without_training(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d)
            rng = np.random.default_rng(0)
            w = self._pupil_weights(rng)
            with contextlib.redirect_stdout(io.StringIO()):
                for i in range(3):
                    feed_arrays(trainer, *linear_batch(rng, w, 100), t_seconds=i + 1)
                    trainer.trigger()
                    self.assertEqual(trainer.step_count, 0)
                    self.assertIsNone(trainer.meanmodes)
                feed_arrays(trainer, *linear_batch(rng, w, 100), t_seconds=4)
                trainer.trigger()
            self.assertEqual(trainer.step_count, 1)
            self.assertIsNotNone(trainer.meanmodes)

    def test_norm_is_estimated_from_all_init_samples(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d)
            rng = np.random.default_rng(1)
            w = self._pupil_weights(rng)
            all_labels = []
            with contextlib.redirect_stdout(io.StringIO()):
                for i in range(4):
                    x, labels = linear_batch(rng, w, 100)
                    labels[:, 1:] += 50.0 * i  # every batch has a different mean
                    all_labels.append(labels[:, 1:NMODES + 1])
                    feed_arrays(trainer, x, labels, t_seconds=i + 1)
                    trainer.trigger()
            np.testing.assert_allclose(cpuArray(trainer.meanmodes),
                                       np.concatenate(all_labels).mean(axis=0), rtol=1e-4)

    def test_freeze_norm_keeps_stats_fixed(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, init_samples=0, norm_alpha=0.5)
            rng = np.random.default_rng(2)
            w = self._pupil_weights(rng)
            with contextlib.redirect_stdout(io.StringIO()):
                feed_arrays(trainer, *linear_batch(rng, w, 100), t_seconds=1)
                trainer.trigger()
                frozen = (float(trainer.meanp), cpuArray(trainer.stdmodes).copy())
                for i in range(3):
                    x, labels = linear_batch(rng, w, 100)
                    feed_arrays(trainer, x * 5, labels * 5, t_seconds=i + 2)
                    trainer.trigger()
            self.assertEqual(float(trainer.meanp), frozen[0])
            np.testing.assert_array_equal(cpuArray(trainer.stdmodes), frozen[1])

    def test_checkpoint_resumes_with_its_stats_and_loads_in_tester(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, network_name='net.pth')
            rng = np.random.default_rng(4)
            w = self._pupil_weights(rng)
            with contextlib.redirect_stdout(io.StringIO()):
                t = 0
                while trainer.step_count < 10:  # first periodic save
                    t += 1
                    feed_arrays(trainer, *linear_batch(rng, w, 100), t_seconds=t)
                    trainer.trigger()
                resumed = self._build(d, network_name='net.pth', load_from_file=True)
            # frozen stats restored: no new initialization phase
            np.testing.assert_allclose(cpuArray(resumed.meanmodes), cpuArray(trainer.meanmodes))
            self.assertEqual(resumed.stdp, trainer.stdp)

            tester = Conv2dNetTester(network_filename=trainer.network_filename, nmodes=NMODES,
                                     channels=CHANNELS, depth=DEPTH, input_channels=2,
                                     target_device_idx=-1)
            saved = trainer._inner_model().state_dict()
            for k, v in tester.model.state_dict().items():
                torch.testing.assert_close(v, saved[k].cpu())


def time_tagged_batch(n, offset=0, h=H, w=W, seed=0):
    """Random inputs; labels whose first trained mode (label_offset=1) is
    offset + the row's position in time, to track which rows went where."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2, h, w)).astype(np.float32)
    labels = rng.standard_normal((n, NMODES + 1)).astype(np.float32)
    labels[:, 1] = offset + np.arange(n)
    return x, labels


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestReplayAndGradClip(unittest.TestCase):

    def _build(self, d, **kwargs):
        opts = dict(input_channels=2, replay_size=100, replay_subsample=2, replay_steps=3,
                    replay_batch=4, diag_interval=0)
        opts.update(kwargs)
        return build_trainer(d, **opts)

    def _feed(self, trainer, n=32, offset=0, t_seconds=1):
        with contextlib.redirect_stdout(io.StringIO()):
            feed_arrays(trainer, *time_tagged_batch(n, offset, seed=offset), t_seconds=t_seconds)
            trainer.trigger()

    def test_replay_is_off_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, input_channels=2, diag_interval=0)
            self._feed(trainer)
            self.assertIsNone(trainer.replay_x)
            self.assertEqual(trainer.step_count, 1)

    def test_buffer_gets_every_nth_training_row_in_time_order(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d)
            self._feed(trainer)
            # 32 rows, 8 to validation, 24 training rows -> every 2nd = 12.
            self.assertEqual(trainer.replay_count, 12)
            t = trainer.replay_y[:12, 0].numpy()
            self.assertTrue(np.all(np.diff(t) >= 2))   # time-ordered, subsampled
            val_t = trainer.val_targets[:, 0].cpu().tolist()
            self.assertFalse(set(t.tolist()) & set(val_t))   # no validation rows

    def test_ring_buffer_keeps_the_most_recent_samples(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, replay_size=20)
            for k in range(3):
                self._feed(trainer, offset=100 * k, t_seconds=k + 1)
            self.assertEqual(trainer.replay_count, 20)
            t = trainer.replay_y[:, 0].numpy()
            self.assertGreaterEqual(t.min(), 100)          # batch 0 fully overwritten
            self.assertEqual(int(np.sum(t >= 200)), 12)   # all of batch 2 kept

    def test_each_trigger_takes_replay_steps_minibatch_steps(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, replay_steps=7)
            with unittest.mock.patch.object(trainer.optimizer, 'step',
                                            wraps=trainer.optimizer.step) as step:
                self._feed(trainer)
            self.assertEqual(step.call_count, 7)

    def test_init_samples_seed_the_buffer(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, init_samples=64, freeze_norm=True)
            self._feed(trainer, offset=0, t_seconds=1)
            self.assertEqual(trainer.replay_count, 0)      # still collecting
            self._feed(trainer, offset=100, t_seconds=2)
            # First batch (32 rows, all of it) every 2nd -> 16, plus the
            # current batch's 24 training rows every 2nd -> 12.
            self.assertEqual(trainer.replay_count, 28)
            self.assertEqual(trainer.step_count, 1)

    def test_single_channel_legacy_input_works_with_replay(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, input_channels=1)
            self._feed(trainer)
            self.assertEqual(trainer.step_count, 1)

    def test_lr_decay_per_trigger_with_floor_and_no_plateau_schedule(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, lr_decay=0.5)
            with unittest.mock.patch.object(trainer.plateau_scheduler, 'step') as plateau:
                self._feed(trainer, t_seconds=1)
                self._feed(trainer, offset=100, t_seconds=2)
            self.assertAlmostEqual(trainer.optimizer.param_groups[0]['lr'], 1e-3 * 0.25)
            plateau.assert_not_called()

            floored = self._build(d, network_name='f.pth', lr_decay=0.01)
            for k in range(3):
                self._feed(floored, offset=100 * k, t_seconds=k + 1)
            self.assertAlmostEqual(floored.optimizer.param_groups[0]['lr'], 0.5e-5)

    def test_lr_decay_is_validated(self):
        with tempfile.TemporaryDirectory() as d:
            for bad in (0.0, 1.0, 1.5):
                with self.assertRaises(ValueError):
                    self._build(d, lr_decay=bad)

    def test_grad_clip_default_and_disabled(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(build_trainer(d, diag_interval=0).grad_clip_value, 1.0)
            trainer = self._build(d, grad_clip_value=0)
            self.assertEqual(trainer.grad_clip_value, float('inf'))
            with unittest.mock.patch('torch.nn.utils.clip_grad_norm_',
                                     wraps=torch.nn.utils.clip_grad_norm_) as clip:
                self._feed(trainer)
            self.assertTrue(clip.called)
            self.assertTrue(all(c.args[1] == float('inf') for c in clip.call_args_list))


def feed_with_gain(trainer, gain, offset=0, t_seconds=1):
    """Time-tagged batch (see time_tagged_batch) with an explicit per-frame
    gain_mod sequence."""
    gain = np.asarray(gain, dtype=np.float32)
    x, labels = time_tagged_batch(len(gain), offset, seed=offset)
    x_val, y_val = BaseValue(value=x), BaseValue(value=labels)
    g_val = BaseValue(value=gain.reshape(-1, 1))
    t = x_val.seconds_to_t(t_seconds)
    x_val.generation_time = y_val.generation_time = g_val.generation_time = t
    trainer.inputs['input_2d_batch'].set(x_val)
    trainer.inputs['labels'].set(y_val)
    trainer.inputs['gain_mod'].set(g_val)
    trainer.check_ready(t)
    with contextlib.redirect_stdout(io.StringIO()) as out:
        trainer.trigger()
    return out.getvalue()


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestTransientRegime(unittest.TestCase):

    def _build(self, d, **kwargs):
        opts = dict(input_channels=2, replay_size=1000, replay_subsample=1, replay_steps=2,
                    replay_batch=4, diag_interval=0, gain_mod_threshold=1.0)
        opts.update(kwargs)
        return build_trainer(d, **opts)

    def test_frames_since_reclose_counts_and_carries_over(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d)
            np.testing.assert_array_equal(
                trainer._frames_since_reclose(np.array([1, 1, 0, 0.5, 1, 1, 1.0])), [1, 2, 0, 0, 1, 2, 3])
            np.testing.assert_array_equal(trainer._frames_since_reclose(np.array([1, 0, 1])), [4, 0, 1])

    def test_counter_advances_on_frames_that_are_filtered_out(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d)
            feed_with_gain(trainer, [0.0] * 8)            # nothing kept, trigger returns early
            self.assertEqual(trainer._frames_at_full_gain, 0)
            feed_with_gain(trainer, [1.0] * 8, offset=100, t_seconds=2)
            self.assertEqual(trainer._frames_at_full_gain, 8)

    def test_weight_zero_drops_transient_frames(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, settle_frames=5, transient_weight=0.0)
            feed_with_gain(trainer, [1.0] * 32)
            # frames since full gain are 1..32: only 6..32 (time tags 5..31)
            # are kept, split between training and validation.
            kept = set(trainer.replay_y[:trainer.replay_count, 0].tolist())
            kept |= set(trainer.val_targets[:, 0].cpu().tolist())
            self.assertEqual(kept, set(float(t) for t in range(5, 32)))

    def test_weight_between_zero_and_one_downweights_transient_frames(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, settle_frames=5, transient_weight=0.25)
            feed_with_gain(trainer, [1.0] * 32)
            t = trainer.replay_y[:trainer.replay_count, 0]
            w = trainer.replay_w[:trainer.replay_count]
            torch.testing.assert_close(w[t <= 4], torch.full_like(w[t <= 4], 0.25))
            torch.testing.assert_close(w[t > 4], torch.ones_like(w[t > 4]))

    def test_weighted_loss_is_a_weighted_mean_of_per_sample_huber(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, loss_delta=1.0)
            p = torch.tensor([[0.5, 3.0], [0.0, 0.0]])
            y = torch.zeros(2, 2)
            w = torch.tensor([1.0, 3.0])
            per_sample = torch.tensor([(0.125 + 2.5) / 2, 0.0])
            expected = (w * per_sample).sum() / w.sum()
            self.assertAlmostEqual(trainer._loss(p, y, w, None).item(), expected.item(), places=6)
            self.assertAlmostEqual(trainer._loss(p, y, None, None).item(), trainer.loss_fn(p, y).item(), places=6)

    def test_settle_frames_without_gain_mod_warns_once_and_trains(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, settle_frames=5, transient_weight=0.0)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                for k in range(2):
                    feed_arrays(trainer, *time_tagged_batch(32, 100 * k, seed=k), t_seconds=k + 1)
                    trainer.trigger()
            self.assertEqual(out.getvalue().count("settle_frames needs the 'gain_mod' input"), 1)
            self.assertEqual(trainer.step_count, 2)

    def test_invalid_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for kwargs in (dict(settle_frames=-1), dict(transient_weight=1.5), dict(transient_weight=-0.1)):
                with self.assertRaises(ValueError):
                    self._build(d, **kwargs)

    def test_diagnostics_report_the_transient(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, settle_frames=5, transient_weight=0.5, diag_interval=10)
            out = ''
            for k in range(10):
                gain = [0.0] * 4 + [1.0] * 60 if k % 3 == 0 else [1.0] * 64
                out += feed_with_gain(trainer, gain, offset=100 * k, t_seconds=k + 1)
            self.assertNotIn('[diag] ERROR', out)
            self.assertIn('loop transient', out)
            self.assertIn('settled frames only', out)
            with open(trainer.diag.jsonl_filename) as f:
                rec = json.loads(f.readlines()[-1])
            bins = {tuple(b['frames']): b['n'] for b in rec['transient_bins']}
            self.assertGreater(bins[(1, 10)], 0)
            self.assertEqual(len(rec['fwd_ridge_fvu_settled_by_group']), 2)   # mode groups 0-4, 5-19


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestFrameStackingAndNormLoss(unittest.TestCase):

    def _build(self, d, **kwargs):
        opts = dict(input_channels=2, n_frames=4, replay_size=1000, replay_subsample=1,
                    replay_steps=2, replay_batch=4, diag_interval=0)
        opts.update(kwargs)
        return build_trainer(d, **opts)

    def _feed_time_frames(self, trainer, gain, first=0, t_seconds=1):
        """Frames whose pixels all equal their time index, labels tagged
        with it too (first mode, label_offset=1), and a gain_mod sequence."""
        n = len(gain)
        t = np.arange(first, first + n, dtype=np.float32)
        x = np.broadcast_to(t[:, None, None, None], (n, 2, H, W)).copy()
        labels = np.random.default_rng(first).standard_normal((n, NMODES + 1)).astype(np.float32)
        labels[:, 1] = t
        x_val, y_val = BaseValue(value=x), BaseValue(value=labels)
        g_val = BaseValue(value=np.asarray(gain, dtype=np.float32).reshape(-1, 1))
        ts = x_val.seconds_to_t(t_seconds)
        x_val.generation_time = y_val.generation_time = g_val.generation_time = ts
        trainer.inputs['input_2d_batch'].set(x_val)
        trainer.inputs['labels'].set(y_val)
        trainer.inputs['gain_mod'].set(g_val)
        trainer.check_ready(ts)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.trigger()

    def test_network_input_has_input_channels_times_n_frames(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d)
            inner = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
            self.assertEqual(inner.encoders[0].block[0].in_channels, 8)

    def test_stacks_true_previous_frames_before_filtering(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d)
            # Frames 0-1 at zero gain are dropped as samples, but frames 2-3
            # must still carry them as their history.
            self._feed_time_frames(trainer, [0.0, 0.0] + [1.0] * 30)
            self._feed_time_frames(trainer, [1.0] * 32, first=32, t_seconds=2)
            n = trainer.replay_count
            t = trainer.replay_y[:n, 0]
            stacked_times = trainer.replay_x[:n, ::2, 0, 0]      # one channel per frame
            for time, times in zip(t.tolist(), stacked_times.tolist()):
                self.assertEqual(times, [max(time - k, 0) for k in range(4)])
            self.assertIn(2.0, t.tolist() + trainer.val_targets[:, 0].cpu().tolist())
            self.assertIn(32.0, t.tolist() + trainer.val_targets[:, 0].cpu().tolist())

    def test_n_frames_is_saved_and_the_tester_uses_and_checks_it(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, network_name='stack.pth')
            for k in range(10):   # first periodic save at step 10
                self._feed_time_frames(trainer, [1.0] * 32, first=32 * k, t_seconds=k + 1)
            with open(trainer.stats_filename) as f:
                self.assertEqual(json.load(f)['n_frames'], 4)

            tester = Conv2dNetTester(network_filename=trainer.network_filename, nmodes=NMODES,
                                     channels=CHANNELS, depth=DEPTH, input_channels=2,
                                     n_frames=4, target_device_idx=-1)
            x = BaseValue(value=np.random.default_rng(0).standard_normal((8, 2, H, W)).astype(np.float32))
            y = BaseValue(value=np.random.default_rng(1).standard_normal((8, NMODES + 1)).astype(np.float32))
            x.generation_time = y.generation_time = x.seconds_to_t(1)
            tester.inputs['input_2d_batch'].set(x)
            tester.inputs['labels'].set(y)
            tester.check_ready(1)
            tester.trigger()
            self.assertEqual(tester.count, 8)

            with self.assertRaisesRegex(ValueError, 'n_frames=4'):
                Conv2dNetTester(network_filename=trainer.network_filename, nmodes=NMODES,
                                channels=CHANNELS, depth=DEPTH, input_channels=2,
                                n_frames=1, target_device_idx=-1)

    def test_norm_loss_adds_the_per_mode_normalized_mse(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, loss_delta=1.0, norm_loss_weight=2.0)
            p = torch.tensor([[0.5, 3.0], [1.0, 0.0]])
            y = torch.zeros(2, 2)
            std = torch.tensor([1.0, 10.0])
            huber = torch.nn.functional.huber_loss(p, y, delta=1.0)
            normalized = ((p / std) ** 2).mean()
            self.assertAlmostEqual(trainer._loss(p, y, None, std).item(),
                                   (huber + 2.0 * normalized).item(), places=6)

    def test_reported_train_loss_stays_plain_nm(self):
        with tempfile.TemporaryDirectory() as d:
            plain = self._build(d, network_name='a.pth', replay_size=0, epoch_len=1, n_frames=1)
            weighted = self._build(d, network_name='b.pth', replay_size=0, epoch_len=1, n_frames=1,
                                   norm_loss_weight=1000.0)
            weighted.model.load_state_dict(plain.model.state_dict())
            for trainer in (plain, weighted):
                torch.manual_seed(0)
                self._feed_time_frames(trainer, [1.0] * 32)
            # Same weights, same data, same split: the reported (pre-step)
            # loss is the same nm loss, despite the huge extra term.
            self.assertAlmostEqual(plain.loss, weighted.loss, places=4)

    def test_invalid_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for kwargs in (dict(n_frames=0), dict(norm_loss_weight=-1.0)):
                with self.assertRaises(ValueError):
                    self._build(d, **kwargs)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestToTorchHelper(unittest.TestCase):

    def test_to_torch_converts_numpy_array(self):
        arr = np.arange(6, dtype=np.float64).reshape(2, 3)
        t = to_torch(arr)
        self.assertEqual(t.dtype, torch.float32)
        np.testing.assert_allclose(t.numpy(), arr)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestSpatialHeadTrainerRoundTrip(unittest.TestCase):

    def _train(self, trainer, n_steps=10):
        rng = np.random.default_rng(0)
        for k in range(n_steps):
            x = BaseValue(value=rng.standard_normal((16, 2, H, W)).astype(np.float32))
            y = BaseValue(value=rng.standard_normal((16, NMODES + 1)).astype(np.float32))
            t = x.seconds_to_t(k + 1)
            x.generation_time = y.generation_time = t
            trainer.inputs['input_2d_batch'].set(x)
            trainer.inputs['labels'].set(y)
            trainer.check_ready(t)
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.trigger()

    def _tester(self, filename, head_type):
        return Conv2dNetTester(network_filename=filename, nmodes=NMODES, channels=CHANNELS,
                               depth=3, input_channels=2, head_type=head_type, target_device_idx=-1)

    def test_head_type_is_saved_checked_and_reloaded(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, network_name='spatial.pth', input_channels=2, depth=3,
                                    head_type='spatial', diag_interval=0)
            inner = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
            self.assertIsInstance(inner.regressor, SpatialRegressionHead)
            self._train(trainer)   # first periodic save at step 10
            with open(trainer.stats_filename) as f:
                self.assertEqual(json.load(f)['head_type'], 'spatial')

            tester = self._tester(trainer.network_filename, 'spatial')
            saved = torch.load(trainer.network_filename, map_location='cpu')
            for k, v in tester.model.state_dict().items():
                torch.testing.assert_close(v, saved[k])

            with self.assertRaisesRegex(ValueError, "head_type='pooled'"):
                self._tester(trainer.network_filename, 'pooled')

            # Resuming with the other head must stop, not silently start a
            # new model that would overwrite the checkpoint.
            with self.assertRaisesRegex(ValueError, 'head_type'):
                build_trainer(d, network_name='spatial.pth', input_channels=2, depth=3,
                              load_from_file=True, diag_interval=0)
            resumed = build_trainer(d, network_name='spatial.pth', input_channels=2, depth=3,
                                    head_type='spatial', load_from_file=True, diag_interval=0)
            inner = resumed.model.module if hasattr(resumed.model, 'module') else resumed.model
            for k, v in inner.state_dict().items():
                torch.testing.assert_close(v.cpu(), saved[k])


if __name__ == '__main__':
    unittest.main()
