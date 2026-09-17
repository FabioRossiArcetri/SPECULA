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
    from specula.lib.efficient_u_net import UNetRegressor
    from specula.processing_objects.conv2d_net_trainer import (
        Conv2dNetTrainer,
        EarlyStopping,
        WeightedHuberLoss,
    )
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
                   norm_alpha=0.001, target_device_idx=-1):
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
        target_device_idx=target_device_idx,
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


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestNetworkFilenameHandling(unittest.TestCase):

    def test_default_suffix_is_appended(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, network_name='mynet', channels=4, nmodes=3, depth=1)
            base = os.path.join(d, 'mynet')
            expected = f"{base}_cvtype0_m3_ch4_dp0.010.pth"
            self.assertEqual(trainer.network_filename, expected)
            self.assertEqual(trainer.stats_filename, expected.replace('.pth', '_stats.json'))

    def test_explicit_extension_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, network_name='mynet.pt', channels=4, nmodes=3, depth=1)
            self.assertTrue(trainer.network_filename.endswith('.pt'))
            self.assertIn('_cvtype0_m3_ch4_dp', trainer.network_filename)

    def test_filename_with_existing_tag_is_used_verbatim(self):
        with tempfile.TemporaryDirectory() as d:
            name = 'mynet_m20.pth'
            trainer = build_trainer(d, network_name=name, channels=4, nmodes=3, depth=1)
            self.assertEqual(trainer.network_filename, os.path.join(d, name))


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestConv2dNetTrainerLoading(unittest.TestCase):

    def _save_checkpoint(self, path, stats_path, nmodes=NMODES, channels=CHANNELS, depth=DEPTH):
        model = UNetRegressor(
            input_channels=1, output_size=nmodes, base_channels=channels,
            input_size=(160, 160), dropout_level=0.01, conv_block_type=0, depth=depth,
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
            # No stats to preserve, so the first trigger() must still (re)compute them.
            self.assertTrue(trainer.firstTrigger)

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

            # Regression: successfully-loaded stats must not be discarded on
            # the next trigger() (firstTrigger must be False, not True).
            self.assertFalse(trainer.firstTrigger)

            loaded_state = trainer.model.state_dict()
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

    def test_full_model_checkpoint_is_loaded(self):
        # torch.save(model, path) (a full model object, not a state_dict) is
        # handled by a separate code branch. torch.load(..., map_location='cpu')
        # is called with weights_only=False so this actually gets loaded
        # instead of being rejected/falling back to a fresh model.
        with tempfile.TemporaryDirectory() as d:
            net_path = os.path.join(d, 'existing_m20_ch4_dp0.010.pth')
            stats_path = net_path.replace('.pth', '_stats.json')
            source_model = UNetRegressor(
                input_channels=1, output_size=NMODES, base_channels=CHANNELS,
                input_size=(160, 160), dropout_level=0.01, conv_block_type=0, depth=DEPTH,
            )
            torch.save(source_model, net_path)
            with open(stats_path, 'w') as f:
                json.dump({'meanp': 0.5, 'stdp': 1.5,
                           'meanmodes': [0.1] * NMODES, 'stdmodes': [1.1] * NMODES,
                           'nmodes': NMODES}, f)

            trainer = build_trainer(d, network_name='existing_m20_ch4_dp0.010.pth',
                                     load_from_file=True)

            self.assertEqual(trainer.meanp, 0.5)
            self.assertEqual(trainer.stdp, 1.5)
            self.assertFalse(trainer.firstTrigger)
            loaded_state = trainer.model.state_dict()
            source_state = source_model.state_dict()
            for key in source_state:
                torch.testing.assert_close(loaded_state[key].cpu(), source_state[key].cpu())

    def test_checkpoint_without_stats_file_warns_and_sets_none(self):
        with tempfile.TemporaryDirectory() as d:
            net_path = os.path.join(d, 'existing_m20_ch4_dp0.010.pth')
            model = UNetRegressor(
                input_channels=1, output_size=NMODES, base_channels=CHANNELS,
                input_size=(160, 160), dropout_level=0.01, conv_block_type=0, depth=DEPTH,
            )
            torch.save(model.state_dict(), net_path)
            # no stats file written on purpose

            trainer = build_trainer(d, network_name='existing_m20_ch4_dp0.010.pth',
                                     load_from_file=True)
            self.assertIsNone(trainer.meanp)
            self.assertIsNone(trainer.stdp)
            self.assertIsNone(trainer.meanmodes)
            self.assertIsNone(trainer.stdmodes)
            self.assertTrue(trainer.firstTrigger)


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
class TestWeightedLosses(unittest.TestCase):
    """WeightedHuberLoss is currently unused by Conv2dNetTrainer (which uses
    a plain, unweighted physical-space MSE -- see TestPhysicalSpaceLoss) but
    is kept as a documented, ready-to-use alternative loss."""

    def test_weighted_huber_loss_quadratic_region(self):
        loss_fn = WeightedHuberLoss(weights=[1.0], delta=1.0, device='cpu')
        preds = torch.tensor([[0.5]])
        targets = torch.tensor([[0.0]])
        # |diff|=0.5 < delta -> pure quadratic: 0.5 * 0.5^2 = 0.125
        self.assertAlmostEqual(loss_fn(preds, targets).item(), 0.125, places=5)

    def test_weighted_huber_loss_linear_region(self):
        loss_fn = WeightedHuberLoss(weights=[1.0], delta=1.0, device='cpu')
        preds = torch.tensor([[3.0]])
        targets = torch.tensor([[0.0]])
        # |diff|=3 > delta: 0.5*delta^2 + delta*(|diff|-delta) = 0.5 + 2 = 2.5
        self.assertAlmostEqual(loss_fn(preds, targets).item(), 2.5, places=5)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestPhysicalSpaceLoss(unittest.TestCase):
    """The training loss must be a plain (unweighted) MSE computed in
    physical (de-normalized) mode units: preds*stdmodes+meanmodes vs. the
    true physical labels -- not the old per-mode-weighted MSE in
    normalized/z-scored space."""

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

    def test_loss_is_plain_mse_in_physical_units(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1, val_split=0.25)
            trainer.model = self._ConstantOutputModel(NMODES)

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

            # firstTrigger: meanmodes/stdmodes are computed directly from
            # the full physical-space batch of labels.
            meanmodes = cpuArray(trainer.meanmodes)
            preds_physical = np.broadcast_to(meanmodes, (n_train, NMODES))
            expected_loss = np.mean((preds_physical - labels[train_idx]) ** 2)

            self.assertAlmostEqual(trainer.loss.item(), expected_loss, places=4)


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
            self.assertEqual(trainer.model.encoders[0].block[0].in_channels, 2)

            feed_batch(trainer)
            trainer.trigger()
            self.assertEqual(trainer.step_count, 1)

    def test_trigger_runs_one_training_step(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            feed_batch(trainer)
            trainer.trigger()

            self.assertEqual(trainer.step_count, 1)
            self.assertTrue(trainer.val_initialized)
            self.assertFalse(trainer.firstTrigger)
            self.assertTrue(hasattr(trainer, 'loss'))
            self.assertTrue(np.isfinite(trainer.loss.item()))
            # First step always improves on the initial min_loss=1e9
            self.assertTrue(os.path.isfile(trainer.network_filename))
            self.assertTrue(os.path.isfile(trainer.stats_filename))

    def test_trigger_creates_missing_parent_directory_before_saving(self):
        with tempfile.TemporaryDirectory() as d:
            nested_dir = os.path.join(d, 'does', 'not', 'exist', 'yet')
            trainer = build_trainer(nested_dir, epoch_len=1)
            self.assertFalse(os.path.isdir(nested_dir))

            feed_batch(trainer)
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
                trainer.outputs['loss'].value, trainer.loss.item(), places=5)

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
class TestToTorchHelper(unittest.TestCase):

    def test_to_torch_converts_numpy_array(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d)
            arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
            t = trainer._to_torch(arr, device='cpu')
            self.assertEqual(t.dtype, torch.float32)
            self.assertEqual(t.device.type, 'cpu')
            np.testing.assert_allclose(t.numpy(), arr)


if __name__ == '__main__':
    unittest.main()
