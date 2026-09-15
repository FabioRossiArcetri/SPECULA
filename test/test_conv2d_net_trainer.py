import io
import os
import json
import tempfile
import unittest
import contextlib

import specula
specula.init(0)  # Default target device

import numpy as np
import torch

from specula.base_value import BaseValue
from specula.lib.efficient_u_net import UNetRegressor
from specula.processing_objects.conv2d_net_trainer import (
    Conv2dNetTrainer,
    EarlyStopping,
    WeightedHuberLoss,
    WeightedMSELoss,
)


NMODES = 20
CHANNELS = 4
DEPTH = 2
H = W = 32
BATCH = 8


def build_trainer(tmp_dir, network_name='testnet.pth', nmodes=NMODES, channels=CHANNELS,
                   depth=DEPTH, epoch_len=1, patience=600, val_split=0.25,
                   load_from_file=False, conv_block_type=0, target_device_idx=-1):
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
            np.testing.assert_allclose(trainer.meanmodes.numpy(), [0.1] * NMODES)
            np.testing.assert_allclose(trainer.stdmodes.numpy(), [1.1] * NMODES)

            loaded_state = trainer.model.state_dict()
            source_state = source_model.state_dict()
            for key in source_state:
                torch.testing.assert_close(loaded_state[key].cpu(), source_state[key].cpu())

    def test_full_model_checkpoint_falls_back_to_new_model(self):
        # torch.save(model, path) (a full model object, not a state_dict) is
        # handled by a separate code branch, but with this torch version
        # torch.load(..., map_location='cpu') defaults to weights_only=True,
        # which refuses to unpickle an arbitrary model class. The load
        # exception is caught by Conv2dNetTrainer, which logs a warning and
        # falls back to a freshly-initialized model instead of crashing.
        with tempfile.TemporaryDirectory() as d:
            net_path = os.path.join(d, 'existing_m20_ch4_dp0.010.pth')
            stats_path = net_path.replace('.pth', '_stats.json')
            source_model = UNetRegressor(
                input_channels=1, output_size=NMODES, base_channels=CHANNELS,
                input_size=(160, 160), dropout_level=0.01, conv_block_type=0, depth=DEPTH,
            )
            torch.save(source_model, net_path)
            with open(stats_path, 'w') as f:
                json.dump({'meanp': 0.0, 'stdp': 1.0,
                           'meanmodes': [0.0] * NMODES, 'stdmodes': [1.0] * NMODES,
                           'nmodes': NMODES}, f)

            trainer = build_trainer(d, network_name='existing_m20_ch4_dp0.010.pth',
                                     load_from_file=True)

            self.assertIsNone(trainer.meanp)
            self.assertIsInstance(trainer.model, torch.nn.Module)

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


class TestConv2dNetTrainerDevice(unittest.TestCase):

    def test_device_matches_cuda_availability(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d)
            expected = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.assertEqual(trainer.device.type, expected)


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


class TestWeightedLosses(unittest.TestCase):

    def test_weighted_mse_loss_matches_manual_computation(self):
        weights = [1.0, 2.0, 0.5]
        loss_fn = WeightedMSELoss(weights=weights, device='cpu')
        preds = torch.tensor([[1.0, 2.0, 3.0]])
        targets = torch.tensor([[0.0, 0.0, 0.0]])
        expected = np.mean(np.array(weights) * np.array([1.0, 4.0, 9.0]))
        self.assertAlmostEqual(loss_fn(preds, targets).item(), expected, places=5)

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


class TestUpdateLossWeights(unittest.TestCase):

    def test_weights_are_inverse_std_normalized_to_nmodes(self):
        # nmodes=6 so the weight-init slice assignment in __init__
        # (ww[1:6] = [...]) does not change the length of the weights buffer.
        nmodes = 6
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, nmodes=nmodes, channels=4, depth=1)
            modes = np.arange(1, 3 * nmodes + 1).reshape(3, nmodes).astype(float)
            trainer.update_loss_weights(modes)

            stds = np.std(modes, axis=0)
            expected = 1.0 / (stds + 1e-8)
            expected = expected / expected.sum() * nmodes

            np.testing.assert_allclose(
                trainer.loss_fn.weights.detach().numpy(), expected, rtol=1e-5)


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
