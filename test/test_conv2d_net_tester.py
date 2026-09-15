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
from specula.processing_objects.conv2d_net_tester import Conv2dNetTester


NMODES = 5
CHANNELS = 4
DEPTH = 1
H = W = 32
BATCH = 4


def save_checkpoint(net_path, stats_path, nmodes=NMODES, channels=CHANNELS, depth=DEPTH):
    model = UNetRegressor(
        input_channels=1, output_size=nmodes, base_channels=channels,
        input_size=(160, 160), dropout_level=0.0, conv_block_type=0, depth=depth,
    )
    torch.save(model.state_dict(), net_path)
    stats = {
        'meanp': 0.0, 'stdp': 1.0,
        'meanmodes': [0.0] * nmodes, 'stdmodes': [1.0] * nmodes,
        'nmodes': nmodes,
    }
    with open(stats_path, 'w') as f:
        json.dump(stats, f)
    return model


def build_tester(tmp_dir, network_name='net.pth', nmodes=NMODES, channels=CHANNELS,
                  depth=DEPTH, target_device_idx=-1):
    net_path = os.path.join(tmp_dir, network_name)
    stats_path = net_path.replace('.pth', '_stats.json')
    save_checkpoint(net_path, stats_path, nmodes=nmodes, channels=channels, depth=depth)
    tester = Conv2dNetTester(
        network_filename=net_path,
        nmodes=nmodes,
        channels=channels,
        depth=depth,
        target_device_idx=target_device_idx,
    )
    return tester, net_path, stats_path


def feed_batch(tester, batch=BATCH, nmodes=NMODES, h=H, w=W, seed=0):
    rng = np.random.default_rng(seed)
    x_val = BaseValue(value=rng.standard_normal((batch, 2, h, w)).astype(np.float32))
    y_val = BaseValue(value=rng.standard_normal((batch, nmodes + 1)).astype(np.float32))
    x_val.generation_time = x_val.seconds_to_t(1)
    y_val.generation_time = y_val.seconds_to_t(1)
    tester.inputs['input_2d_batch'].set(x_val)
    tester.inputs['labels'].set(y_val)
    tester.check_ready(1)


class TestConv2dNetTesterConstruction(unittest.TestCase):

    def test_missing_model_file_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                Conv2dNetTester(network_filename=os.path.join(d, 'nope.pth'),
                                 nmodes=NMODES, channels=CHANNELS, depth=DEPTH)

    def test_missing_stats_file_raises(self):
        with tempfile.TemporaryDirectory() as d:
            net_path = os.path.join(d, 'net.pth')
            model = UNetRegressor(
                input_channels=1, output_size=NMODES, base_channels=CHANNELS,
                input_size=(160, 160), dropout_level=0.0, conv_block_type=0, depth=DEPTH,
            )
            torch.save(model.state_dict(), net_path)
            with self.assertRaises(FileNotFoundError):
                Conv2dNetTester(network_filename=net_path,
                                 nmodes=NMODES, channels=CHANNELS, depth=DEPTH)

    def test_full_model_checkpoint_is_loaded(self):
        # torch.save(model, path) (a full model object, not a state_dict) is
        # handled by a dedicated branch in the source. torch.load(...,
        # map_location='cpu') is called with weights_only=False so this
        # actually gets loaded instead of raising.
        with tempfile.TemporaryDirectory() as d:
            net_path = os.path.join(d, 'net.pth')
            stats_path = net_path.replace('.pth', '_stats.json')
            source_model = UNetRegressor(
                input_channels=1, output_size=NMODES, base_channels=CHANNELS,
                input_size=(160, 160), dropout_level=0.0, conv_block_type=0, depth=DEPTH,
            )
            torch.save(source_model, net_path)
            with open(stats_path, 'w') as f:
                json.dump({'meanp': 0.0, 'stdp': 1.0,
                           'meanmodes': [0.0] * NMODES, 'stdmodes': [1.0] * NMODES,
                           'nmodes': NMODES}, f)

            tester = Conv2dNetTester(network_filename=net_path, nmodes=NMODES,
                                      channels=CHANNELS, depth=DEPTH, target_device_idx=-1)

            loaded_state = tester.model.state_dict()
            source_state = source_model.state_dict()
            for key in source_state:
                torch.testing.assert_close(loaded_state[key].cpu(), source_state[key].cpu())

    def test_successful_construction_loads_stats_and_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d)
            self.assertEqual(tester.meanp, 0.0)
            self.assertEqual(tester.stdp, 1.0)
            np.testing.assert_allclose(tester.meanmodes, [0.0] * NMODES)
            np.testing.assert_allclose(tester.stdmodes, [1.0] * NMODES)
            self.assertEqual(
                set(tester.outputs.keys()),
                {'loss', 'prediction', 'targets', 'error'})
            self.assertTrue(tester.model.training is False)  # eval() was called


class TestConv2dNetTesterTrigger(unittest.TestCase):

    def test_trigger_computes_predictions_and_updates_stats(self):
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d)
            feed_batch(tester)
            tester.trigger()

            self.assertEqual(tester.count, BATCH)
            self.assertEqual(len(tester.all_errors), 1)
            self.assertEqual(tester.outputs['prediction'].shape, (BATCH, NMODES))
            self.assertIsNotNone(tester.total_abs_error)
            self.assertIsNotNone(tester.total_squared_error)

    def test_trigger_noop_when_inputs_missing(self):
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d)
            tester.local_inputs['input_2d_batch'] = None
            tester.local_inputs['labels'] = None
            tester.trigger()
            self.assertEqual(tester.count, 0)

    def test_multiple_triggers_accumulate_count_and_errors(self):
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d)
            feed_batch(tester, seed=1)
            tester.trigger()
            feed_batch(tester, seed=2)
            tester.trigger()

            self.assertEqual(tester.count, 2 * BATCH)
            self.assertEqual(len(tester.all_errors), 2)


class TestConv2dNetTesterFinalize(unittest.TestCase):

    def test_finalize_with_no_data_prints_message(self):
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                tester.finalize()
            self.assertIn('No data processed', buf.getvalue())

    def test_finalize_computes_expected_statistics(self):
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d)
            feed_batch(tester, seed=3)
            tester.trigger()

            # Ground truth, derived directly from the per-sample error that
            # trigger() recorded, independent of the accumulator internals.
            error = tester.all_errors[0]
            expected_mae = np.mean(np.abs(error), axis=0, keepdims=True)
            expected_rmse = np.sqrt(np.mean(error**2, axis=0, keepdims=True))

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                tester.finalize()

            self.assertIn('FINAL TEST STATISTICS', buf.getvalue())
            self.assertEqual(tester.outputs['mean_absolute_error'].shape, (1, NMODES))
            self.assertEqual(tester.outputs['root_mean_squared_error'].shape, (1, NMODES))
            self.assertEqual(tester.outputs['smape'].shape, (NMODES,))
            self.assertTrue(np.isscalar(tester.outputs['overall_smape'])
                             or tester.outputs['overall_smape'].shape == ())
            np.testing.assert_allclose(
                tester.outputs['mean_absolute_error'], expected_mae, rtol=1e-5)
            np.testing.assert_allclose(
                tester.outputs['root_mean_squared_error'], expected_rmse, rtol=1e-5)

    def test_finalize_handles_varying_batch_sizes_across_triggers(self):
        # Regression check: total_abs_error/total_squared_error used to keep
        # the shape of whichever batch created them, so a second trigger()
        # with a different batch size would fail to broadcast into it.
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d)
            feed_batch(tester, batch=4, seed=1)
            tester.trigger()
            feed_batch(tester, batch=7, seed=2)
            tester.trigger()

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                tester.finalize()

            self.assertEqual(tester.count, 11)
            self.assertEqual(tester.outputs['mean_absolute_error'].shape, (1, NMODES))

            # all_targets must accumulate across triggers just like
            # all_errors, matching each trigger's batch size.
            self.assertEqual([t.shape[0] for t in tester.all_targets], [4, 7])

            expected_mae = np.mean(
                np.abs(np.concatenate(tester.all_errors, axis=0)), axis=0, keepdims=True)
            np.testing.assert_allclose(
                tester.outputs['mean_absolute_error'], expected_mae, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
