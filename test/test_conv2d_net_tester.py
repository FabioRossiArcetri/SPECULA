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

    def test_full_model_checkpoint_currently_raises_on_load(self):
        # torch.save(model, path) (a full model object, not a state_dict) is
        # handled by a dedicated branch in the source, but with this torch
        # version torch.load(..., map_location='cpu') defaults to
        # weights_only=True, which refuses to unpickle an arbitrary model
        # class. Unlike Conv2dNetTrainer, Conv2dNetTester does not catch
        # this, so construction currently fails outright for such files.
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

            with self.assertRaises(Exception):
                Conv2dNetTester(network_filename=net_path, nmodes=NMODES,
                                 channels=CHANNELS, depth=DEPTH, target_device_idx=-1)

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

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                tester.finalize()

            self.assertIn('FINAL TEST STATISTICS', buf.getvalue())
            # NOTE: total_abs_error/total_squared_error are accumulated
            # per-sample (never reduced over the batch axis), so with a
            # single trigger() call these outputs keep the (batch, nmodes)
            # shape rather than a true per-mode (1, nmodes) summary.
            self.assertEqual(tester.outputs['mean_absolute_error'].shape, (BATCH, NMODES))
            self.assertEqual(tester.outputs['root_mean_squared_error'].shape, (BATCH, NMODES))
            self.assertEqual(tester.outputs['smape'].shape, (NMODES,))
            self.assertTrue(np.isscalar(tester.outputs['overall_smape'])
                             or tester.outputs['overall_smape'].shape == ())
            self.assertTrue(np.all(tester.outputs['mean_absolute_error'] >= 0))
            self.assertTrue(np.all(tester.outputs['root_mean_squared_error'] >= 0))


if __name__ == '__main__':
    unittest.main()
