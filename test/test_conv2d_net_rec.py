import os
import json
import tempfile
import unittest

import specula
specula.init(0)  # Default target device

import numpy as np

from specula.base_value import BaseValue
from specula.data_objects.slopes import Slopes

# torch is an optional dependency (see pyproject.toml's "nn" extra): skip
# every test in this module rather than failing collection when it's absent.
try:
    import torch
    from specula.lib.efficient_u_net import UNetRegressor
    from specula.processing_objects.conv2d_net_rec import Conv2dNetRec
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


NMODES = 5
CHANNELS = 4
DEPTH = 1
MASK_SIDE = 8
N_VALID = 6  # number of valid subapertures


def save_checkpoint(net_path, stats_path, nmodes=NMODES, channels=CHANNELS, depth=DEPTH,
                     input_channels=2):
    model = UNetRegressor(
        input_channels=input_channels, output_size=nmodes, base_channels=channels,
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


def build_rec(tmp_dir, network_name='net.pth', nmodes=NMODES, channels=CHANNELS, depth=DEPTH,
              input_channels=2, target_device_idx=-1, **kwargs):
    net_path = os.path.join(tmp_dir, network_name)
    stats_path = net_path.replace('.pth', '_stats.json')
    save_checkpoint(net_path, stats_path, nmodes=nmodes, channels=channels, depth=depth,
                    input_channels=input_channels)
    rec = Conv2dNetRec(
        network_filename=net_path,
        nmodes=nmodes,
        channels=channels,
        depth=depth,
        input_channels=input_channels,
        target_device_idx=target_device_idx,
        **kwargs,
    )
    return rec, net_path, stats_path


def build_slopes(target_device_idx, seed=0):
    """A Slopes object with a valid single_mask/display_map so get2d()
    returns a real (2, MASK_SIDE, MASK_SIDE) map, as a PyrSlopec/ShSlopec
    output would."""
    rng = np.random.default_rng(seed)
    slopes = Slopes(length=2 * N_VALID, interleave=False, target_device_idx=target_device_idx)
    slopes.slopes[:] = rng.standard_normal(2 * N_VALID)
    slopes.single_mask = np.zeros((MASK_SIDE, MASK_SIDE), dtype=bool)
    slopes.display_map = np.arange(N_VALID)
    return slopes


def run_once(rec, slopes, target_device_idx):
    slopes.generation_time = rec.seconds_to_t(1)
    rec.inputs['in_slopes'].set(slopes)
    rec.setup()
    t = rec.seconds_to_t(1)
    rec.check_ready(t)
    rec.trigger()
    rec.post_trigger()
    from specula import cpuArray
    return cpuArray(rec.outputs['out_modes'].value).copy()


class TestConv2dNetRecConstruction(unittest.TestCase):

    @unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
    def test_missing_model_file_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                Conv2dNetRec(network_filename=os.path.join(d, 'nope.pth'),
                            nmodes=NMODES, channels=CHANNELS, depth=DEPTH, target_device_idx=-1)

    @unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
    def test_missing_stats_file_raises(self):
        with tempfile.TemporaryDirectory() as d:
            net_path = os.path.join(d, 'net.pth')
            model = UNetRegressor(
                input_channels=2, output_size=NMODES, base_channels=CHANNELS,
                input_size=(160, 160), dropout_level=0.0, conv_block_type=0, depth=DEPTH,
            )
            torch.save(model.state_dict(), net_path)
            with self.assertRaises(FileNotFoundError):
                Conv2dNetRec(network_filename=net_path, nmodes=NMODES,
                            channels=CHANNELS, depth=DEPTH, target_device_idx=-1)

    @unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
    def test_successful_construction(self):
        with tempfile.TemporaryDirectory() as d:
            rec, _, _ = build_rec(d)
            self.assertEqual(rec.model.encoders[0].block[0].in_channels, 2)
            self.assertEqual(set(rec.outputs.keys()), {'out_modes'})
            self.assertTrue(rec.model.training is False)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestConv2dNetRecTrigger(unittest.TestCase):

    def test_trigger_produces_modes_of_correct_shape(self):
        with tempfile.TemporaryDirectory() as d:
            rec, _, _ = build_rec(d)
            slopes = build_slopes(target_device_idx=-1)
            out = run_once(rec, slopes, target_device_idx=-1)
            self.assertEqual(out.shape, (NMODES,))
            self.assertTrue(np.all(np.isfinite(out)))

    def test_without_baseline_output_is_bare_denormalized_network_output(self):
        with tempfile.TemporaryDirectory() as d:
            rec, _, _ = build_rec(d)
            rec.meanmodes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            rec.stdmodes = np.array([2.0, 1.0, 0.5, 4.0, 1.5])
            rec.model = lambda x: torch.zeros(1, NMODES)

            slopes = build_slopes(target_device_idx=-1)
            out = run_once(rec, slopes, target_device_idx=-1)

            np.testing.assert_allclose(out, rec.meanmodes, atol=1e-5)

    def test_baseline_is_added_back_to_network_output(self):
        with tempfile.TemporaryDirectory() as d:
            rec, _, _ = build_rec(d)
            rec.meanmodes = np.zeros(NMODES)
            rec.stdmodes = np.ones(NMODES)
            rec.model = lambda x: torch.zeros(1, NMODES)  # predicted residual == 0

            baseline_vals = np.array([10.0, -1.0, 0.5, 3.0, -2.0])
            baseline = BaseValue(value=baseline_vals.astype(np.float32))
            slopes = build_slopes(target_device_idx=-1)
            baseline.generation_time = slopes.generation_time = 1
            rec.inputs['baseline'].set(baseline)

            out = run_once(rec, slopes, target_device_idx=-1)
            np.testing.assert_allclose(out, baseline_vals, atol=1e-5)

    def test_baseline_offset_selects_correct_columns(self):
        with tempfile.TemporaryDirectory() as d:
            rec, _, _ = build_rec(d, baseline_offset=2)
            rec.meanmodes = np.zeros(NMODES)
            rec.stdmodes = np.ones(NMODES)
            rec.model = lambda x: torch.zeros(1, NMODES)

            baseline_full = np.arange(10.0)  # [0..9]
            baseline = BaseValue(value=baseline_full.astype(np.float32))
            slopes = build_slopes(target_device_idx=-1)
            baseline.generation_time = slopes.generation_time = 1
            rec.inputs['baseline'].set(baseline)

            out = run_once(rec, slopes, target_device_idx=-1)
            np.testing.assert_allclose(out, baseline_full[2:2 + NMODES], atol=1e-5)


if __name__ == '__main__':
    unittest.main()
