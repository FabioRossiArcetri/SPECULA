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
                     input_channels=2, head_type=None):
    model = UNetRegressor(
        input_channels=input_channels, output_size=nmodes, base_channels=channels,
        dropout_level=0.0, conv_block_type=0, depth=depth,
        head_type=head_type or 'pooled',
    )
    torch.save(model.state_dict(), net_path)
    stats = {
        'meanp': 0.0, 'stdp': 1.0,
        'meanmodes': [0.0] * nmodes, 'stdmodes': [1.0] * nmodes,
        'nmodes': nmodes,
    }
    if head_type is not None:   # None: a stats file from before head_type existed
        stats['head_type'] = head_type
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
                dropout_level=0.0, conv_block_type=0, depth=DEPTH,
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
    def test_sanity_check_passes_with_baseline_wired(self):
        # Regression test: sanity_check() (run by Simul/LoopControl before
        # any real simulation) failed here because Conv2dNetRec inherits
        # BaseModalrec.input_names(), which doesn't declare 'baseline', so
        # the framework rejected it as an undeclared input the moment a
        # real YAML actually wired one up -- something no direct-Python
        # unit test here caught until this one.
        with tempfile.TemporaryDirectory() as d:
            rec, _, _ = build_rec(d)
            baseline = BaseValue(value=np.zeros(NMODES, dtype=np.float32))
            rec.inputs['baseline'].set(baseline)
            rec.sanity_check()  # must not raise

    @unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
    def test_n_frames_stacks_the_previous_slope_maps(self):
        with tempfile.TemporaryDirectory() as d:
            net_path = os.path.join(d, 'net.pth')
            stats_path = net_path.replace('.pth', '_stats.json')
            model = save_checkpoint(net_path, stats_path, input_channels=2 * 2)
            with open(stats_path) as f:
                stats = json.load(f)
            stats['n_frames'] = 2
            with open(stats_path, 'w') as f:
                json.dump(stats, f)

            rec = Conv2dNetRec(network_filename=net_path, nmodes=NMODES, channels=CHANNELS,
                               depth=DEPTH, input_channels=2, n_frames=2, target_device_idx=-1)
            first, second = build_slopes(-1, seed=0), build_slopes(-1, seed=1)
            out1 = run_once(rec, first, -1)
            out2 = run_once(rec, second, -1)

            m1 = np.asarray(first.get2d(), dtype=np.float32)
            m2 = np.asarray(second.get2d(), dtype=np.float32)
            model.eval()
            with torch.no_grad():
                # meanp/stdp = 0/1 and meanmodes/stdmodes = 0/1 in the stats:
                # the output is the raw network output on the stacked maps.
                exp1 = model(torch.from_numpy(np.concatenate([m1, m1])[None])).numpy()[0]
                exp2 = model(torch.from_numpy(np.concatenate([m2, m1])[None])).numpy()[0]
            np.testing.assert_allclose(out1, exp1, rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(out2, exp2, rtol=1e-4, atol=1e-5)

            with self.assertRaisesRegex(ValueError, 'n_frames=2'):
                Conv2dNetRec(network_filename=net_path, nmodes=NMODES, channels=CHANNELS,
                             depth=DEPTH, input_channels=2, target_device_idx=-1)

    @unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
    def test_spatial_head_checkpoint_is_loaded_and_checked(self):
        with tempfile.TemporaryDirectory() as d:
            net_path = os.path.join(d, 'net.pth')
            stats_path = net_path.replace('.pth', '_stats.json')
            model = save_checkpoint(net_path, stats_path, depth=2, head_type='spatial')

            rec = Conv2dNetRec(network_filename=net_path, nmodes=NMODES, channels=CHANNELS,
                               depth=2, input_channels=2, head_type='spatial', target_device_idx=-1)
            slopes = build_slopes(-1)
            out = run_once(rec, slopes, -1)
            x = torch.from_numpy(np.asarray(slopes.get2d(), dtype=np.float32))[None]
            with torch.no_grad():
                expected = model.eval()(x).numpy()[0]
            np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)

            with self.assertRaisesRegex(ValueError, "head_type='spatial'"):
                Conv2dNetRec(network_filename=net_path, nmodes=NMODES, channels=CHANNELS,
                             depth=2, input_channels=2, target_device_idx=-1)

    @unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
    def test_calibrate_gain_scales_the_prediction_per_mode(self):
        with tempfile.TemporaryDirectory() as d:
            net_path = os.path.join(d, 'net.pth')
            stats_path = net_path.replace('.pth', '_stats.json')
            save_checkpoint(net_path, stats_path)
            with open(stats_path) as f:
                stats = json.load(f)
            stats['meanmodes'] = [1.0] * NMODES
            stats['mode_gain'] = [1.0, 0.5, 0.25, 0.1, 1.0]   # last two: not correctable / no shrink
            with open(stats_path, 'w') as f:
                json.dump(stats, f)

            rec = Conv2dNetRec(network_filename=net_path, nmodes=NMODES, channels=CHANNELS,
                               depth=DEPTH, input_channels=2, calibrate_gain=True,
                               target_device_idx=-1)
            rec.meanmodes = np.ones(NMODES)
            rec.stdmodes = np.ones(NMODES)
            rec.model = lambda x: torch.full((1, NMODES), 2.0)   # -> 3.0 before calibration
            out = run_once(rec, build_slopes(-1), -1)
            # the deviation from meanmodes (2.0) is multiplied by 1/gain, capped at 3
            np.testing.assert_allclose(out, 1.0 + np.array([2.0, 4.0, 6.0, 2.0, 2.0]), atol=1e-5)

    @unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
    def test_calibrate_gain_without_a_measurement_raises(self):
        with tempfile.TemporaryDirectory() as d:
            net_path = os.path.join(d, 'net.pth')
            save_checkpoint(net_path, net_path.replace('.pth', '_stats.json'))
            with self.assertRaisesRegex(ValueError, 'no per-mode gain'):
                Conv2dNetRec(network_filename=net_path, nmodes=NMODES, channels=CHANNELS,
                             depth=DEPTH, input_channels=2, calibrate_gain=True,
                             target_device_idx=-1)

    @unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
    def test_sanity_check_passes_without_baseline_wired(self):
        with tempfile.TemporaryDirectory() as d:
            rec, _, _ = build_rec(d)
            rec.sanity_check()  # must not raise: baseline is optional


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
