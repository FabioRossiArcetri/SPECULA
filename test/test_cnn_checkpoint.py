import json
import os
import tempfile
import unittest

try:
    import torch
    from specula.lib.efficient_u_net import UNetRegressor
    from specula.lib.cnn_checkpoint import (check_trained_settings, load_stats, load_trained_network,
                                            save_checkpoint, stats_filename)
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


NMODES = 5
STATS = {'meanp': 0.0, 'stdp': 1.0, 'meanmodes': [0.0] * NMODES, 'stdmodes': [1.0] * NMODES}


def network(head_type='pooled', n_frames=1, head_grid=32):
    return UNetRegressor(input_channels=2 * n_frames, output_size=NMODES, base_channels=4,
                         dropout_level=0.0, depth=2, conv_block_type=0, head_type=head_type,
                         head_grid=head_grid)


def load(path, head_type='pooled', n_frames=1, head_grid=32):
    return load_trained_network(path, nmodes=NMODES, input_channels=2, n_frames=n_frames, channels=4,
                                depth=2, dropout=0.0, conv_block_type=0, head_type=head_type,
                                head_grid=head_grid)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestCnnCheckpoint(unittest.TestCase):

    def test_stats_filename_is_next_to_the_weights(self):
        self.assertEqual(stats_filename('a/b/net.pth'), 'a/b/net_stats.json')
        self.assertEqual(stats_filename('a/b/net_dp0.000.pth'), 'a/b/net_dp0.000_stats.json')

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'sub', 'net.pth')    # missing directories are created
            model = network('spatial', n_frames=4)
            save_checkpoint(model, path, dict(STATS, n_frames=4, head_type='spatial'))
            loaded, stats = load(path, 'spatial', n_frames=4)
            self.assertFalse(loaded.training)
            self.assertEqual(stats['n_frames'], 4)
            for k, v in model.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[k], v)

    def test_missing_files_raise(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'net.pth')
            with self.assertRaisesRegex(FileNotFoundError, 'Model file'):
                load(path)
            torch.save(network().state_dict(), path)
            with self.assertRaisesRegex(FileNotFoundError, 'Statistics file'):
                load(path)
            with self.assertRaises(FileNotFoundError):
                load_stats(path)

    def test_trained_settings_mismatch_raises_with_the_trained_value(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'net.pth')
            save_checkpoint(network('spatial', n_frames=4), path, dict(STATS, n_frames=4, head_type='spatial'))
            check_trained_settings(path, n_frames=4, head_type='spatial')
            with self.assertRaisesRegex(ValueError, 'n_frames=4'):
                check_trained_settings(path, n_frames=1)
            with self.assertRaisesRegex(ValueError, "head_type='spatial'"):
                load(path, 'pooled', n_frames=4)

    def test_head_grid_is_checked_too(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'net.pth')
            save_checkpoint(network('spatial', head_grid=16), path,
                            dict(STATS, head_type='spatial', head_grid=16))
            load(path, 'spatial', head_grid=16)
            with self.assertRaisesRegex(ValueError, 'head_grid=16'):
                load(path, 'spatial', head_grid=32)

    def test_old_stats_without_the_settings_mean_the_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'net.pth')
            with open(stats_filename(path), 'w') as f:
                json.dump(STATS, f)
            check_trained_settings(path, n_frames=1, head_type='pooled')
            with self.assertRaisesRegex(ValueError, "head_type='pooled'"):
                check_trained_settings(path, head_type='spatial')
            with self.assertRaisesRegex(ValueError, 'n_frames=1'):
                check_trained_settings(path, n_frames=4)

    def test_calibration_factor_undoes_the_measured_shrinkage(self):
        from specula.lib.cnn_checkpoint import calibration_factor
        stats = dict(STATS, mode_gain=[1.0, 0.5, 0.25, 0.1, -0.3])
        f = calibration_factor(stats, NMODES)
        # 1/gain where the network predicts the mode, capped at 3, and left
        # alone where it barely does (gain <= 0.2): amplifying those would
        # mostly amplify noise
        self.assertEqual(list(f), [1.0, 2.0, 3.0, 1.0, 1.0])
        self.assertIsNone(calibration_factor(STATS, NMODES))            # no measurement
        self.assertIsNone(calibration_factor(dict(STATS, mode_gain=[1.0]), NMODES))   # wrong length

    def test_missing_stats_file_is_not_checked(self):
        check_trained_settings('/nonexistent/net.pth', n_frames=4)


if __name__ == '__main__':
    unittest.main()
