import json
import os
import tempfile
import unittest

try:
    import torch
    from specula.lib.cnn_checkpoint import (NETWORK_KEYS, build_network, calibration_factor,
                                            check_same_network, load_stats, load_trained_network,
                                            predict, save_checkpoint, stats_filename)
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


NMODES = 5
NETWORK = dict(nmodes=NMODES, input_channels=2, n_frames=1, channels=4, depth=2,
               conv_block_type=0, head_type='pooled', head_grid=32, dropout=0.0)
STATS = {'meanp': 0.0, 'stdp': 1.0, 'meanmodes': [0.0] * NMODES, 'stdmodes': [1.0] * NMODES}


def save(path, **network):
    net = dict(NETWORK, **network)
    model = build_network(net)
    save_checkpoint(model, path, dict(STATS, network=net))
    return model, net


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestCnnCheckpoint(unittest.TestCase):

    def test_stats_filename_is_next_to_the_weights(self):
        self.assertEqual(stats_filename('a/b/net.pth'), 'a/b/net_stats.json')
        self.assertEqual(stats_filename('a/b/net_dp0.000.pth'), 'a/b/net_dp0.000_stats.json')

    def test_the_network_is_rebuilt_from_the_checkpoint_alone(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'sub', 'net.pth')    # missing directories are created
            model, net = save(path, head_type='spatial', head_grid=16, n_frames=4)
            loaded, stats = load_trained_network(path)
            self.assertFalse(loaded.training)
            self.assertEqual(stats['network'], net)
            for k, v in model.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[k], v)

    def test_every_architecture_argument_is_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'net.pth')
            save(path)
            self.assertEqual(set(load_stats(path)['network']), set(NETWORK_KEYS))

    def test_missing_files_raise(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'net.pth')
            with self.assertRaisesRegex(FileNotFoundError, 'Model file'):
                load_trained_network(path)
            torch.save(build_network(NETWORK).state_dict(), path)
            with self.assertRaisesRegex(FileNotFoundError, 'Statistics file'):
                load_trained_network(path)

    def test_a_checkpoint_without_its_architecture_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'net.pth')
            save_checkpoint(build_network(NETWORK), path, STATS)
            with self.assertRaisesRegex(ValueError, 'does not record the network'):
                load_trained_network(path)

    def test_resuming_with_a_different_network_names_every_difference(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'net.pth')
            _, net = save(path)
            check_same_network(path, net)
            with self.assertRaises(ValueError) as cm:
                check_same_network(path, dict(net, depth=3, n_frames=4))
            self.assertIn('depth=3 (checkpoint: 2)', str(cm.exception))
            self.assertIn('n_frames=4 (checkpoint: 1)', str(cm.exception))

    def test_no_checkpoint_yet_is_not_checked(self):
        check_same_network('/nonexistent/net.pth', NETWORK)

    def test_predict_in_chunks_matches_one_pass_and_denormalizes(self):
        model = build_network(NETWORK).eval()
        x = torch.randn(7, 2, 16, 16)
        mean, std = torch.arange(NMODES) * 1.0, torch.full((NMODES,), 2.0)
        with torch.no_grad():
            expected = model(x) * std + mean
        torch.testing.assert_close(predict(model, x, mean, std, chunk=3), expected)
        self.assertFalse(predict(model.train(), x, mean, std).requires_grad)
        self.assertFalse(model.training)      # left in eval mode

    def test_calibration_factor_undoes_the_measured_shrinkage(self):
        stats = dict(STATS, mode_gain=[1.0, 0.5, 0.25, 0.1, -0.3])
        # 1/gain where the network predicts the mode, capped at 3, and left
        # alone where it barely does (gain <= 0.2): amplifying those would
        # mostly amplify noise
        self.assertEqual(list(calibration_factor(stats, NMODES)), [1.0, 2.0, 3.0, 1.0, 1.0])
        self.assertIsNone(calibration_factor(STATS, NMODES))                           # no measurement
        self.assertIsNone(calibration_factor(dict(STATS, mode_gain=[1.0]), NMODES))    # wrong length


if __name__ == '__main__':
    unittest.main()
