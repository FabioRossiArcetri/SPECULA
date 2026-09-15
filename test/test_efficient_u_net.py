import unittest

import torch

from specula.lib.efficient_u_net import get_num_groups, UNetRegressor


class TestGetNumGroups(unittest.TestCase):

    def test_prefers_max_groups_when_divisible(self):
        self.assertEqual(get_num_groups(64, max_groups=32), 32)

    def test_falls_back_to_smaller_divisor(self):
        # 48 is not divisible by 32, but is by 16
        self.assertEqual(get_num_groups(48, max_groups=32), 16)

    def test_falls_back_to_one_for_prime_channel_count(self):
        self.assertEqual(get_num_groups(7), 1)

    def test_custom_max_groups(self):
        self.assertEqual(get_num_groups(8, max_groups=4), 4)

    def test_channels_equal_to_one(self):
        self.assertEqual(get_num_groups(1), 1)


class TestUNetRegressorForward(unittest.TestCase):
    """Exercise the network actually used by Conv2dNetTrainer/Tester:
    every conv_block_type must build and produce the requested output shape."""

    def _build(self, conv_block_type, depth=2, base_channels=4, output_size=5,
               input_size=(32, 32)):
        return UNetRegressor(
            input_channels=1,
            output_size=output_size,
            base_channels=base_channels,
            input_size=input_size,
            dropout_level=0.0,
            depth=depth,
            conv_block_type=conv_block_type,
        )

    def test_conv_block_type_0_forward_shape(self):
        model = self._build(conv_block_type=0)
        x = torch.randn(2, 1, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (2, 5))

    def test_conv_block_type_1_depthwise_separable_forward_shape(self):
        model = self._build(conv_block_type=1)
        x = torch.randn(2, 1, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (2, 5))

    def test_conv_block_type_2_inverted_residual_forward_shape(self):
        model = self._build(conv_block_type=2)
        x = torch.randn(2, 1, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (2, 5))

    def test_conv_block_type_3_coord_attention_forward_shape(self):
        model = self._build(conv_block_type=3)
        x = torch.randn(2, 1, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (2, 5))

    def test_conv_block_type_4_dense_forward_shape(self):
        model = self._build(conv_block_type=4)
        x = torch.randn(2, 1, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (2, 5))

    def test_forward_independent_of_input_spatial_size(self):
        # AdaptiveFeatureExtractor pools to a fixed spatial size, so the
        # regression head's input dimension - and thus the output shape -
        # does not depend on the actual input H, W (only on input_size used
        # at construction time for shape inference).
        model = self._build(conv_block_type=0, input_size=(32, 32))
        out_small = model(torch.randn(1, 1, 32, 32))
        out_large = model(torch.randn(1, 1, 64, 64))
        self.assertEqual(out_small.shape, (1, 5))
        self.assertEqual(out_large.shape, (1, 5))

    def test_forward_handles_odd_input_size_via_interpolation(self):
        # With an input size not evenly divisible by 2**depth, the decoder's
        # upsampled feature map and the corresponding encoder skip connection
        # end up with mismatched spatial sizes; the model must reconcile
        # them via interpolation instead of crashing.
        model = self._build(conv_block_type=0, depth=2, input_size=(33, 33))
        out = model(torch.randn(1, 1, 33, 33))
        self.assertEqual(out.shape, (1, 5))

    def test_unsupported_conv_block_type_raises(self):
        with self.assertRaises(AttributeError):
            self._build(conv_block_type=99)

    def test_output_requires_grad_for_backward(self):
        model = self._build(conv_block_type=0)
        x = torch.randn(2, 1, 32, 32, requires_grad=False)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # gradients must flow back to at least one trainable parameter
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)


if __name__ == '__main__':
    unittest.main()
