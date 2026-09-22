"""
Saving and loading the networks trained by Conv2dNetTrainer.

A checkpoint is two files:

- ``<name>.pth``: the network weights (a state_dict);
- ``<name>_stats.json``: the normalization statistics (meanp/stdp for the
  input, meanmodes/stdmodes for the output) plus the settings the network
  was trained with that the code loading it must match (TRAINED_SETTINGS).
"""

import json
import os

import torch

from specula.lib.efficient_u_net import UNetRegressor


# Settings saved in the stats file that the code loading the network must
# match, with the value implied by stats files written before they existed.
TRAINED_SETTINGS = {'n_frames': 1, 'head_type': 'pooled', 'head_grid': 32}


def stats_filename(network_filename):
    return os.path.splitext(network_filename)[0] + '_stats.json'


def check_trained_settings(network_filename, **settings):
    """Raise a clear error if the stats file saved with the checkpoint says
    the network was trained with different settings (e.g. n_frames=4,
    head_type='spatial') than the ones given -- otherwise the mismatch shows
    up only as an obscure tensor-shape error when loading the weights. Does
    nothing if there is no stats file."""
    filename = stats_filename(network_filename)
    if not os.path.isfile(filename):
        return
    with open(filename, 'r') as f:
        stats = json.load(f)
    for key, value in settings.items():
        trained = stats.get(key, TRAINED_SETTINGS[key])
        if trained != value:
            raise ValueError(f'{key}={value!r}, but the network in {filename} was trained '
                             f'with {key}={trained!r}: set {key} to {trained!r}')


def load_stats(network_filename):
    filename = stats_filename(network_filename)
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'Statistics file not found at {filename}. '
                                f'Make sure to train the model first!')
    with open(filename, 'r') as f:
        return json.load(f)


def load_weights(model, network_filename):
    model.load_state_dict(torch.load(network_filename, map_location='cpu', weights_only=True))


def save_checkpoint(model, network_filename, stats):
    os.makedirs(os.path.dirname(network_filename) or '.', exist_ok=True)
    torch.save(model.state_dict(), network_filename)
    with open(stats_filename(network_filename), 'w') as f:
        json.dump(stats, f, indent=2)


def load_trained_network(network_filename, nmodes, input_channels, n_frames, channels,
                         depth, dropout, conv_block_type, head_type, head_grid=32):
    """Build the network, load its weights and return it in eval mode, on
    the CPU, together with its stats (a dict). The arguments must match
    the ones it was trained with (see Conv2dNetTrainer)."""
    if not os.path.isfile(network_filename):
        raise FileNotFoundError(f'Model file not found at {network_filename}')
    stats = load_stats(network_filename)
    check_trained_settings(network_filename, n_frames=n_frames, head_type=head_type,
                           head_grid=head_grid)
    model = UNetRegressor(
        input_channels=input_channels * n_frames,
        output_size=nmodes,
        base_channels=channels,
        dropout_level=dropout,
        depth=depth,
        conv_block_type=conv_block_type,
        head_type=head_type,
        head_grid=head_grid,
    )
    load_weights(model, network_filename)
    return model.eval(), stats
