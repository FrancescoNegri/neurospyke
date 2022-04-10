import numpy as np
import matplotlib.pyplot as plt
from .. import utils

def _parse_kwargs(spikes_times, n_channels, **kwargs):
    # Basic checks
    kwargs_list = [
        {'key': 'channel_height', 'default': 0.5, 'type': float},
        {'key': 'color', 'default': 'black', 'type': str},
        {'key': 'dpi', 'default': 300, 'type': float},
        {'key': 'linewidth', 'default': 0.5, 'type': float},
        {'key': 'title', 'default': 'Spike Train', 'type': str},
        {'key': 'reverse', 'default': False, 'type': bool},
        {'key': 'vertical_spacing', 'default': 0.25, 'type': float},
        {'key': 'xlim', 'default': (0, np.amax([spikes_times[channel_idx][-1] for channel_idx in np.arange(n_channels)]) * 1.01), 'type': tuple}
    ]
    kwargs = utils.check_kwargs_list(kwargs_list, **kwargs)

    # Additional checks
    if kwargs.get('vertical_spacing') < 0 or kwargs.get('vertical_spacing') > 1:
        raise ValueError("'vertical_spacing' expected to be a value between 0 and 1, received " + str(kwargs.get('vertical_spacing')))
    
    # Dependant checks
    kwargs_list = [
        {'key': 'figsize', 'default': (16, kwargs.get('channel_height') * n_channels), 'type': tuple}
    ]
    kwargs = utils.check_kwargs_list(kwargs_list, **kwargs)
    
    return kwargs

def _parse_parameters(bursts_start_idxs, bursts_end_idxs, sampling_time, channel_labels):
    if type(bursts_start_idxs) is not list:
        bursts_start_idxs = [bursts_start_idxs]
    bursts_start_idxs = np.array(bursts_start_idxs, dtype=object)

    if type(bursts_end_idxs) is not list:
        bursts_end_idxs = [bursts_end_idxs]
    bursts_end_idxs = np.array(bursts_end_idxs, dtype=object)

    if (channel_labels is not None) and (type(channel_labels) is not list):
        channel_labels = [channel_labels]

    return bursts_start_idxs, bursts_end_idxs, sampling_time, channel_labels

def plot_spike_train(bursts_start_idxs, bursts_end_idxs, sampling_time, channel_labels=None, **kwargs):    
    bursts_start_idxs, bursts_end_idxs, sampling_time, channel_labels = _parse_parameters(bursts_start_idxs, bursts_end_idxs, sampling_time, channel_labels)
    n_channels = bursts_start_idxs.shape[0]
    spikes_times = sampling_time * spikes_idxs

    kwargs = _parse_kwargs(spikes_times, n_channels, **kwargs)

    plt.figure(figsize=kwargs.get('figsize'), dpi=kwargs.get('dpi'))

    if (channel_labels is None) or (len(channel_labels) != n_channels):
        channel_labels = [str(i) for i in np.arange(1, n_channels + 1, 1)]

    if (kwargs.get('reverse') is True) and (n_channels > 1):
        spikes_times = np.flip(spikes_times, axis=0)
        channel_labels = np.flip(channel_labels)
    
    for channel_idx in np.arange(n_channels):
        for spike_idx in np.arange(len(spikes_times[channel_idx])):
            y = [channel_idx * kwargs.get('channel_height'), channel_idx * kwargs.get('channel_height') + kwargs.get('channel_height') * (1 - kwargs.get('vertical_spacing'))]
            x = [spikes_times[channel_idx][spike_idx], spikes_times[channel_idx][spike_idx]]
            plt.plot(x, y, color=kwargs.get('color'), linewidth=kwargs.get('linewidth'))

    plt.title(kwargs.get('title'))
    plt.xlabel('Time (s)')
    plt.ylabel('Channels')

    ax = plt.gca()
    
    ax.set_xlim(kwargs.get('xlim')[0], kwargs.get('xlim')[1])
    ax.set_ylim(0, kwargs.get('channel_height') * n_channels)
    
    yticks = np.arange(kwargs.get('channel_height') / 2, round(n_channels * kwargs.get('channel_height') + kwargs.get('channel_height') / 2, 8), kwargs.get('channel_height'))
    ax.set_yticks(yticks)
    ax.set_yticklabels(channel_labels)
    ax.tick_params(axis='y', which='both', length=0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return