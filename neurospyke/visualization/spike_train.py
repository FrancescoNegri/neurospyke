import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .. import utils

def _parse_kwargs(spikes_times, n_channels, **kwargs):
    # Basic checks
    kwargs_list = [
        {'key': 'ax', 'default': None, 'type': None},
        {'key': 'boxoff', 'default': True, 'type': bool},
        {'key': 'channel_height', 'default': 0.5, 'type': float},
        {'key': 'channel_labels', 'default': [str(i) for i in np.arange(1, n_channels + 1, 1)], 'type': tuple},
        {'key': 'color', 'default': ['black' for _ in np.arange(1, n_channels + 1, 1)], 'type': list},
        {'key': 'dpi', 'default': 300, 'type': float},
        {'key': 'figsize', 'default': (16, kwargs.get('channel_height', 0.5) * n_channels), 'type': tuple},
        {'key': 'linewidth', 'default': 0.5, 'type': float},
        {'key': 'num', 'default': None, 'type': str},
        {'key': 'sampling_time', 'default': None, 'type': float},
        {'key': 'title', 'default': 'Spike Train', 'type': str},
        {'key': 'reverse', 'default': False, 'type': bool},
        {'key': 'vertical_spacing', 'default': 0.25, 'type': float},
        {'key': 'xlabel', 'default': 'Time (s)' if kwargs.get('sampling_time', None) is not None else 'Samples', 'type': str},
        {'key': 'xlim', 'default': (0, np.amax([spikes_times[channel_idx][-1] for channel_idx in np.arange(n_channels)]) * 1.01), 'type': tuple},
        {'key': 'ylabel', 'default': 'Channels', 'type': str}
    ]
    kwargs = utils.check_kwargs_list(kwargs_list, **kwargs)

    # Additional checks
    if (kwargs.get('channel_labels') is not None) and (len(kwargs.get('channel_labels')) != n_channels):
        raise ValueError("'channel_labels' expected to contain " + str(n_channels) + " elements, received " + str(len(kwargs.get('channel_labels'))))
    if len(kwargs.get('color')) != n_channels:
        if len(kwargs.get('color')) == 1:
            kwargs['color'] = [kwargs.get('color')[0] for _ in np.arange(1, n_channels + 1, 1)]
        else:
            raise ValueError("'color' expected to contain " + str(n_channels) + " elements, received " + str(len(kwargs.get('color'))))
    if kwargs.get('vertical_spacing') < 0 or kwargs.get('vertical_spacing') > 1:
        raise ValueError("'vertical_spacing' expected to be a value between 0 and 1, received " + str(kwargs.get('vertical_spacing')))
    
    return kwargs

def plot_spike_train(spikes_idxs, **kwargs):
    if ((type(spikes_idxs) is np.ndarray) and (len(np.shape(spikes_idxs)))) or (len(spikes_idxs) == 1):
        if (hasattr(spikes_idxs[0], '__iter__') is False):
            spikes_idxs = [spikes_idxs]        
    
    spikes_idxs = np.array(spikes_idxs, dtype=object)

    n_channels = spikes_idxs.shape[0]
    spikes_times = kwargs.get('sampling_time') * spikes_idxs if kwargs.get('sampling_time') is not None else spikes_idxs

    kwargs = _parse_kwargs(spikes_times, n_channels, **kwargs)

    if kwargs.get('ax') is None:
        plt.figure(num=kwargs.get('num'), figsize=kwargs.get('figsize'), dpi=kwargs.get('dpi'))
        ax = plt.gca()
    else:
        ax = kwargs.get('ax')

    if (kwargs.get('reverse') is True) and (n_channels > 1):
        spikes_times = np.flip(spikes_times, axis=0)
        kwargs['channel_labels'] = np.flip(kwargs.get('channel_labels'))
        kwargs['color'] = np.flip(kwargs.get('color'))
    
    for channel_idx in np.arange(n_channels):
        for spike_idx in np.arange(len(spikes_times[channel_idx])):
            y = [channel_idx * kwargs.get('channel_height'), channel_idx * kwargs.get('channel_height') + kwargs.get('channel_height') * (1 - kwargs.get('vertical_spacing'))]
            x = [spikes_times[channel_idx][spike_idx], spikes_times[channel_idx][spike_idx]]
            ax.plot(x, y, color=kwargs.get('color')[channel_idx], linewidth=kwargs.get('linewidth'))

    ax.set_title(kwargs.get('title'))
    ax.set_xlabel(kwargs.get('xlabel'))
    ax.set_ylabel(kwargs.get('ylabel'))

    ax.set_xlim(kwargs.get('xlim'))
    ax.set_ylim(0, kwargs.get('channel_height') * n_channels)
    
    yticks = np.arange(kwargs.get('channel_height') / 2, round(n_channels * kwargs.get('channel_height') + kwargs.get('channel_height') / 2, 8), kwargs.get('channel_height'))
    ax.set_yticks(yticks)
    if kwargs.get('channel_labels') is not None:
        ax.set_yticklabels(kwargs.get('channel_labels'))
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis='y', which='both', length=0)

    if kwargs.get('sampling_time') is None:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    if kwargs.get('boxoff') is True:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    return