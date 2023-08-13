import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator
from .. import utils

def _compute_default_xlim(spikes_idxs, n_channels, margin = 1.01):
    xlim = [0, 0]
    
    for channel_idx in range(n_channels):
        if np.size(spikes_idxs[channel_idx]) > 0:
            channel_max = np.amax(spikes_idxs[channel_idx]) * margin
            if channel_max > xlim[1]:
                xlim[1] = channel_max
    
    xlim = tuple(xlim)
    return xlim

def _parse_kwargs(spikes_idxs, n_channels, **kwargs):
    # Basic checks
    kwargs_list = [
        {'key': 'ax', 'default': None, 'type': None},
        {'key': 'boxoff', 'default': True, 'type': bool},
        {'key': 'channels_height', 'default': 0.8, 'type': float},
        {'key': 'channels_labels', 'default': [str(i) for i in range(n_channels)], 'type': tuple},
        {'key': 'color', 'default': ['black' for _ in range(n_channels)], 'type': list},
        {'key': 'dpi', 'default': 300, 'type': float},
        {'key': 'figsize', 'default': (16, 0.25 * n_channels), 'type': tuple},
        {'key': 'linewidth', 'default': 0.5, 'type': float},
        {'key': 'num', 'default': None, 'type': str},
        {'key': 'sampling_time', 'default': None, 'type': float},
        {'key': 'title', 'default': 'Raster Plot', 'type': str},
        {'key': 'reverse', 'default': False, 'type': bool},
        {'key': 'xlabel', 'default': 'Time (s)' if kwargs.get('sampling_time', None) is not None else 'Samples', 'type': str},
        {'key': 'xlim', 'default': _compute_default_xlim(spikes_idxs, n_channels), 'type': tuple},
        {'key': 'ylabel', 'default': 'Trials', 'type': str}
    ]
    kwargs = utils.check_kwargs_list(kwargs_list, **kwargs)

    # Additional checks
    if (kwargs.get('channels_labels') is not None) and (len(kwargs.get('channels_labels')) != n_channels):
        raise ValueError("'channels_labels' expected to contain " + str(n_channels) + " elements, received " + str(len(kwargs.get('channels_labels'))))
    if len(kwargs.get('color')) != n_channels:
        if len(kwargs.get('color')) == 1:
            kwargs['color'] = [kwargs.get('color')[0] for _ in range(n_channels)]
        else:
            raise ValueError("'color' expected to contain " + str(n_channels) + " elements, received " + str(len(kwargs.get('color'))))
    
    return kwargs

def plot_raster(spikes, **kwargs):
    '''
    Plot a raster plot given the spikes of different trials or channels as input.

    Parameters
    ----------
    spikes : ndarray or list of ndarray
        An array containing the detected spikes. It can be expressed both
        as a spike train or the indices at which spikes occur. Multiple
        channels or trials may be passed as a list of ndarray.
    '''
    if isinstance(spikes, np.ndarray) and (len(spikes.squeeze().shape)) == 1:
        spikes = [spikes]

    n_channels = len(spikes)

    for channel_idx in range(n_channels):
        if spikes[channel_idx].dtype == 'bool':
            spikes[channel_idx] = utils.convert_train_to_idxs(spikes[channel_idx])
        
        spikes[channel_idx] = spikes[channel_idx].squeeze()
        spikes[channel_idx] = kwargs.get('sampling_time') * spikes[channel_idx] if kwargs.get('sampling_time') is not None else spikes[channel_idx]

    spikes_idxs = spikes

    kwargs = _parse_kwargs(spikes_idxs, n_channels, **kwargs)

    if kwargs.get('ax') is None:
        plt.figure(num=kwargs.get('num'), figsize=kwargs.get('figsize'), dpi=kwargs.get('dpi'))
        ax = plt.gca()
    else:
        ax = kwargs.get('ax')

    if kwargs.get('reverse'):
        spikes_idxs = list(reversed(spikes_idxs))
        kwargs['channels_labels'] =list(reversed(kwargs.get('channels_labels')))
        kwargs['color'] = list(reversed(kwargs.get('color')))

    ax.eventplot(spikes_idxs, orientation='horizontal', linelengths=kwargs.get('channels_height'), linewidth=kwargs.get('linewidth'), color=kwargs.get('color'))

    ax.set_title(kwargs.get('title'))
    ax.set_xlabel(kwargs.get('xlabel'))
    ax.set_ylabel(kwargs.get('ylabel'))

    ax.set_xlim(kwargs.get('xlim'))
    
    if kwargs.get('channels_labels') is not None:
        if n_channels <= 10:
            yticks = range(0, n_channels, 1)
        elif (n_channels > 10) and (n_channels <= 50):
            yticks = range(0, n_channels, 5)
        else:
            yticks = range(0, n_channels, 10)

        if kwargs.get('reverse'):
            yticks = [n_channels - item - 1 for item in yticks]

        ax.set_yticks(yticks)
        ax.set_yticklabels([kwargs.get('channels_labels')[i] for i in yticks])
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])

    if kwargs.get('sampling_time') is None:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if kwargs.get('boxoff') is True:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False) 

    return