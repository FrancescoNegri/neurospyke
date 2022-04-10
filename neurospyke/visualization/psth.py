import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .. import utils

def _parse_kwargs(**kwargs):
    kwargs_list = [
        {'key': 'alpha', 'default': 0.25, 'type': float},
        {'key': 'barplot', 'default': False, 'type': bool},
        {'key': 'boxoff', 'default': True, 'type': bool},
        {'key': 'color', 'default': '#1f77b4', 'type': str},
        {'key': 'dpi', 'default': 100, 'type': float},
        {'key': 'figsize', 'default': (6, 3), 'type': tuple},
        {'key': 'linewidth', 'default': 2, 'type': float},
        {'key': 'normalize', 'default': False, 'type': bool},
        {'key': 'num', 'default': None, 'type': str},
        {'key': 'sampling_time', 'default': None, 'type': float},
        {'key': 'title', 'default': 'PSTH Histogram', 'type': str},
        {'key': 'xlabel', 'default': 'Time from stimulus (s)' if kwargs.get('sampling_time', None) is not None else 'Samples from stimulus', 'type': str},
        {'key': 'xlim', 'default': (0, None), 'type': tuple},
        {'key': 'ylabel', 'default': 'Spikes Count', 'type': str},
        {'key': 'ylim', 'default': (0, None), 'type': tuple}
    ]
    kwargs = utils.check_kwargs_list(kwargs_list, **kwargs)

    return kwargs

def plot_PSTH(spikes_count, window_length, **kwargs):
    kwargs = _parse_kwargs(**kwargs)

    plt.figure(num=kwargs.get('num'), figsize=kwargs.get('figsize'), dpi=kwargs.get('dpi'))
    bins = np.size(spikes_count, axis=0)

    if kwargs.get('normalize') is True:
        spikes_count = spikes_count/np.sum(spikes_count)
    
    if kwargs.get('barplot') is True:
        plt.bar(np.arange(bins), spikes_count, width=1, align='edge', color=kwargs.get('color'))
    else:
        plt.fill_between(np.arange(bins), spikes_count, facecolor=kwargs.get('color'), alpha=kwargs.get('alpha'))
        plt.plot(spikes_count, linewidth=kwargs.get('linewidth'), color=kwargs.get('color'))

    plt.title(kwargs.get('title'))
    plt.xlabel(kwargs.get('xlabel'))
    plt.ylabel(kwargs.get('ylabel'))

    ax = plt.gca()
    ax.set_xlim(kwargs.get('xlim'))
    ax.set_ylim(kwargs.get('ylim'))

    xticks = np.array(ax.get_xticks())
    xticks_idxs = np.where(ax.get_xticks() <= bins)
    xticklabels = np.array([np.round(label * window_length / bins, 5) for label in xticks])
    ax.set_xticks(xticks[xticks_idxs])
    ax.set_xticklabels(xticklabels[xticks_idxs])

    if kwargs.get('sampling_time') is None:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if kwargs.get('boxoff') is True:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    return
