import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator
from .. import utils

def _parse_xlim(xlim, duration, n_bins):
    if xlim[1] is not None:
        xlim = (xlim[0], xlim[1] * n_bins / duration)
    return xlim


def _parse_kwargs(**kwargs):
    kwargs_list = [
        {'key': 'alpha', 'default': 0.25, 'type': float},
        {'key': 'ax', 'default': None, 'type': None},
        {'key': 'barplot', 'default': False, 'type': bool},
        {'key': 'boxoff', 'default': True, 'type': bool},
        {'key': 'color', 'default': '#1f77b4', 'type': None},
        {'key': 'dpi', 'default': 100, 'type': float},
        {'key': 'figsize', 'default': (6, 3), 'type': tuple},
        {'key': 'linewidth', 'default': 2, 'type': float},
        {'key': 'normalize', 'default': True, 'type': bool},
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

def plot_PSTH(spikes_count, duration, **kwargs):
    kwargs = _parse_kwargs(**kwargs)

    spikes_count.squeeze()
    (n_trials, n_bins) = spikes_count.shape
    spikes_count = np.sum(spikes_count, axis=0) 

    if kwargs.get('ax') is None:
        plt.figure(num=kwargs.get('num'), figsize=kwargs.get('figsize'), dpi=kwargs.get('dpi'))
        kwargs['ax'] = plt.gca()
    
    ax = kwargs.get('ax')

    if kwargs.get('normalize') is True:
        spikes_count = spikes_count / n_trials
    
    if kwargs.get('barplot') is True:
        plt.bar(np.arange(n_bins), spikes_count, width=1, align='edge', color=kwargs.get('color'))
    else:
        plt.fill_between(np.arange(n_bins+1), np.insert(spikes_count, 0, 0), facecolor=to_rgba(kwargs.get('color'), kwargs.get('alpha')))
        plt.plot(np.insert(spikes_count, 0, 0), linewidth=kwargs.get('linewidth'), color=kwargs.get('color'))

    plt.title(kwargs.get('title'))
    plt.xlabel(kwargs.get('xlabel'))
    plt.ylabel(kwargs.get('ylabel'))

    ax = plt.gca()
    ax.set_xlim(_parse_xlim(kwargs.get('xlim'), duration, n_bins))
    ax.set_ylim(kwargs.get('ylim'))

    xticks = np.array(ax.get_xticks())
    xticks_idxs = np.where(ax.get_xticks() <= n_bins)
    xticklabels = np.array([np.round(label * duration / n_bins, 5) for label in xticks])
    ax.set_xticks(xticks[xticks_idxs])
    ax.set_xticklabels(xticklabels[xticks_idxs])

    if kwargs.get('sampling_time') is None:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if kwargs.get('boxoff') is True:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    return
