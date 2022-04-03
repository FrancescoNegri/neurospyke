import numpy as np
import matplotlib.pyplot as plt
from .. import utils

def _parse_kwargs(**kwargs):
    kwargs_list = [
        {'key': 'alpha', 'default': 0.25, 'type': float},
        {'key': 'color', 'default': '#1f77b4', 'type': str},
        {'key': 'dpi', 'default': 100, 'type': float},
        {'key': 'figsize', 'default': (6, 3), 'type': tuple},
        {'key': 'is_barplot', 'default': False, 'type': bool},
        {'key': 'linewidth', 'default': 2, 'type': float},
        {'key': 'plot_title', 'default': 'PTSH Histogram', 'type': str}
    ]
    kwargs = utils.check_kwargs_list(kwargs_list, **kwargs)

    return kwargs


def plot_PTSH(spikes_count, window_length, **kwargs):
    kwargs = _parse_kwargs(**kwargs)

    plt.figure(figsize=kwargs.get('figsize'), dpi=kwargs.get('dpi'))
    bins = np.size(spikes_count, axis=0)
    
    if kwargs.get('is_barplot') is True:
        plt.bar(np.arange(bins), spikes_count, align='edge', color=kwargs.get('color'))
    else:
        plt.fill_between(np.arange(bins), spikes_count, facecolor=kwargs.get('color'), alpha=kwargs.get('alpha'))
        plt.plot(spikes_count, linewidth=kwargs.get('linewidth'), color=kwargs.get('color'))

    plt.title(kwargs.get('plot_title'))
    plt.xlabel('Time from stimulus (s)')
    plt.ylabel('Spikes Count')

    ax = plt.gca()
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    xticks = np.array(ax.get_xticks())
    xticks_idxs = np.where(ax.get_xticks() <= bins)
    xticklabels = np.array([np.round(label * window_length / bins, 5) for label in xticks])
    ax.set_xticks(xticks[xticks_idxs])
    ax.set_xticklabels(xticklabels[xticks_idxs])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return
