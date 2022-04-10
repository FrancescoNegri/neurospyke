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
        {'key': 'range', 'default': None, 'type': tuple},
        {'key': 'sampling_time', 'default': None, 'type': float},
        {'key': 'title', 'default': 'ISI Histogram', 'type': str},
        {'key': 'xlabel', 'default': 'ISI (s)' if kwargs.get('sampling_time', None) is not None else 'ISI (samples)', 'type': str},
        {'key': 'xlim', 'default': (0, None), 'type': tuple},
        {'key': 'ylabel', 'default': 'Spikes Count', 'type': str},
        {'key': 'ylim', 'default': (0, None), 'type': tuple}
    ]
    kwargs = utils.check_kwargs_list(kwargs_list, **kwargs)

    return kwargs

def plot_ISIH(ISI, bins, **kwargs):
    kwargs = _parse_kwargs(**kwargs)

    plt.figure(num=kwargs.get('num'), figsize=kwargs.get('figsize'), dpi=kwargs.get('dpi'))
    hist = np.histogram(ISI, bins=bins, range=kwargs.get('range'))
    bins = [(hist[1][i] + hist[1][i+1]) / 2 for i in range(np.size(hist[1]) - 1)]
    spikes_count = hist[0]

    if kwargs.get('normalize') is True:
        spikes_count = spikes_count/np.sum(spikes_count)

    if kwargs.get('barplot') is True:
        plt.bar(bins, spikes_count, width=(bins[1] - bins[0]), align='edge', color=kwargs.get('color'))
    else:
        plt.fill_between(bins, spikes_count, facecolor=kwargs.get('color'), alpha=kwargs.get('alpha'))
        plt.plot(bins, spikes_count, linewidth=kwargs.get('linewidth'), color=kwargs.get('color'))

    plt.title(kwargs.get('title'))
    plt.xlabel(kwargs.get('xlabel'))
    plt.ylabel(kwargs.get('ylabel'))

    ax = plt.gca()
    ax.set_xlim(kwargs.get('xlim'))
    ax.set_ylim(kwargs.get('ylim'))

    if kwargs.get('sampling_time') is None:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if kwargs.get('boxoff') is True:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    return