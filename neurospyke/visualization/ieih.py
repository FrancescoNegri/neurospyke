import matplotlib.pyplot as plt
import numpy as np

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
        {'key': 'title', 'default': 'IEI Histogram', 'type': str},
        {'key': 'xlabel', 'default': 'IEI (s)' if kwargs.get('sampling_time', None) is not None else 'IEI (samples)', 'type': str},
        {'key': 'xlim', 'default': (0, None), 'type': tuple},
        {'key': 'ylabel', 'default': 'Events Count', 'type': str},
        {'key': 'ylim', 'default': (0, None), 'type': tuple}
    ]
    kwargs = utils.check_kwargs_list(kwargs_list, **kwargs)

    return kwargs

def plot_IEIH(IEI:np.ndarray, bins:int = None, **kwargs):
    '''
    Plot an Inter-Event-Interval histogram, such as the Inter-Spike-Interval histogram.

    Parameters
    ----------
    IEI : ndarray
        The input Inter-Event-Interval to plot either in samples or in seconds.
    bins : int, default=None
        The number of equal-width bins for the histogram.
    sampling_time : float, optional
        The sampling time for the recorded data. If specified, the algorithm
        will work in the time domain (the other parameters should then be
        specified in seconds). Otherwise, it will work with samples.
    '''
    kwargs = _parse_kwargs(**kwargs)

    plt.figure(num=kwargs.get('num'), figsize=kwargs.get('figsize'), dpi=kwargs.get('dpi'))

    if bins is not None:
        hist = np.histogram(IEI, bins=bins, range=kwargs.get('range'))
    else:
        hist = np.histogram(IEI, range=kwargs.get('range'))
    
    bins = [(hist[1][i] + hist[1][i+1]) / 2 for i in range(np.size(hist[1]) - 1)]
    events_count = hist[0]

    if kwargs.get('normalize') is True:
        events_count = events_count/np.sum(events_count)

    if kwargs.get('barplot') is True:
        plt.bar(bins, events_count, width=(bins[1] - bins[0]), align='edge', color=kwargs.get('color'))
    else:
        plt.fill_between(bins, events_count, facecolor=kwargs.get('color'), alpha=kwargs.get('alpha'))
        plt.plot(bins, events_count, linewidth=kwargs.get('linewidth'), color=kwargs.get('color'))

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