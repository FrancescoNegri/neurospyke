import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .. import utils

def _parse_kwargs(**kwargs):
    kwargs_list = [
        {'key': 'ax', 'default': None, 'type': None},
        {'key': 'boxoff', 'default': True, 'type': bool},
        {'key': 'color', 'default': '#1f77b4', 'type': str},
        {'key': 'dpi', 'default': 100, 'type': float},
        {'key': 'figsize', 'default': (6, 3), 'type': tuple},
        {'key': 'linewidth', 'default': 0.25, 'type': float},
        {'key': 'marker', 'default': '*', 'type': str},
        {'key': 'markercolor', 'default': 'red', 'type': str},
        {'key': 'markersize', 'default': 2, 'type': float},
        {'key': 'num', 'default': None, 'type': str},
        {'key': 'sampling_time', 'default': None, 'type': float},
        {'key': 'title', 'default': 'Spike Plot', 'type': str},
        {'key': 'xlabel', 'default': 'Time (s)' if kwargs.get('sampling_time', None) is not None else 'Samples', 'type': str},
        {'key': 'xlim', 'default': None, 'type': tuple},
        {'key': 'ylabel', 'default': 'Voltage (µV)', 'type': str},
        {'key': 'ylim', 'default': None, 'type': tuple}
    ]
    kwargs = utils.check_kwargs_list(kwargs_list, **kwargs)

    return kwargs

def plot_spikes(data, spikes_idxs, **kwargs):
    kwargs = _parse_kwargs(**kwargs)

    if kwargs.get('ax') is None:
        plt.figure(num=kwargs.get('num'), figsize=kwargs.get('figsize'), dpi=kwargs.get('dpi'))
        ax = plt.gca()
    else:
        ax = kwargs.get('ax')
    
    times = kwargs.get('sampling_time') * np.arange(0, len(data), 1) if kwargs.get('sampling_time') is not None else np.arange(0, len(data), 1)
    spikes_times = kwargs.get('sampling_time') * spikes_idxs if kwargs.get('sampling_time') is not None else spikes_idxs
    
    ax.plot(times, data, linewidth=kwargs.get('linewidth'), color=kwargs.get('color'))
    ax.plot(spikes_times, data[spikes_idxs], color=kwargs.get('markercolor'), marker=kwargs.get('marker'), markersize=kwargs.get('markersize'), linestyle='None')

    ax.set_title(kwargs.get('title'))
    ax.set_xlabel(kwargs.get('xlabel'))
    ax.set_ylabel(kwargs.get('ylabel'))

    ax.set_xlim(0, times[-1]) if kwargs.get('xlim') is None else ax.set_xlim(kwargs.get('xlim'))
    ax.set_ylim(None, None) if kwargs.get('ylim') is None else ax.set_ylim(kwargs.get('ylim'))

    if kwargs.get('sampling_time') is None:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if kwargs.get('boxoff') is True:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    return