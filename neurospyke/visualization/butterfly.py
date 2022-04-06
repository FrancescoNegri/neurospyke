import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .. import utils

def _parse_kwargs(**kwargs):
    kwargs_list = [
        {'key': 'boxoff', 'default': True, 'type': bool},
        {'key': 'dpi', 'default': 100, 'type': float},
        {'key': 'figsize', 'default': (6, 3), 'type': tuple},
        {'key': 'linewidth', 'default': 0.5, 'type': float},
        {'key': 'num', 'default': None, 'type': str},
        {'key': 'sampling_time', 'default': None, 'type': float},
        {'key': 'title', 'default': 'Butterfly Plot', 'type': str},
        {'key': 'window_length', 'default': 0.001 if kwargs.get('sampling_time', None) is not None else 20, 'type': float},
        {'key': 'xlabel', 'default': 'Time distance from spike (ms)' if kwargs.get('sampling_time', None) is not None else 'Samples', 'type': str},
        {'key': 'xlim', 'default': None, 'type': tuple},
        {'key': 'ylabel', 'default': 'Voltage (µV)', 'type': str},
        {'key': 'ylim', 'default': None, 'type': tuple}
    ]
    kwargs = utils.check_kwargs_list(kwargs_list, **kwargs)

    return kwargs

def plot_butterfly(data, spikes_idxs, **kwargs):
    kwargs = _parse_kwargs(**kwargs)

    plt.figure(num=kwargs.get('num'), figsize=kwargs.get('figsize'), dpi=kwargs.get('dpi'))
    window_half_length = utils.get_in_samples(kwargs.get('window_length') / 2, kwargs.get('sampling_time'))
    window_times = kwargs.get('sampling_time') * 1000 * np.arange(-window_half_length, window_half_length + 1, 1) if kwargs.get('sampling_time') is not None else np.arange(-window_half_length, window_half_length + 1, 1)

    for i in range(len(spikes_idxs)):
        window_data = data[range(spikes_idxs[i] - window_half_length, spikes_idxs[i] + window_half_length + 1, 1)]
        plt.plot(window_times, window_data, linewidth=kwargs.get('linewidth'))

    plt.title(kwargs.get('title'))
    plt.xlabel(kwargs.get('xlabel'))
    plt.ylabel(kwargs.get('ylabel'))

    ax = plt.gca()
    ax.set_xlim(window_times[0], window_times[-1]) if kwargs.get('xlim') is None else ax.set_xlim(kwargs.get('xlim'))
    ax.set_ylim(None, None) if kwargs.get('ylim') is None else ax.set_ylim(kwargs.get('ylim'))

    if kwargs.get('sampling_time') is None:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if kwargs.get('boxoff') is True:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    return