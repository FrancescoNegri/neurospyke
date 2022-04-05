import math
import numpy as np
import matplotlib.pyplot as plt
from .. import utils

def _parse_kwargs(**kwargs):
    kwargs_list = [
        {'key': 'dpi', 'default': 100, 'type': float},
        {'key': 'figsize', 'default': (6, 3), 'type': tuple},
        {'key': 'linewidth', 'default': 0.5, 'type': float},
        {'key': 'title', 'default': 'Butterfly Plot', 'type': str},
        {'key': 'window_length', 'default': 0.001, 'type': float}
    ]
    kwargs = utils.check_kwargs_list(kwargs_list, **kwargs)

    return kwargs

def plot_butterfly(data, sampling_time, spikes_idxs, **kwargs):
    kwargs = _parse_kwargs(**kwargs)

    plt.figure(figsize=kwargs.get('figsize'), dpi=kwargs.get('dpi'))
    window_half_length = kwargs.get('window_length') / 2
    window_half_length_idx = math.floor(window_half_length / sampling_time)
    window_times = sampling_time * 1000 * np.arange(-window_half_length_idx, window_half_length_idx, 1)
    
    for i in range(len(spikes_idxs)):
        window_data = data[range(spikes_idxs[i] - window_half_length_idx, spikes_idxs[i] + window_half_length_idx, 1)]
        plt.plot(window_times, window_data, linewidth=kwargs.get('linewidth'))

    plt.title(kwargs.get('title'))
    plt.xlabel('Time distance from spike (ms)')
    plt.ylabel('Voltage (µV)')

    ax = plt.gca()
    ax.set_xlim(window_times[0], window_times[-1])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return