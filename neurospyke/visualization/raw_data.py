import numpy as np
import matplotlib.pyplot as plt
from .. import utils

def _parse_kwargs(**kwargs):
    kwargs_list = [
        {'key': 'color', 'default': '#1f77b4', 'type': str},
        {'key': 'dpi', 'default': 100, 'type': float},
        {'key': 'figsize', 'default': (6, 3), 'type': tuple},
        {'key': 'linewidth', 'default': 0.25, 'type': float},
        {'key': 'title', 'default': 'Raw Data Plot', 'type': str}
    ]
    kwargs = utils.check_kwargs_list(kwargs_list, **kwargs)

    return kwargs

def plot_raw_data(data, sampling_time, **kwargs):
    kwargs = _parse_kwargs(**kwargs)

    plt.figure(figsize=kwargs.get('figsize'), dpi=kwargs.get('dpi'))
    times = sampling_time * np.arange(0, len(data), 1)
    
    plt.plot(times, data, linewidth=kwargs.get('linewidth'), color=kwargs.get('color'))

    plt.title(kwargs.get('title'))
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (µV)')

    ax = plt.gca()
    ax.set_xlim(0, times[-1])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return