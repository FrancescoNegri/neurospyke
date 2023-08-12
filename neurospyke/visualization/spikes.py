import matplotlib.pyplot as plt
import numpy as np

from .. import utils
from .raw_data import plot_raw_data

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

def plot_spikes(data:np.ndarray, spikes:np.ndarray, **kwargs):
    kwargs = _parse_kwargs(**kwargs)

    # Cast data type to float
    data = data.astype(np.float64).squeeze()
    spikes = spikes.squeeze()

    if spikes.dtype == 'bool':
        spikes_idxs = utils.convert_train_to_idxs(spikes)
    else:
        spikes_idxs = spikes

    if kwargs.get('ax') is None:
        plt.figure(num=kwargs.get('num'), figsize=kwargs.get('figsize'), dpi=kwargs.get('dpi'))
        kwargs['ax'] = plt.gca()
    
    ax = kwargs.get('ax')

    plot_raw_data(data, **kwargs)

    spikes_times = kwargs.get('sampling_time') * spikes_idxs if kwargs.get('sampling_time') is not None else spikes_idxs
    ax.plot(spikes_times, data[spikes_idxs], color=kwargs.get('markercolor'), marker=kwargs.get('marker'), markersize=kwargs.get('markersize'), linestyle='None')

    return