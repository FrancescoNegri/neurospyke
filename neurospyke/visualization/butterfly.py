import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator
from .. import utils
from ..spikes.sorting import get_waveforms 

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

def plot_butterfly(data:np.ndarray, spikes:np.ndarray, **kwargs):
    '''
    Plot the so-called Butterfly Plot, displaying all the detected
    spikes on the same axes.
    The spikes are centered in zero and a user-specified window is
    taken to visualize the spike shape before and after the event,
    with parameters specified either in the time domain or in samples.

    Parameters
    ----------
    data : numpy.ndarray
        The array of recorded data.
    spikes : numpy.ndarray
        An array containing the detected spikes. It can be express both
        as a spike train or as a list of the indices at which spikes occur.
    window_length : float
        The length for the detection window to employ while plotting
        spikes, expressed in seconds or samples.
    sampling_time : float, optional
        The sampling time for the recorded data. If specified, the algorithm
        will work in the time domain (the other parameters should then be
        specified in seconds). Otherwise, it will work with samples.
    '''
    kwargs = _parse_kwargs(**kwargs)

    # Cast data type to float
    data = data.astype(np.float64).squeeze()
    spikes = spikes.squeeze()

    if spikes.dtype == 'bool':
        spikes_idxs = utils.convert_train_to_idxs(spikes)
    else:
        spikes_idxs = spikes

    plt.figure(num=kwargs.get('num'), figsize=kwargs.get('figsize'), dpi=kwargs.get('dpi'))
    window_half_length = utils.get_in_samples(kwargs.get('window_length') / 2, kwargs.get('sampling_time'))
    window_times = kwargs.get('sampling_time') * 1000 * np.arange(-window_half_length, window_half_length, 1) if kwargs.get('sampling_time') is not None else np.arange(-window_half_length, window_half_length, 1)
    window_times = np.tile(window_times, (np.size(spikes_idxs), 1))

    plt.plot(window_times.T, np.transpose(get_waveforms(data, spikes_idxs, **kwargs)), linewidth=kwargs.get('linewidth'))

    plt.title(kwargs.get('title'))
    plt.xlabel(kwargs.get('xlabel'))
    plt.ylabel(kwargs.get('ylabel'))

    ax = plt.gca()
    ax.set_xlim(window_times[0, 0], window_times[0, -1]) if kwargs.get('xlim') is None else ax.set_xlim(kwargs.get('xlim'))
    ax.set_ylim(None, None) if kwargs.get('ylim') is None else ax.set_ylim(kwargs.get('ylim'))

    if kwargs.get('sampling_time') is None:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if kwargs.get('boxoff') is True:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    return