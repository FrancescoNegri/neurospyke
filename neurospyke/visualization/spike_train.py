import numpy as np
import matplotlib.pyplot as plt

def _check_kwargs_list(kwargs_list, **kwargs):
    for kwarg in kwargs_list:
        kwargs[kwarg['key']] = kwargs.get(kwarg['key'], kwarg['default'])

        if kwarg['type'] is not None:
            try:
                kwargs[kwarg['key']] = kwarg['type'](kwargs.get(kwarg['key']))
            except:
                raise TypeError("'" + kwarg['key'] + "' expected to be '" + kwarg['type'].__name__ + "', received '" + str(type(kwargs.get(kwarg['key'])).__name__) + "'")

    return kwargs

def _parse_kwargs(spikes_times, n_channels, **kwargs):
    # Basic checks
    kwargs_list = [
        {'key': 'channel_height', 'default': 0.5, 'type': float},
        {'key': 'color', 'default': 'black', 'type': None},
        {'key': 'dpi', 'default': 300, 'type': None},
        {'key': 'linewidth', 'default': 0.5, 'type': None},
        {'key': 'plot_title', 'default': 'Spike Train', 'type': str},
        {'key': 'reverse', 'default': False, 'type': bool},
        {'key': 'vertical_spacing', 'default': 0.25, 'type': float},
        {'key': 'xlim', 'default': (0, np.amax([spikes_times[channel_idx][-1] for channel_idx in np.arange(n_channels)]) * 1.01), 'type': None}
    ]
    kwargs = _check_kwargs_list(kwargs_list, **kwargs)

    # Additional checks
    if kwargs.get('vertical_spacing') < 0 or kwargs.get('vertical_spacing') > 1:
        raise ValueError("'vertical_spacing' expected to be a value between 0 and 1, received " + str(kwargs.get('vertical_spacing')))
    
    # Dependant checks
    kwargs_list = [
        {'key': 'figsize', 'default': (16, kwargs.get('channel_height') * n_channels), 'type': None}
    ]
    kwargs = _check_kwargs_list(kwargs_list, **kwargs)
    
    return kwargs

def _parse_parameters(spikes_idxs, sampling_time, channel_labels):
    if type(spikes_idxs) is not list:
        spikes_idxs = [spikes_idxs]
    spikes_idxs = np.array(spikes_idxs, dtype=object)

    if (channel_labels is not None) and (type(channel_labels) is not list):
        channel_labels = [channel_labels]

    return spikes_idxs, sampling_time, channel_labels

def plot_spike_train(spikes_idxs, sampling_time, channel_labels=None, **kwargs):    
    spikes_idxs, sampling_time, channel_labels = _parse_parameters(spikes_idxs, sampling_time, channel_labels)
    n_channels = spikes_idxs.shape[0]
    spikes_times = sampling_time * spikes_idxs

    kwargs = _parse_kwargs(spikes_times, n_channels, **kwargs)

    plt.figure(figsize=kwargs.get('figsize'), dpi=kwargs.get('dpi'))

    if (channel_labels is None) or (len(channel_labels) != n_channels):
        channel_labels = [str(i) for i in np.arange(1, n_channels + 1, 1)]

    if (kwargs.get('reverse') is True) and (n_channels > 1):
        spikes_times = np.flip(spikes_times, axis=0)
        channel_labels = np.flip(channel_labels)
    
    for channel_idx in np.arange(n_channels):
        for spike_idx in np.arange(len(spikes_times[channel_idx])):
            y = [channel_idx * kwargs.get('channel_height'), channel_idx * kwargs.get('channel_height') + kwargs.get('channel_height') * (1 - kwargs.get('vertical_spacing'))]
            x = [spikes_times[channel_idx][spike_idx], spikes_times[channel_idx][spike_idx]]
            plt.plot(x, y, color=kwargs.get('color'), linewidth=kwargs.get('linewidth'))

    plt.title(kwargs.get('plot_title'))
    plt.xlabel('Time (s)')
    plt.ylabel('Channels')

    ax = plt.gca()
    
    ax.set_xlim(kwargs.get('xlim')[0], kwargs.get('xlim')[1])
    ax.set_ylim(0, kwargs.get('channel_height') * n_channels)
    
    yticks = np.arange(kwargs.get('channel_height') / 2, round(n_channels * kwargs.get('channel_height') + kwargs.get('channel_height') / 2, 8), kwargs.get('channel_height'))
    ax.set_yticks(yticks)
    ax.set_yticklabels(channel_labels)
    ax.tick_params(axis='y', which='both', length=0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return