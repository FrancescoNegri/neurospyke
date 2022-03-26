import numpy as np
import matplotlib.pyplot as plt

def _parse_kwargs(n_channels, **kwargs):
    if 'channel_height' not in kwargs:
        kwargs['channel_height'] = 0.5
    if 'dpi' not in kwargs:
        kwargs['dpi'] = 300
    if 'figsize' not in kwargs:
        kwargs['figsize'] = (16, kwargs.get('channel_height') * n_channels)
    if 'linewidth' not in kwargs:
        kwargs['linewidth'] = 0.5
    if 'vertical_spacing' not in kwargs:
        kwargs['vertical_spacing'] = 0.25

    return kwargs

def _parse_parameters(spikes_idxs, sampling_time, channel_labels):
    if type(spikes_idxs) is not list:
        spikes_idxs = [spikes_idxs]
    spikes_idxs = np.array(spikes_idxs, dtype=object)

    if (channel_labels is not None) and (type(channel_labels) is not list):
        channel_labels = [channel_labels]

    return spikes_idxs, sampling_time, channel_labels

def plot_spike_train(spikes_idxs, sampling_time, channel_labels=None, plot_title='Spike Train', **kwargs):    
    spikes_idxs, sampling_time, channel_labels = _parse_parameters(spikes_idxs, sampling_time, channel_labels)
    n_channels = spikes_idxs.shape[0]

    kwargs = _parse_kwargs(n_channels, **kwargs)
    channel_height = kwargs.get('channel_height')

    plt.figure(figsize=kwargs.get('figsize'), dpi=kwargs.get('dpi'))

    spikes_times = sampling_time * spikes_idxs
    
    for channel_idx in np.arange(n_channels):
        for spike_idx in np.arange(len(spikes_idxs[channel_idx])):
            y = [channel_idx * channel_height, channel_idx * channel_height + channel_height * (1 - kwargs.get('vertical_spacing'))]
            x = [spikes_times[channel_idx][spike_idx], spikes_times[channel_idx][spike_idx]]
            plt.plot(x, y, color='black', linewidth=kwargs.get('linewidth'))

    plt.title(plot_title)
    plt.xlabel('Time (s)')
    plt.ylabel('Channels')

    ax = plt.gca()
    ax.set_xlim(0, np.amax([spikes_times[channel_idx][-1] for channel_idx in np.arange(n_channels)]) * 1.01)
    ax.set_ylim(0, channel_height * n_channels)
    
    yticks = np.arange(channel_height / 2, n_channels * channel_height + channel_height / 2, channel_height)
    ax.set_yticks(yticks)
    if channel_labels is None or (len(channel_labels) != n_channels):
        channel_labels = [str(i) for i in np.arange(1, n_channels + 1, 1)]
    ax.set_yticklabels(channel_labels)
    ax.tick_params(axis='y', which='both', length=0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return