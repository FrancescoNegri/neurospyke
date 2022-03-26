import numpy as np
import matplotlib.pyplot as plt

def _parse_parameters(spikes_idxs, sampling_time, channel_labels, channel_height):
    if type(spikes_idxs) is not list:
        spikes_idxs = [spikes_idxs]
    spikes_idxs = np.array(spikes_idxs, dtype=object)

    if (channel_labels is not None) and (type(channel_labels) is not list):
        channel_labels = [channel_labels]

    return spikes_idxs, sampling_time, channel_labels, channel_height

def plot_spike_train(spikes_idxs, sampling_time, channel_labels=None, channel_height=0.5, plot_title='Spike Train'):    
    spikes_idxs, sampling_time, channel_labels, channel_height = _parse_parameters(spikes_idxs, sampling_time, channel_labels, channel_height)

    n_channels = spikes_idxs.shape[0]

    fig_height = channel_height * n_channels
    plt.figure(figsize=(16, fig_height), dpi=300)

    spikes_times = sampling_time * spikes_idxs
    
    for channel_idx in np.arange(n_channels):
        for spike_idx in np.arange(len(spikes_idxs[channel_idx])):
            y = [channel_idx * channel_height, channel_idx * channel_height + channel_height * 0.75]
            x = [spikes_times[channel_idx][spike_idx], spikes_times[channel_idx][spike_idx]]
            plt.plot(x, y, color='black', linewidth=0.5)

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