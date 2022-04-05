import numpy as np
import matplotlib.pyplot as plt

def plot_spikes(data, sampling_time, spikes_idxs, plot_title='Spike Plot'):
    plt.figure(figsize=(10, 5))
    times = sampling_time*np.arange(0, len(data), 1)
    spikes_times = sampling_time*spikes_idxs
    
    plt.plot(times, data, linewidth=0.25)
    plt.plot(spikes_times, data[spikes_idxs], color='red', marker='*', markersize=3, linestyle='None')

    plt.title(plot_title)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (µV)')

    ax = plt.gca()
    ax.set_xlim(0, times[-1])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return