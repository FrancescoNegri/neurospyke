import numpy as np
import math
import matplotlib.pyplot as plt

def plot_butterfly(data, sampling_time, spikes_idxs, window_half_length, plot_title='Butterfly Plot'):
    plt.figure(figsize=(10, 5))
    window_half_length_idx = math.floor(window_half_length/sampling_time)
    window_times = sampling_time*np.arange(-window_half_length_idx, window_half_length_idx, 1)
    
    for i in range(len(spikes_idxs)):
        window_data = data[range(spikes_idxs[i]-window_half_length_idx, spikes_idxs[i]+window_half_length_idx, 1)]
        plt.plot(window_times, window_data, linewidth=0.5)

    plt.title(plot_title)
    plt.xlabel('Time distance from spike (s)')
    plt.ylabel('Voltage (µV)')

    ax = plt.gca()
    ax.set_xlim(window_times[0], window_times[-1])

    return