import numpy as np
import matplotlib.pyplot as plt

def plot_raw_data(data, sampling_time, plot_title='Raw Data Plot'):
    plt.figure(figsize=(10, 5))
    times = sampling_time*np.arange(0, len(data), 1)
    
    plt.plot(times, data, linewidth=0.25)

    plt.title(plot_title)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (µV)')

    ax = plt.gca()
    ax.set_xlim(0, times[-1])

    return