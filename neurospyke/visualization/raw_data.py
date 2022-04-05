import numpy as np
import matplotlib.pyplot as plt

    times = sampling_time * np.arange(0, len(data), 1)
    
    plt.plot(times, data, linewidth=0.25)

    plt.title(plot_title)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (µV)')

    ax = plt.gca()
    ax.set_xlim(0, times[-1])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return