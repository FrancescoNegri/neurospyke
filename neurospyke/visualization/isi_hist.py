import numpy as np
import matplotlib.pyplot as plt

def plot_ISI_hist(ISI, bins, plot_title='ISI Histogram', range=None):
    plt.figure(figsize=(10, 5))
    
    if range is not None:
        range = (range[0], range[1])

    plt.hist(ISI, bins=bins, range=range)
    
    plt.title(plot_title)
    plt.xlabel('ISI (s)')
    plt.ylabel('Spikes Count')

    ax = plt.gca()
    ax.set_xlim([0, None])

    return