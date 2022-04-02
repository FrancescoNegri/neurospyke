import numpy as np
import matplotlib.pyplot as plt

def plot_PTSH(spikes_count, window_length, is_barplot=False, plot_title='PTSH Histogram'):
    plt.figure(figsize=(10, 5))
    bins = np.size(spikes_count, axis=0)
    
    if is_barplot is True:
        plt.bar(np.arange(bins), spikes_count, align='edge')
    else:
        plt.fill_between(np.arange(bins), spikes_count, alpha=0.25)
        plt.plot(spikes_count, linewidth=2)

    plt.title(plot_title)
    plt.xlabel('Time from stimulus (s)')
    plt.ylabel('Spikes Count')

    ax = plt.gca()
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    xticks = np.array(ax.get_xticks())
    xticks_idxs = np.where(ax.get_xticks() <= bins)
    xticklabels = np.array([np.round(label * window_length / bins, 5) for label in xticks])
    ax.set_xticks(xticks[xticks_idxs])
    ax.set_xticklabels(xticklabels[xticks_idxs])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return
