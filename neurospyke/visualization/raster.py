import numpy as np
import matplotlib.pyplot as plt
from .. import utils
from .spike_train import plot_spike_train

def plot_raster(spikes_obj, sampling_time, n_cols = 1, is_train = True, figsize=[12, 8], title='Stimulus-Related Raster Plots', xlim=None):
    if np.size(np.shape(spikes_obj)) == 2:
        spikes_obj = np.array([spikes_obj])
    
    n_channels = np.size(spikes_obj, 0)
    n_rows = int(np.ceil(n_channels/n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, constrained_layout=True, dpi=200)

    for channel_idx in np.arange(n_channels):
        spikes_matrix = spikes_obj[channel_idx]
        spikes_idxs = []
        for trial_idx in np.arange(np.size(spikes_matrix, 0)):
            if is_train:
                spikes_idxs.append(utils.convert_spike_train_to_spikes_idxs(spikes_matrix[trial_idx]))
            else:
                spikes_idxs.append(spikes_matrix[trial_idx])

        row = int(np.floor(channel_idx / n_cols))
        col = channel_idx % n_cols

        plot_spike_train(spikes_idxs, ax=axs[row, col], sampling_time=sampling_time, ylabel='Trials', channel_labels=None, reverse=True, title='Channel #' + str(channel_idx), vertical_spacing=0.1, linewidth=0.4, xlim=xlim)
    
    fig.suptitle(title, fontsize=16)