import math
import numpy as np
from ... import utils

def PSTH_train(spike_train, stimuli_idxs, sampling_time, window_length, bins):
    window_length = utils.get_in_samples(window_length, sampling_time)
    
    spikes_count = []

    for idx in range(len(stimuli_idxs)):
        window_idxs = np.arange(stimuli_idxs[idx], stimuli_idxs[idx] + window_length, 1, dtype=np.int64)
        window_spike_train = spike_train[window_idxs]

        bin_size = math.floor(len(window_spike_train) / bins)

        current_count = []
        for bin_idx in range(bins):
            current_count.append(np.sum(window_spike_train[np.arange(bin_idx*bin_size, bin_idx*bin_size + bin_size, dtype=np.int64)]))

        spikes_count.append(current_count)

    avg_spikes_count = np.mean(spikes_count, axis=0)

    return avg_spikes_count, spikes_count

def PSTH(spikes_idxs, stimuli_idxs, sampling_time, window_length, bins):
    spike_train = utils.convert_spikes_idxs_to_spike_train(spikes_idxs, sampling_time)

    avg_spikes_count, spikes_count = PSTH_train(spike_train, stimuli_idxs, sampling_time, window_length, bins)

    return avg_spikes_count, spikes_count
