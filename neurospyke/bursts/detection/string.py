import numpy as np
from ... import utils

def string_method(spikes_idxs, sampling_time, min_spikes, max_ISI, min_IBI=None):
    max_ISI = utils.get_in_samples(max_ISI, sampling_time)

    bursts_starts_idxs = []
    bursts_ends_idxs = []

    current_burst_length = 0
    current_burst_start = None

    for i in range(np.size(spikes_idxs)-1):
        if spikes_idxs[i+1] - spikes_idxs[i] > max_ISI:
            if current_burst_length > min_spikes:
                bursts_starts_idxs.append(current_burst_start)
                bursts_ends_idxs.append(spikes_idxs[i])
                current_burst_length = 0
                current_burst_start = None
            else:
                current_burst_length = 0
                current_burst_start = None
        else:
            if current_burst_start is None:
                current_burst_start = spikes_idxs[i]
            current_burst_length = current_burst_length + 1

    bursts_starts_idxs = np.array(bursts_starts_idxs, dtype=np.int64)
    bursts_ends_idxs = np.array(bursts_ends_idxs, dtype=np.int64)

    return bursts_starts_idxs, bursts_ends_idxs
