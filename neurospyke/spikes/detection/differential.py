import math
import numpy as np
from ... import utils

def differential_threshold(data, sampling_time, threshold, window_length):
    # Cast data type to float
    data = data.astype(np.float64)

    # Convert all parameters from time-domain to samples
    window_length = utils.get_in_samples(window_length, sampling_time)
    n_windows = math.floor(len(data) / window_length)

    spikes_idxs = []
    spikes_values = []

    for i in range(n_windows):
        window_idx = window_length * i
        window_data = data[np.arange(window_idx, window_idx+window_length, 1)]
        max_value = np.amax(window_data)
        min_value = np.amin(window_data)

        if abs(max_value - min_value) >= threshold:
            spikes_idxs.append(np.argmax(window_data)+window_idx)
            spikes_values.append(max_value)

    return spikes_idxs, spikes_values