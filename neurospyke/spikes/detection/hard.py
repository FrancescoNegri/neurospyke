import numpy as np
from ... import utils
from scipy.signal import find_peaks

def hard_threshold_local_maxima_samples(data, threshold, refractory_period, use_abs=False):
    if use_abs is True:
        data = abs(data)
    
    spikes_idxs, _ = find_peaks(data, height=threshold, distance=refractory_period)
    spikes_values = data[spikes_idxs]

    spikes_idxs = np.array(spikes_idxs, dtype=np.int64)
    spikes_values = np.array(spikes_values, dtype=np.float64)
    return spikes_idxs, spikes_values

def hard_threshold_local_maxima(data, sampling_time, threshold, refractory_period, use_abs=False):
    # Convert all parameters from time-domain to samples
    refractory_period = utils.get_in_samples(refractory_period, sampling_time)
    
    spikes_idxs, spikes_values = hard_threshold_local_maxima_samples(data, threshold, refractory_period, use_abs)

    return spikes_idxs, spikes_values