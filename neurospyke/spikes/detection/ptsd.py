import numpy as np
from scipy.signal import argrelmax, argrelmin
from ... import utils

def PTSD_samples(data, threshold, refractory_period, peak_lifetime_period, overshoot):
    # Cast data type to float
    data = data.astype(np.float64)

    spikes_idxs = []
    spikes_values = []

    max_idxs = argrelmax(data)[0]
    max_values = data[max_idxs]

    for i in range(len(max_idxs)):
        if max_idxs[i] + peak_lifetime_period <= len(data):
            window_data = data[np.arange(max_idxs[i], max_idxs[i] + peak_lifetime_period)]
        else:
            window_data = data[np.arange(max_idxs[i], len(data))]
        
        min_idx = argrelmin(window_data)[0]
        min_value = data[min_idx]

        if len(min_idx) > 1:
            min_idx = min_idx[0]
            min_value = min_value[0]
        elif len(min_idx) == 1:
            pass
        else:
            if max_idxs[i] + peak_lifetime_period + overshoot <= len(data):
                window_data = data[np.arange(max_idxs[i], max_idxs[i] + peak_lifetime_period + overshoot)]
            else:
                window_data = data[np.arange(max_idxs[i], len(data))]
            
            min_idx = argrelmin(window_data)[0]
            min_value = data[min_idx]

            if len(min_idx) > 1:
                min_idx = min_idx[0]
                min_value = min_value[0]
            elif len(min_idx) == 1:
                pass
            else:
                max_values[i] = None

        if max_values[i] is not None:
            if abs(max_values[i] - min_value) >= threshold:
                if len(spikes_idxs) == 0:
                    spikes_idxs.append(max_idxs[i])
                    spikes_values.append(max_values[i])
                elif abs(spikes_idxs[-1] - i) > refractory_period:
                    spikes_idxs.append(max_idxs[i])
                    spikes_values.append(max_values[i])

    spikes_idxs = np.array(spikes_idxs, dtype=np.int64)
    spikes_values = np.array(spikes_values, dtype=np.float64)
    return spikes_idxs, spikes_values

def PTSD(data, sampling_time, threshold, refractory_period, peak_lifetime_period, overshoot):
    # Convert all parameters from time-domain to samples
    refractory_period = utils.get_in_samples(refractory_period, sampling_time)
    peak_lifetime_period = utils.get_in_samples(peak_lifetime_period, sampling_time)
    overshoot = utils.get_in_samples(overshoot, sampling_time)

    spikes_idxs, spikes_values = PTSD_samples(data, threshold, refractory_period, peak_lifetime_period, overshoot)

    return spikes_idxs, spikes_values