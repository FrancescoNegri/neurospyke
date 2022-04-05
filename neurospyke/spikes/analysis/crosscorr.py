import math
import numpy as np
from ... import utils

def cross_correlation(reference_spike_train, target_spike_train, tau, window_length, sampling_time):    
    reference_length = np.size(reference_spike_train, 0)
    target_length = np.size(target_spike_train, 0)
    
    if reference_length > target_length:
        target_spike_train = np.append(target_spike_train, np.zeros(reference_length-target_length))
    elif reference_length < target_length:
        reference_spike_train = np.append(reference_spike_train, np.zeros(target_length-reference_length))

    n_max = math.floor(window_length/tau)
    tau = utils.get_in_samples(tau, sampling_time)

    cross_correlation = np.zeros((n_max+1, 2))

    for n in range(n_max+1):
        zeros_array = np.zeros(n*tau)

        target_pos = np.append(zeros_array, target_spike_train)
        reference_pos = np.append(reference_spike_train, zeros_array)

        target_neg = np.append(target_spike_train, zeros_array)
        reference_neg = np.append(zeros_array, reference_spike_train)

        cross_correlation[n, 0] = np.sum(target_pos * reference_pos)
        cross_correlation[n, 1] = np.sum(target_neg * reference_neg)

    cross_correlation = np.append(np.flip(cross_correlation[np.arange(1, n_max+1, 1), 1]), np.append(cross_correlation[0, 0], cross_correlation[np.arange(1, n_max+1, 1), 0]))
    cross_correlation = cross_correlation.astype(np.int64)

    return cross_correlation
