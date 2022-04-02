import math
import numpy as np

def convert_spike_train_to_spikes_idxs(spike_train):
    spikes_idxs = np.argwhere(spike_train != 0)
    spikes_idxs = np.array([spikes_idxs[i][0] for i in range(len(spikes_idxs))])

    return spikes_idxs

def convert_spikes_idxs_to_spike_train(spikes_idxs, sampling_time, duration=None):
    if duration is None:
        duration = spikes_idxs[-1] * sampling_time * 1.02

    spike_train = np.zeros(math.floor(duration/sampling_time))
    spike_train[spikes_idxs] = 1

    return spike_train

def get_in_samples(value, sampling_time):
    value = math.floor(value/sampling_time)

    return value