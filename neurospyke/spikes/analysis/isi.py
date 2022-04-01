import numpy as np

def get_ISI(spikes_idxs, sampling_time):
    ISI = np.array([(spikes_idxs[i] - spikes_idxs[i-1]) for i in range(1, len(spikes_idxs), 1)])
    ISI = ISI * sampling_time

    return ISI