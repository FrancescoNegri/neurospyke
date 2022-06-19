import math
import numpy as np

def check_kwargs_list(kwargs_list, **kwargs):
    for kwarg in kwargs_list:
        kwargs[kwarg['key']] = kwargs.get(kwarg['key'], kwarg['default'])

        if kwarg['type'] is not None:
            if (kwargs.get(kwarg['key']) is None) and (kwarg['default'] is None):
                pass
            elif kwargs.get(kwarg['key']) is None:
                pass
            else:
                try:
                    if ((kwarg['type'] == list) or (kwarg['type'] == tuple)) and (type(kwargs.get(kwarg['key'])) is str):
                        kwargs[kwarg['key']] = [kwargs.get(kwarg['key'])]

                    kwargs[kwarg['key']] = kwarg['type'](kwargs.get(kwarg['key']))
                except:
                    raise TypeError("'" + kwarg['key'] + "' expected to be '" + kwarg['type'].__name__ + "', received '" + str(type(kwargs.get(kwarg['key'])).__name__) + "'")

    return kwargs

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

def get_in_samples(value, sampling_time=None):
    if sampling_time is not None:
        value = math.floor(value/sampling_time)
    else:
        value = math.floor(value)

    return value