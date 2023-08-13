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

def convert_train_to_idxs(train:np.ndarray):
    train.squeeze()
    idxs = np.argwhere(train).astype(np.int64)

    return idxs

def convert_idxs_to_train(idxs:np.ndarray, duration:float = None, sampling_time:float = None):
    if duration is None:
        train = np.zeros(idxs[-1] + 1, dtype=bool)
    elif sampling_time is None:
        train = np.zeros(duration, dtype=bool)
    else:
        train = np.zeros(math.floor(duration/sampling_time), dtype=bool)
    
    train[idxs] = True

    return train

def get_in_samples(value, sampling_time:float = None):
    if sampling_time is not None:
        value = math.floor(value/sampling_time)
    else:
        value = math.floor(value)

    return value