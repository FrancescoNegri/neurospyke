import numpy as np
from ... import utils

def _parse_kwargs(**kwargs):
    kwargs_list = [
        {'key': 'sampling_time', 'default': None, 'type': float},
        {'key': 'window_length', 'default': 0.001 if kwargs.get('sampling_time', None) is not None else 20, 'type': float},
    ]
    kwargs = utils.check_kwargs_list(kwargs_list, **kwargs)

    return kwargs

def get_waveforms(data, spikes_idxs, **kwargs):
    kwargs = _parse_kwargs(**kwargs)

    data.squeeze()
    spikes_idxs.squeeze()

    window_half_length = utils.get_in_samples(kwargs.get('window_length') / 2, kwargs.get('sampling_time'))
    
    windows_samples =  np.arange(-window_half_length, window_half_length)
    windows_samples = np.tile(windows_samples, (np.size(spikes_idxs), 1))

    spikes_idxs = np.tile(spikes_idxs, (np.size(windows_samples, axis=1), 1)).T
    
    windows_samples = windows_samples + spikes_idxs

    windows_samples = np.ravel(windows_samples)

    waveforms = data[windows_samples]
    waveforms = np.reshape(waveforms, np.shape(spikes_idxs))

    return waveforms