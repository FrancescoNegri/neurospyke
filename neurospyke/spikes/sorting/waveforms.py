import numpy as np

from ... import utils

def _parse_kwargs(**kwargs):
    kwargs_list = [
        {'key': 'sampling_time', 'default': None, 'type': float},
        {'key': 'window_length', 'default': 0.001 if kwargs.get('sampling_time', None) is not None else 20, 'type': float},
    ]
    kwargs = utils.check_kwargs_list(kwargs_list, **kwargs)

    return kwargs

def get_waveforms(data:np.ndarray, events:np.ndarray, **kwargs):
    '''
    Get the data surrounding certain set of events, according to a
    specified window. This is useful for instance to obtain the waveforms
    for all the spikes detected in a recording.

    Parameters
    ----------
    data : ndarray
        The array of recorded data.
    events : ndarray
        An array containing the events of interest. It can be express both
        as a event train or as a list of the indices at which events occur.
    window_length : float, optional
        The length for the detection window to employ while isolating
        events, expressed in seconds or samples.
    sampling_time : float, optional
        The sampling time for the recorded data. If specified, the algorithm
        will work in the time domain (the other parameters should then be
        specified in seconds). Otherwise, it will work with samples.

    Returns
    -------
    waveforms : ndarray
        A (2 x n_events) matrix where each row represents a different waveform surrounding
        the specified events, according to the specified window.
    '''
    kwargs = _parse_kwargs(**kwargs)

    data.squeeze()
    events.squeeze()

    if events.dtype == 'bool':
        events_idxs = utils.convert_train_to_idxs(events)
    else:
        events_idxs = events

    window_half_length = utils.get_in_samples(kwargs.get('window_length') / 2, kwargs.get('sampling_time'))
    
    windows_samples =  np.arange(-window_half_length, window_half_length)
    windows_samples = np.tile(windows_samples, (np.size(events_idxs), 1))

    events_idxs = np.tile(events_idxs, (np.size(windows_samples, axis=1), 1)).T
    
    windows_samples = windows_samples + events_idxs

    windows_samples = np.ravel(windows_samples)

    waveforms = data[windows_samples]
    waveforms = np.reshape(waveforms, np.shape(events_idxs))

    return waveforms