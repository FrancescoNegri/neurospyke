import math
import numpy as np
from ... import utils

def _parse_kwargs(**kwargs):
    kwargs_list = [
        {'key': 'polarity', 'default': -1, 'type': int},
        {'key': 'sampling_time', 'default': None, 'type': float}
    ]
    kwargs = utils.check_kwargs_list(kwargs_list, **kwargs)

    return kwargs

def differential_threshold(data:np.ndarray, threshold:float, window_length:float, refractory_period:float, **kwargs):
    '''
    Use the Spike Detection Differential Threshold (SDDT) algorithm
    to detect spikes, with parameters specified either in the time
    domain or in samples.

    Parameters
    ----------
    data : ndarray
        The array of recorded data.
    threshold : float
        A threshold employed by the algorithm to detect a spike.
    window_length : float
        The length for the detection window to employ while looking
        for spikes to detect, expressed in seconds or samples.
    refractory_period : float
        The detection algorithm refractory period, expressed in seconds or samples.
    polarity : {-1, 1}, defualt=-1
        The polarity of spikes to look for. -1 means negative polarity,
        1 mean positive ones.
    sampling_time : float, optional
        The sampling time for the recorded data. If specified, the algorithm
        will work in the time domain (the other parameters should then be
        specified in seconds). Otherwise, it will work with samples.
    
    Returns
    -------
    spikes_idxs : ndarray
        An array containing all the indices of detected spikes.
    spikes_values ndarray
        An array containing all the values (i.e. amplitude) of detected spikes.
    
    References
    ----------
    [1] Maccione, A., Gandolfo, M., Massobrio, P., Novellino, A., Martinoia, S., & Chiappalone, M. (2009). A novel algorithm for precise identification of spikes in extracellularly recorded neuronal signals. Journal of Neuroscience Methods, 177(1), 241–249. https://doi.org/10.1016/j.jneumeth.2008.09.026
    '''
    kwargs = _parse_kwargs(**kwargs)
    
    # Convert all parameters from time-domain to samples (if sampling_time not None) and force to int
    window_length = utils.get_in_samples(window_length, kwargs.get('sampling_time'))
    refractory_period = utils.get_in_samples(refractory_period, kwargs.get('sampling_time'))

    # Cast data type to float
    data = data.astype(np.float64).squeeze()

    if kwargs.get('polarity') == -1:
        data = -data

    n_windows = math.floor(len(data) / window_length)

    spikes_idxs = []

    for i in range(n_windows):
        window_idx = window_length * i
        window_data = data[np.arange(window_idx, window_idx + window_length, 1)]
        max_value = np.amax(window_data)
        min_value = np.amin(window_data)

        if abs(max_value - min_value) >= threshold:
            if (len(spikes_idxs) != 0 and (np.argmax(window_data) + window_idx - spikes_idxs[-1]) > refractory_period):
                spikes_idxs.append(np.argmax(window_data) + window_idx)
            elif len(spikes_idxs) == 0:
                spikes_idxs.append(np.argmax(window_data) + window_idx)

    if kwargs.get('polarity') == -1:
        data = -data

    spikes_idxs = np.array(spikes_idxs, dtype=np.int64)
    spikes_values = data[spikes_idxs].astype(np.float64)
    
    return spikes_idxs, spikes_values