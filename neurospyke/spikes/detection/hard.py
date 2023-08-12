import numpy as np
from scipy.signal import find_peaks
from ... import utils

def _parse_kwargs(**kwargs):
    kwargs_list = [
        {'key': 'sampling_time', 'default': None, 'type': float},
        {'key': 'polarity', 'default': -1, 'type': int}
    ]
    kwargs = utils.check_kwargs_list(kwargs_list, **kwargs)

    return kwargs

def hard_threshold(data:np.ndarray, threshold:float, refractory_period:float, **kwargs):
    '''
    Use the Hard Threshold Local Maxima algorithm to detect spikes,
    with parameters specified either in the time domain or in samples.

    Parameters
    ----------
    data : ndarray
        The array of recorded data.
    threshold : float
        A threshold employed by the algorithm to detect a spike.
    refractory_period : float
        The detection algorithm refractory period, expressed in seconds or samples.
    sampling_time : float, optional
        The sampling time for the recorded data. If specified, the algorithm
        will work in the time domain (the other parameters should then be
        specified in seconds). Otherwise, it will work with samples.
    polarity : {-1, 0, 1}, defualt=-1
        The polarity of spikes to look for. -1 means negative polarity,
        1 mean positive ones, while 0 applies the absolute value to the
        signal.
    
    Returns
    -------
    spikes_idxs : ndarray
        An array containing all the indices of detected spikes.
    spikes_values ndarray
        An array containing all the values (i.e. amplitude) of detected spikes.

    References
    ----------
    [1] Gibson, S., Judy, J. W., & Markovic, D. (2010). Technology-Aware Algorithm Design for Neural Spike Detection, Feature Extraction, and Dimensionality Reduction. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 18(5), 469–478. https://doi.org/10.1109/tnsre.2010.2051683
    '''
    kwargs = _parse_kwargs(**kwargs)
    
    # Convert all parameters from time-domain to samples (if sampling_time not None) and force to int
    refractory_period = utils.get_in_samples(refractory_period, kwargs.get('sampling_time'))

    # Cast data type to float
    data = data.astype(np.float64).squeeze()
    
    if kwargs.get('polarity') == -1:
        spikes_idxs, _ = find_peaks(-data, height=-threshold, distance=refractory_period)
    elif kwargs.get('polarity') == 0:
        spikes_idxs, _ = find_peaks(abs(data), height=threshold, distance=refractory_period)
    else:
        spikes_idxs, _ = find_peaks(data, height=threshold, distance=refractory_period)

    spikes_idxs = spikes_idxs.astype(np.int64)
    spikes_values = data[spikes_idxs].astype(np.float64)
    
    return spikes_idxs, spikes_values