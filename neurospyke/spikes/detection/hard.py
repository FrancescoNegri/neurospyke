import numpy as np
from scipy.signal import find_peaks
from ... import utils

def _parse_kwargs(**kwargs):
    kwargs_list = [
        {'key': 'sampling_time', 'default': None, 'type': float},
        {'key': 'use_abs', 'default': False, 'type': bool}
    ]
    kwargs = utils.check_kwargs_list(kwargs_list, **kwargs)

    return kwargs

def hard_threshold_local_maxima(data:np.ndarray, threshold:float, refractory_period:float, **kwargs):
    '''
    Use the Hard Threshold Local Maxima algorithm to detect spikes,
    with parameters specified either in the time domain or in samples.

    Parameters
    ----------
    data : numpy.ndarray
        The array of recorded data.
    threshold : float
        A threshold employed by the algorithm to detect a spike.
    refractory_period : float
        The detection algorithm refractory period, expressed in seconds or samples.
    sampling_time : float, optional
        The sampling time for the recorded data. If specified, the algorithm
        will work in the time domain (the other parameters should then be
        specified in seconds). Otherwise, it will work with samples.
    use_abs : bool, defualt=False
        A flag indicating if the absolute value of the data is to be used.
    
    Returns
    -------
    spikes_idxs : numpy.ndarray
        An array containing all the indices of detected spikes.
    spikes_values numpy.ndarray
        An array containing all the values (i.e. amplitude) of detected spikes.

    References
    ----------
    [1] S. Gibson et al. “Technology-Aware Algorithm Design for Neural Spike Detection, Feature Extraction, and Dimensionality Reduction.” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 18, no. 5, 2010, pp. 469-478., https://doi.org/10.1109/tnsre.2010.2051683
    '''
    kwargs = _parse_kwargs(**kwargs)
    
    # Convert all parameters from time-domain to samples (if sampling_time not None) and force to int
    refractory_period = utils.get_in_samples(refractory_period, kwargs.get('sampling_time'))

    # Cast data type to float
    data = data.astype(np.float64)

    if kwargs.get('use_abs') is True:
        data = abs(data)
    
    spikes_idxs, _ = find_peaks(data, height=threshold, distance=refractory_period)
    spikes_values = data[spikes_idxs]

    spikes_idxs = np.array(spikes_idxs, dtype=np.int64)
    spikes_values = np.array(spikes_values, dtype=np.float64)
    
    return spikes_idxs, spikes_values