import numpy as np
from scipy.signal import argrelmax, argrelmin
from ... import utils

def _parse_kwargs(**kwargs):
    kwargs_list = [
        {'key': 'sampling_time', 'default': None, 'type': float}
    ]
    kwargs = utils.check_kwargs_list(kwargs_list, **kwargs)

    return kwargs

def PTSD(data:np.ndarray, threshold:float, refractory_period:float, peak_lifetime_period:float, overshoot:float, **kwargs):
    '''
    Use the Precision Timing Spike Detection (PTSD) algorithm to detect spikes,
    with parameters specified either in the time domain or in samples.

    Parameters
    ----------
    data : numpy.ndarray
        The array of recorded data.
    threshold : float
        A threshold employed by the algorithm to detect a spike.
    refractory_period : float
        The detection algorithm refractory period, expressed in seconds or samples.
    peak_lifetime_period : float
        The maximum duration of a spike, between its positive and
        negative peaks. Expressed in seconds or samples.
    overshoot : float
        An extra time interval extending the peak lifetime period in case
        no spike is found inside it, expressed in seconds or samples.
    sampling_time : float, optional
        The sampling time for the recorded data. If specified, the algorithm
        will work in the time domain (the other parameters should then be
        specified in seconds). Otherwise, it will work with samples.
    
    Returns
    -------
    spikes_idxs : numpy.ndarray
        An array containing all the indices of detected spikes.
    spikes_values numpy.ndarray
        An array containing all the values (i.e. amplitude) of detected spikes.
    
    References
    ----------
    [1] A. Maccione et al. “A novel algorithm for precise identification of spikes in extracellularly recorded neuronal signals.” Journal of neuroscience methods vol. 177,1 (2009): 241-9. https://doi.org/10.1016/j.jneumeth.2008.09.026
    '''
    kwargs = _parse_kwargs(**kwargs)
    
    # Convert all parameters from time-domain to samples (if sampling_time not None) and force to int
    refractory_period = utils.get_in_samples(refractory_period, kwargs.get('sampling_time'))
    peak_lifetime_period = utils.get_in_samples(peak_lifetime_period, kwargs.get('sampling_time'))
    overshoot = utils.get_in_samples(overshoot, kwargs.get('sampling_time'))

    # Cast data type to float
    data = data.astype(np.float64)

    spikes_idxs = []
    spikes_values = []

    max_idxs = argrelmax(data)[0]
    max_values = data[max_idxs]

    for i in range(len(max_idxs)):
        if max_idxs[i] + peak_lifetime_period <= len(data):
            window_data = data[np.arange(max_idxs[i], max_idxs[i] + peak_lifetime_period)]
        else:
            window_data = data[np.arange(max_idxs[i], len(data))]
        
        min_idx = argrelmin(window_data)[0]
        min_value = data[min_idx]

        if len(min_idx) > 1:
            min_idx = min_idx[0]
            min_value = min_value[0]
        elif len(min_idx) == 1:
            pass
        else:
            if max_idxs[i] + peak_lifetime_period + overshoot <= len(data):
                window_data = data[np.arange(max_idxs[i], max_idxs[i] + peak_lifetime_period + overshoot)]
            else:
                window_data = data[np.arange(max_idxs[i], len(data))]
            
            min_idx = argrelmin(window_data)[0]
            min_value = data[min_idx]

            if len(min_idx) > 1:
                min_idx = min_idx[0]
                min_value = min_value[0]
            elif len(min_idx) == 1:
                pass
            else:
                max_values[i] = None

        if max_values[i] is not None:
            if abs(max_values[i] - min_value) >= threshold:
                if len(spikes_idxs) == 0:
                    spikes_idxs.append(max_idxs[i])
                    spikes_values.append(max_values[i])
                elif abs(spikes_idxs[-1] - i) > refractory_period:
                    spikes_idxs.append(max_idxs[i])
                    spikes_values.append(max_values[i])

    spikes_idxs = np.array(spikes_idxs, dtype=np.int64)
    spikes_values = np.array(spikes_values, dtype=np.float64)
    
    return spikes_idxs, spikes_values