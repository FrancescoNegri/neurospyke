import numpy as np
from ... import utils
from scipy.signal import find_peaks

def hard_threshold_local_maxima_samples(data:np.ndarray, threshold:float, refractory_period:int, use_abs:bool=False):
    '''
    Use the Hard Threshold Local Maxima algorithm to detect spikes,
    with parameters specified as samples.

    Parameters
    ----------
    data : numpy.ndarray
        The array of recorded data.
    threshold : float
        A threshold employed by the algorithm to detect a spike.
    refractory_period : int
        The detection algorithm refractory period, expressed in samples.
    use_abs : bool
        A flag indicating if the absolute value of the data is to be used.
    
    Returns
    -------
    spikes_idxs : numpy.ndarray
        An array containing all the indices of detected spikes.
    spikes_values numpy.ndarray
        An array containing all the values (i.e. amplitude) of detected spikes.
    '''
    # Cast data type to float
    data = data.astype(np.float64)

    if use_abs is True:
        data = abs(data)
    
    spikes_idxs, _ = find_peaks(data, height=threshold, distance=refractory_period)
    spikes_values = data[spikes_idxs]

    spikes_idxs = np.array(spikes_idxs, dtype=np.int64)
    spikes_values = np.array(spikes_values, dtype=np.float64)
    return spikes_idxs, spikes_values

def hard_threshold_local_maxima(data:np.ndarray, sampling_time:float, threshold:float, refractory_period:float, use_abs:bool=False):
    '''
    Use the Hard Threshold Local Maxima algorithm to detect spikes,
    with parameters specified in the time domain.

    Parameters
    ----------
    data : numpy.ndarray
        The array of recorded data.
    sampling_time : float
        The sampling time for the recorded data.
    threshold : float
        A threshold employed by the algorithm to detect a spike.
    refractory_period : float
        The detection algorithm refractory period, expressed in seconds.
    use_abs : bool
        A flag indicating if the absolute value of the data is to be used.
    
    Returns
    -------
    spikes_idxs : numpy.ndarray
        An array containing all the indices of detected spikes.
    spikes_values numpy.ndarray
        An array containing all the values (i.e. amplitude) of detected spikes.
    '''
    # Convert all parameters from time-domain to samples
    refractory_period = utils.get_in_samples(refractory_period, sampling_time)
    
    spikes_idxs, spikes_values = hard_threshold_local_maxima_samples(data, threshold, refractory_period, use_abs)

    return spikes_idxs, spikes_values