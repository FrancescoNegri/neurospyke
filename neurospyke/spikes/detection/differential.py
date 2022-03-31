import math
import numpy as np
from ... import utils

def differential_threshold_samples(data:np.ndarray, threshold:float, window_length:int):
    '''
    Use the Differential Threshold algorithm to detect spikes,
    with parameters specified as samples.

    Parameters
    ----------
    data : numpy.ndarray
        The array of recorded data.
    threshold : float
        A threshold employed by the algorithm to detect a spike.
    window_length : int
        The length for the detection window to employ while looking
        for spikes to detect, expressed in samples.
    
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
    # Cast data type to float
    data = data.astype(np.float64)

    n_windows = math.floor(len(data) / window_length)

    spikes_idxs = []
    spikes_values = []

    for i in range(n_windows):
        window_idx = window_length * i
        window_data = data[np.arange(window_idx, window_idx+window_length, 1)]
        max_value = np.amax(window_data)
        min_value = np.amin(window_data)

        if abs(max_value - min_value) >= threshold:
            spikes_idxs.append(np.argmax(window_data)+window_idx)
            spikes_values.append(max_value)

    spikes_idxs = np.array(spikes_idxs, dtype=np.int64)
    spikes_values = np.array(spikes_values, dtype=np.float64)
    return spikes_idxs, spikes_values

def differential_threshold(data:np.ndarray, sampling_time:float, threshold:float, window_length:float):
    '''
    Use the Differential Threshold algorithm to detect spikes,
    with parameters specified in the time domain.

    Parameters
    ----------
    data : numpy.ndarray
        The array of recorded data.
    sampling_time : float
        The sampling time for the recorded data.
    threshold : float
        A threshold employed by the algorithm to detect a spike.
    window_length : float
        The length for the detection window to employ while looking
        for spikes to detect, expressed in seconds.
    
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
    # Convert all parameters from time-domain to samples
    window_length = utils.get_in_samples(window_length, sampling_time)
    
    spikes_idxs, spikes_values = differential_threshold_samples(data, threshold, window_length)

    return spikes_idxs, spikes_values