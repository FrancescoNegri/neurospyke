import numpy as np
from . import utils

def get_IEI(spikes:np.ndarray, sampling_time:float = None):
    '''
    Get the Inter-Event-Interval of any type of event, such
    as the Inter-Spike-Interval, either in samples or seconds.

    Parameters
    ----------
    spikes : ndarray
        An array containing the detected spikes. It can be expressed both
        as a spike train or the indices at which spikes occur.
    sampling_time : float, optional
        The sampling time for the recorded data. If specified, the algorithm
        will work in the time domain. Otherwise, it will work with samples.

    Returns
    -------
    IEI : ndarray
        The Inter-Event-Interval expressed in samples or seconds.
    '''
    spikes.squeeze()

    if spikes.dtype == 'bool':
        spikes_idxs = utils.convert_train_to_idxs(spikes)
    else:
        spikes_idxs = spikes
    
    IEI = np.diff(spikes_idxs)

    if sampling_time is not None:
        IEI = IEI * sampling_time

    return IEI