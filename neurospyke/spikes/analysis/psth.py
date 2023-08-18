import numpy as np

from ... import utils

def _parse_kwargs(**kwargs):
    kwargs_list = [
        {'key': 'bin_size', 'default': 1e-3 if kwargs.get('sampling_time', None) is not None else 20, 'type': float},
        {'key': 'stimuli', 'default': None, 'type': None},
        {'key': 'sampling_time', 'default': None, 'type': float}
    ]
    kwargs = utils.check_kwargs_list(kwargs_list, **kwargs)

    return kwargs

def PSTH(data, duration:float, **kwargs):
    '''
    Count the number of spikes in bins over different trials according to the
    parameters specified, allowing to obtain Post-Stimulus Time Histogram.

    Parameters
    ----------
    data : ndarray or list of ndarray
        A list of ndarrays containing all the trials. Otherwise, an array containing
        the detected spikes. It can be expressed both as a spike train or the indices
        at which spikes occur. In such a case, also the stimuli parameter must be
        specified.
    duration : float
        The duration of a trial, either in samples or in seconds.
    bin_size : float
        The size of a single bin, either in samples or in seconds.
    stimuli : ndarray, optional
        An array containing the stimuli determining the start of a new trial.
        It can be expressed both as a train of stimuli or the indices at which
        stimuli occur.
    sampling_time : float, optional
        The sampling time for the recorded data. If specified, the algorithm
        will work in the time domain (the other parameters should then be
        specified in seconds). Otherwise, it will work with samples.

    Returns
    -------
    spikes_count : ndarray
        A (n_trials x n_bins) matrix containing the number of spikes in each
        bin and for each trial.
    '''
    kwargs = _parse_kwargs(**kwargs)

    data.squeeze()

    duration = utils.get_in_samples(duration, kwargs.get('sampling_time'))
    kwargs['bin_size'] = utils.get_in_samples(kwargs.get('bin_size'), kwargs.get('sampling_time'))

    if kwargs.get('stimuli') is not None:
        data = utils.get_trials(data, kwargs.get('stimuli'), duration)

    trials = data
    trials = np.array([utils.convert_idxs_to_train(trial, duration) for trial in trials])
    n_bins = np.floor(duration / kwargs.get('bin_size')).astype(np.int64)

    spikes_count = np.zeros([trials.shape[0], n_bins], dtype=np.int64)

    for bin_idx in range(n_bins): 
        idxs = np.arange(bin_idx*kwargs.get('bin_size'), bin_idx*kwargs.get('bin_size') + kwargs.get('bin_size'), dtype=np.int64)
        spikes_count[:, bin_idx] = np.sum(trials[:, idxs])

    return spikes_count
