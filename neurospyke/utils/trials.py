import numpy as np

from scipy.interpolate import interp1d
from . import utils

def get_trials(spikes:np.ndarray, events:np.ndarray, duration:float = None, sampling_time:float = None, shuffle:bool = False, n : int = None, random_state : int = None):
    '''
    Get a list of ndarray containing all the trials for each of the specified events.

    Parameters
    ----------
    spikes : ndarray
        An array containing the detected spikes. It can be expressed both
        as a spike train or the indices at which spikes occur.
    events : ndarray
        An array containing the events determining the start of a new trial.
        It can be expressed both as a event train or the indices at which events occur.
    duration : float, optional
        The duration of a trial, either in samples or in seconds. If not specified,
        trials will last until the following event.
    n : int, optional
        The number of trials to return. They will be ordered according
        to events, unless shuffle is set to True. If not specified, all trials
        will be returned.
    shuffle : bool, default=False
        If False, trials are returned according to the order specified by events.
        If True, trials are shuffled and returned in a random order.
    random_state : int, optional
        Random seed used to initialize the pseudo-random number generator to allow
        reproducibility.
    sampling_time : float, optional
        The sampling time for the recorded data. If specified, the algorithm
        will work in the time domain (the other parameters should then be
        specified in seconds). Otherwise, it will work with samples.

    Returns
    -------
    trials : list
        A list of ndarrays containing all the trials following the order of the
        input events.
    '''
    rng = np.random.RandomState(random_state)

    spikes.squeeze()
    if spikes.dtype == 'bool':
        spikes_idxs = utils.convert_train_to_idxs(spikes)
    else:
        spikes_idxs = spikes

    events.squeeze()
    if events.dtype == 'bool':
        events_idxs = utils.convert_train_to_idxs(events)
    else:
        events_idxs = events

    spikes_idxs = spikes_idxs[(spikes_idxs >= events_idxs[0])]

    if duration is not None:
        duration = utils.get_in_samples(duration, sampling_time)
        spikes_idxs = spikes_idxs[(spikes_idxs < events_idxs[-1] + duration)]

    interpolation_function = interp1d(events_idxs, events_idxs, kind='previous', fill_value='extrapolate')
    trials_events_idxs = interpolation_function(spikes_idxs)

    if n is not None:
        n = np.min([n, events_idxs.size])
    else:
        n = events_idxs.size

    if shuffle is True:
        events_idxs = events_idxs[rng.choice(events_idxs.size, n, replace=False)]
    else:
        events_idxs = events_idxs[np.arange(n)]

    trials = []

    for events_idx in events_idxs:
        idxs = np.argwhere(trials_events_idxs == events_idx)[:, 0]
        trial = (spikes_idxs[idxs] - events_idx).astype(np.int64)
        if duration is not None:
            trial = trial[trial < duration]
        trials.append(trial)

    return trials