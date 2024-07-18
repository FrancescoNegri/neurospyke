import numpy as np
from scipy.interpolate import interp1d

def leader_follower(spikes):
    n_trains = len(spikes)
    pairs = [(n,m) for n in np.arange(n_trains) for m in np.arange(n_trains)]

    D = np.zeros((n_trains, n_trains))

    for n, m in pairs:
        D[n, m] = _compute_D_n_m()

def _compute_D_n_m(reference_spike_train, target_spike_train):
    if reference_spike_train.size > 1 and target_spike_train.size > 1:
        interp_func = interp1d(target_spike_train, target_spike_train, kind='nearest', fill_value='extrapolate', assume_sorted=True)
        nearest = interp_func(reference_spike_train).astype(np.int64)
        mapping = np.searchsorted(target_spike_train, nearest, sorter=np.argsort(target_spike_train))

        M = np.zeros((reference_spike_train.size, 4))

        t1 = np.diff(reference_spike_train)
        t2 = np.diff(target_spike_train)

        M[:, 0] = np.concatenate([[np.nan], t1])
        M[:, 1] = np.concatenate([t1, [np.nan]])
        M[:, 2] = np.concatenate([[np.nan], t2])[mapping]
        M[:, 3] = np.concatenate([t2, [np.nan]])[mapping]

        tau = np.nanmin(M, axis=1) / 2
        C = np.abs(nearest - reference_spike_train) < tau
        D = np.sum(np.sign(nearest - reference_spike_train) * C) / reference_spike_train.size
    else:
        D = np.nan

    return D
