import numpy as np
import pywt
from scipy.signal import convolve2d, get_window, find_peaks
from ... import utils

def _parse_kwargs(**kwargs):
    kwargs_list = [
        {'key': 'sampling_time', 'default': None, 'type': float},
        {'key': 'wavelet_level', 'default': 2, 'type': int},
        {'key': 'wavelet_name', 'default': 'sym6', 'type': str},
        {'key': 'window', 'default': 'hamming', 'type': None},
        {'key': 'window_samples', 'default': 25, 'type': int},
        {'key': 'window_symmetric', 'default': True, 'type': bool},
        {'key': 'threshold', 'default': 6, 'type': float},
        {'key': 'polarity', 'default': -1, 'type': int},
    ]
    kwargs = utils.check_kwargs_list(kwargs_list, **kwargs)

    return kwargs

def OSWTTEO(data:np.ndarray, refractory_period:float, peak_duration:float, **kwargs):
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
    peak_duration = utils.get_in_samples(peak_duration, kwargs.get('sampling_time'))

    # Cast data type to float
    data = data.astype(np.float64)

    spikes_idxs = []
    spikes_values = []

    L = np.size(data)
    data = data.reshape(-1, 1) 

    pow = np.power(2, kwargs.get('wavelet_level'))
    if L % pow > 0:
        # TODO: check this if
        Lok = np.ceil(L / pow) * pow
        Ldiff = int(Lok - L)
        data = np.concatenate((data, np.zeros((Ldiff, 1))))
    
    wavelet = pywt.Wavelet(kwargs.get('wavelet_name'))
    lo_D = np.array(wavelet.dec_lo)
    hi_D = np.array(wavelet.dec_hi)

    out_ = np.zeros(data.shape)
    ss = data.copy()  # Make a copy of data

    for k in range(1, kwargs.get('wavelet_level') + 1):
        # Extension
        lf = np.size(lo_D)
        ss = extendswt(ss, lf)
        
        # Convolution
        swa = convolve2d(ss, np.array([lo_D]).T, mode='valid')
        swa = swa[1:, :]  # Even number of filter coefficients
        
        # Apply TEO to SWT output
        temp = np.abs(TEO(swa, 1))
        
        if kwargs.get('window_samples'):
            wind = get_window(kwargs.get('window'), kwargs.get('window_samples'), fftbins=(not kwargs.get('window_symmetric')))
            temp2 = convolve2d(temp, np.array([wind]), mode='same')
            # Note that wind = (25,), np.array([wind]) = (1, 25)
        else:
            temp2 = temp
        
        out_ += temp2
        
        # Dyadic upscaling of filter coefficients
        lo_D = np.repeat(lo_D, 2)
        lo_D[np.arange(1, lo_D.size, 2)] = 0
        
        # Update ss
        ss = swa

    # Standard detection
    lambda_swtteo = np.percentile(out_, 99)
    lambda_data = kwargs.get('threshold') * np.median(np.abs(data))
    data_th = np.zeros(data.shape)
    data_th[out_ > lambda_swtteo] = kwargs.get('polarity') * data[out_ > lambda_swtteo]

    ts, pmin = peakseek(data_th, refractory_period, lambda_data)
    pmin = pmin * kwargs.get('polarity')
    E = out_[ts]

    # GET PEAK-TO-PEAK VALUES
    tloc = np.tile(ts, (2 * peak_duration + 1, 1)) + np.arange(-peak_duration, peak_duration + 1)[:, np.newaxis]
    tloc[tloc < 0] = 0
    tloc[tloc >= len(data)] = len(data) - 1

    pmax = np.max(data[tloc], axis=0)
    Imax = np.argmax(data[tloc], axis=0)
    p2pamp = pmax + pmin

    # Get peak width and exclude pw > peak_duration
    tlocmin = np.flipud(np.tile(ts, (peak_duration + 1, 1)) + np.arange(-peak_duration, 1)[:, np.newaxis])
    tlocmin[tlocmin < 0] = 0
    tlocmin[tlocmin >= len(data)] = len(data) - 1

    tlocmax = np.tile(ts, (peak_duration, 1)) + np.arange(1, peak_duration + 1)[:, np.newaxis]
    tlocmax[tlocmax < 0] = 0
    tlocmax[tlocmax >= len(data)] = len(data) - 1

    Imax1 = np.zeros(tlocmin.shape[1], dtype=int)
    Imax2 = np.zeros(tlocmin.shape[1], dtype=int)

    for ii in range(tlocmin.shape[1]):
        peak_indices, _ = find_peaks(data[tlocmin[:, ii], 0])
        if len(peak_indices) == 0:
            thispeak = peak_duration
        else:
            thispeak = peak_indices[0]
        Imax1[ii] = -thispeak

    for ii in range(tlocmax.shape[1]):
        peak_indices, _ = find_peaks(data[tlocmax[:, ii], 0])
        if len(peak_indices) == 0:
            thispeak = peak_duration
        else:
            thispeak = peak_indices[0]
        Imax2[ii] = thispeak + 1

    pW = Imax2 - Imax1

    # Exclude values
    pw_ex = pW > peak_duration
    pm_ex = pmax[:, 0] <= 0

    ex = pw_ex | pm_ex
    keep = np.invert(ex)
    ts = ts[keep]
    p2pamp = p2pamp[keep, 0]
    pmax = pmax[keep, 0]
    pmin = pmin[keep, 0]
    E = E[keep, 0]
    pW = pW[keep]

    spikes_idxs = np.array(ts, dtype=np.int64)
    spikes_values = np.array(p2pamp, dtype=np.float64)
    
    return spikes_idxs, spikes_values

def extendswt(x, lf):
    r, c = x.shape
    y = np.zeros((r + lf, c))
    
    y[:lf // 2, :] = x[-lf // 2:, :]
    y[lf // 2:lf // 2 + r, :] = x
    y[-lf // 2:] = x[:lf // 2, :]
    
    return y

def TEO(x, k):
    return x**2 - myTEOcircshift(x, [-k, 0]) * myTEOcircshift(x, [k, 0])

def myTEOcircshift(Y, k):
    colshift = k[0]
    rowshift = k[1]
    
    temp = np.roll(Y, k, axis=(0, 1))

    if colshift < 0:
        temp[temp.shape[0] + colshift:, :] = np.flipud(Y[Y.shape[0] + colshift:, :])
    elif colshift > 0:
        temp[:colshift, :] = np.flipud(Y[:colshift, :])

    if rowshift < 0:
        # TODO: check this case
        temp[:, temp.shape[1] + rowshift + 1:] = np.fliplr(Y[:, Y.shape[1] + rowshift + 1:])
    elif rowshift > 0:
        # TODO: check this case
        temp[:, :rowshift] = np.fliplr(Y[:, :rowshift])

    X = temp
    return X

def peakseek(x, minpeakdist=1, minpeakh=None):
    if x.shape[0] == 1:
        x = x.T

    locs = np.where((x[1:-1] >= x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1

    if minpeakh is not None:
        locs = locs[x[locs, 0] > minpeakh]

    if minpeakdist > 1:
        while True:
            del_vals = np.diff(locs) < minpeakdist

            if not np.any(del_vals):
                break

            pks = x[locs, 0]

            mins = np.argmin(np.vstack((pks[np.hstack((del_vals, False))], pks[np.hstack((False, del_vals))])), axis=0)

            deln = np.where(del_vals)[0]

            deln = np.concatenate((deln[mins == 0], deln[mins == 1] + 1))

            locs = np.delete(locs, deln)

    pks = x[locs]
    
    return locs, pks