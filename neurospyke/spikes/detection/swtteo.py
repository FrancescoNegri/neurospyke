import numpy as np
import pywt
from scipy.signal import convolve, get_window, find_peaks
from ... import utils

def _parse_kwargs(**kwargs):
    kwargs_list = [
        {'key': 'peak_duration', 'default': 0.0025, 'type': float},
        {'key': 'polarity', 'default': -1, 'type': int},
        {'key': 'refractory_period', 'default': 0.001, 'type': float},
        {'key': 'threshold', 'default': 6, 'type': float},
        {'key': 'wavelet_level', 'default': 2, 'type': int},
        {'key': 'wavelet_name', 'default': 'sym6', 'type': str},
        {'key': 'window', 'default': 'hamming', 'type': None},
        {'key': 'window_samples', 'default': 25, 'type': int},
        {'key': 'window_symmetric', 'default': True, 'type': bool}
    ]
    kwargs = utils.check_kwargs_list(kwargs_list, **kwargs)

    return kwargs

def SWTTEO(data:np.ndarray, sampling_time:float, **kwargs):
    '''
    Use the stationary wavelet transform and Teager Energy Operator (SWTTEO) algorithm to detect spikes,
    with parameters specified in the time domain.

    Parameters
    ----------
    data : ndarray
        The array of recorded data.
    sampling_time : float
        The sampling time for the recorded data.
    peak_duration : float, default=0.0025
        The maximum duration of a spike, between its positive and
        negative peaks. Expressed in seconds.
    polarity : {-1, 1}, defualt=-1
        The polarity of spikes to look for. -1 means negative polarity,
        1 mean positive ones.
    refractory_period : float, default=0.001
        The detection algorithm refractory period, expressed in seconds.
    threshold : float, default=6
        A multiplication coefficient for the threshold employed by the algorithm to
        detect a spike.
    wavelet_level : int, default=2
        The wavelet decomposition level, specified as an integer.
    wavelet_name : str, default='sym6'
        The wavelet name. Refer to the PyWavelet documentation for an extensive list
        of the available built-in wavelets.
    window : str or float or tuple, default='hamming'
        The type of smoothing window to employ. Refer to the SciPy documentation for a
        detailed explanation of this parameter.
    window_samples : int, default=25
        The number of samples in the smoothing window.
    window_symmetric : bool, default=True
        If True, create a “symmetric” window, for use in filter design.
        If False, create a “periodic” window, ready to use with ifftshift
        and be multiplied by the result of an FFT. 
    
    Returns
    -------
    spikes_idxs : numpy.ndarray
        An array containing all the indices of detected spikes.
    spikes_values numpy.ndarray
        An array containing all the values (i.e. amplitude) of detected spikes.
    
    References
    ----------
    [1] Lieb, F., Stark, H. G., & Thielemann, C. (2017). A stationary wavelet transform and a time-frequency based spike detection algorithm for extracellular recorded data. Journal of Neural Engineering, 14(3), 036013. https://doi.org/10.1088/1741-2552/aa654b
    '''
    kwargs = _parse_kwargs(**kwargs)
    
    # Convert all parameters from time-domain to samples (if sampling_time not None) and force to int
    refractory_period = utils.get_in_samples(kwargs.get('refractory_period'), sampling_time)
    peak_duration = utils.get_in_samples(kwargs.get('peak_duration'), sampling_time)

    # Cast data type to float
    data = data.astype(np.float64).squeeze()
    L = np.size(data)

    pow = np.power(2, kwargs.get('wavelet_level'))
    if L % pow > 0:
        # TODO: check this if
        L_ok = np.ceil(L / pow) * pow
        L_diff = int(L_ok - L)
        data = np.concatenate((data, np.zeros((L_diff))))
    
    wavelet = pywt.Wavelet(kwargs.get('wavelet_name'))
    lo_D = np.array(wavelet.dec_lo)
    # hi_D = np.array(wavelet.dec_hi)

    out = np.zeros(data.shape)
    ss = data.copy()  # Make a copy of data

    for k in range(1, kwargs.get('wavelet_level') + 1):
        # Extension
        lf = np.size(lo_D)
        ss = extend_swt(ss, lf)
        
        # Convolution
        swa = convolve(ss, lo_D, mode='valid')
        swa = swa[1:]  # Even number of filter coefficients
        
        # Apply TEO to SWT output
        temp = np.abs(TEO(swa, 1))
        
        if kwargs.get('window_samples'):
            window = get_window(kwargs.get('window'), kwargs.get('window_samples'), fftbins=(not kwargs.get('window_symmetric')))
            temp = convolve(temp, window, mode='same')
        
        out += temp
        
        # Dyadic upscaling of filter coefficients
        lo_D = np.repeat(lo_D, 2)
        lo_D[np.arange(1, lo_D.size, 2)] = 0
        
        # Update ss
        ss = swa

    # Standard detection
    lambda_swtteo = np.percentile(out, 99)
    lambda_data = kwargs.get('threshold') * np.median(np.abs(data))
    data_th = np.zeros(data.shape)
    data_th[out > lambda_swtteo] = kwargs.get('polarity') * data[out > lambda_swtteo]

    ts, pmin = seek_peaks(data_th, refractory_period, lambda_data)
    pmin = pmin * kwargs.get('polarity')
    E = out[ts]

    # Get peak-to-peak values
    tloc = np.tile(ts, (2 * peak_duration + 1, 1)) + np.arange(-peak_duration, peak_duration + 1)[:, np.newaxis]
    tloc[tloc < 0] = 0
    tloc[tloc >= len(data)] = len(data) - 1

    pmax = np.max(data[tloc], axis=0)
    # Imax = np.argmax(data[tloc], axis=0)
    p2p_amplitude = pmax + pmin

    # Get peak width and exclude peak_width > peak_duration
    tlocmin = np.flipud(np.tile(ts, (peak_duration + 1, 1)) + np.arange(-peak_duration, 1)[:, np.newaxis])
    tlocmin[tlocmin < 0] = 0
    tlocmin[tlocmin >= len(data)] = len(data) - 1

    tlocmax = np.tile(ts, (peak_duration, 1)) + np.arange(1, peak_duration + 1)[:, np.newaxis]
    tlocmax[tlocmax < 0] = 0
    tlocmax[tlocmax >= len(data)] = len(data) - 1

    Imax1 = np.zeros(tlocmin.shape[1], dtype=int)
    Imax2 = np.zeros(tlocmin.shape[1], dtype=int)

    for ii in range(tlocmin.shape[1]):
        peak_indices, _ = find_peaks(data[tlocmin[:, ii]])
        if len(peak_indices) == 0:
            this_peak = peak_duration
        else:
            this_peak = peak_indices[0]
        Imax1[ii] = -this_peak

        peak_indices, _ = find_peaks(data[tlocmax[:, ii]])
        if len(peak_indices) == 0:
            this_peak = peak_duration
        else:
            this_peak = peak_indices[0]
        Imax2[ii] = this_peak + 1

    peak_width = Imax2 - Imax1

    # Exclude values
    keep_idxs = (peak_width <= peak_duration) & (pmax[:] > 0)

    ts = ts[keep_idxs]
    p2p_amplitude = p2p_amplitude[keep_idxs]
    pmax = pmax[keep_idxs]
    pmin = pmin[keep_idxs]
    E = E[keep_idxs]
    peak_width = peak_width[keep_idxs]

    spikes_idxs = np.array(ts, dtype=np.int64)
    spikes_values = np.array(data[spikes_idxs], dtype=np.float64)
    
    return spikes_idxs, spikes_values

def extend_swt(x, lf):
    r = np.size(x)
    y = np.zeros((r + lf))
    
    y[:lf // 2] = x[-lf // 2:]
    y[lf // 2:lf // 2 + r] = x
    y[-lf // 2:] = x[:lf // 2]
    
    return y

def TEO(x, k):
    return x**2 - roll_TEO(x, -k) * roll_TEO(x, k)

def roll_TEO(x, k):
    col_shift = k
    
    y = np.roll(x, k)

    if col_shift < 0:
        y[y.shape[0] + col_shift:] = np.flipud(x[x.shape[0] + col_shift:])
    elif col_shift > 0:
        y[:col_shift] = np.flipud(x[:col_shift])

    return y

def seek_peaks(x, min_peak_distance=1, min_peak_height=None):
    locs = np.where((x[1:-1] >= x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1

    if min_peak_height is not None:
        locs = locs[x[locs] > min_peak_height]

    if min_peak_distance > 1:
        while True:
            del_vals = np.diff(locs) < min_peak_distance

            if not np.any(del_vals):
                break

            pks = x[locs]

            mins = np.argmin(np.vstack((pks[np.hstack((del_vals, False))], pks[np.hstack((False, del_vals))])), axis=0)

            deln = np.where(del_vals)[0]

            deln = np.concatenate((deln[mins == 0], deln[mins == 1] + 1))

            locs = np.delete(locs, deln)

    pks = x[locs]
    
    return locs, pks