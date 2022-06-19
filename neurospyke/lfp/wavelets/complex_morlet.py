import numpy as np

def complex_morlet(sampling_frequency, n_cycles, central_frequency):

    sampling_time = 1/sampling_frequency

    sigma_f = central_frequency / n_cycles
    sigma_t = 1 / (2*np.pi*sigma_f)

    duration = 2*sigma_t
    time_window = np.arange(-2*duration, 2*duration, sampling_time)
    
    normalization_constant = np.power(sigma_t * np.sqrt(np.pi), -1/2)

    wavelet = np.multiply(
        np.multiply(
            normalization_constant, 
            np.exp(np.divide(-np.power(time_window, 2), (2 * np.power(sigma_t, 2))))
        ), 
        np.exp(np.multiply(2 * 1j * np.pi * central_frequency, time_window))
    )

    return wavelet