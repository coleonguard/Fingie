# filtering.py

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

def ema_filter(data, alpha=0.1):
    """
    Applies an Exponential Moving Average (EMA) filter to the data.

    Args:
        data (numpy.ndarray): The input data to be filtered.
        alpha (float, optional): Smoothing factor between 0 and 1. Default is 0.1.

    Returns:
        numpy.ndarray: The filtered data.
    """
    ema_data = np.zeros_like(data)
    ema_data[0] = data[0]
    for t in range(1, len(data)):
        ema_data[t] = alpha * data[t] + (1 - alpha) * ema_data[t - 1]
    return ema_data

def kalman_filter(data, process_variance=1e-5, measurement_variance=1e-1):
    """
    Applies a simple 1D Kalman Filter to the data.

    Args:
        data (numpy.ndarray): The input data to be filtered.
        process_variance (float, optional): Process variance (Q). Default is 1e-5.
        measurement_variance (float, optional): Measurement variance (R). Default is 1e-1.

    Returns:
        numpy.ndarray: The filtered data.
    """
    n_iter = len(data)
    sz = (n_iter,)  # size of array

    # Allocate space for arrays
    xhat = np.zeros(sz)      # a posteri estimate of x
    P = np.zeros(sz)         # a posteri error estimate
    xhatminus = np.zeros(sz) # a priori estimate of x
    Pminus = np.zeros(sz)    # a priori error estimate
    K = np.zeros(sz)         # gain or blending factor

    # Initial guesses
    xhat[0] = data[0]
    P[0] = 1.0

    for k in range(1, n_iter):
        # Time update
        xhatminus[k] = xhat[k - 1]
        Pminus[k] = P[k - 1] + process_variance

        # Measurement update
        K[k] = Pminus[k] / (Pminus[k] + measurement_variance)
        xhat[k] = xhatminus[k] + K[k] * (data[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return xhat

def low_pass_filter(data, cutoff_freq, fs, order=5):
    """
    Applies a Butterworth Low Pass Filter to the data.

    Args:
        data (numpy.ndarray): The input data to be filtered.
        cutoff_freq (float): The cutoff frequency of the filter.
        fs (float): The sampling frequency of the data.
        order (int, optional): The order of the filter. Default is 5.

    Returns:
        numpy.ndarray: The filtered data.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def savitzky_golay_filter(data, window_length=5, polyorder=2):
    """
    Applies a Savitzky-Golay filter to the data.

    Args:
        data (numpy.ndarray): The input data to be filtered.
        window_length (int, optional): The length of the filter window (must be odd). Default is 5.
        polyorder (int, optional): The order of the polynomial used to fit the samples. Default is 2.

    Returns:
        numpy.ndarray: The filtered data.
    """
    if window_length % 2 == 0:
        window_length += 1  # Window length must be odd
    y = savgol_filter(data, window_length, polyorder)
    return y

def apply_filter(method, data, **kwargs):
    """
    General function to apply a specified filter to the data.

    Args:
        method (str): The filtering method to apply. Choices are:
            - 'ema': Exponential Moving Average filter.
            - 'kalman': Kalman filter.
            - 'low_pass': Low Pass filter.
            - 'savitzky_golay': Savitzky-Golay filter.
        data (numpy.ndarray): The input data to be filtered.
        **kwargs: Additional keyword arguments specific to each filter.

    Returns:
        numpy.ndarray: The filtered data.
    """
    if method == 'ema':
        return ema_filter(data, **kwargs)
    elif method == 'kalman':
        return kalman_filter(data, **kwargs)
    elif method == 'low_pass':
        return low_pass_filter(data, **kwargs)
    elif method == 'savitzky_golay':
        return savitzky_golay_filter(data, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
