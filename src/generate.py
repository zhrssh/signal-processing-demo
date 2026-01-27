"""
Module for generating syntethic signals. For demo purposes only.
"""
import numpy as np


def generate_signal(freq: float, fs: float, duration: float) -> np.ndarray:
    """
    Generates a signal based on specification.

    Parameters:
    freq (float): Frequency of the signal.
    fs (float): Sampling rate.
    duration (float): Duration in seconds.

    Returns:
    (np.ndarray): Generated signal
    """

    # Check input type
    if type(freq) is not float:
        raise TypeError("'freq' must be a float type.")
    if type(fs) is not float:
        raise TypeError("'fs' must be a float type.")
    if type(duration) is not float:
        raise TypeError("'duration' must be a float type.")

    # Check value
    if freq <= 0:
        raise ValueError("'freq' must be greater than 0.")
    if fs <= 0:
        raise ValueError("'fs' must be greater than 0.")
    if duration <= 0:
        raise ValueError("'duration' must be greater than 0.")

    t = np.arange(0, duration, 1/fs)
    signal = np.sin(2*np.pi*freq*t)
    return signal


def generate_gaussian_noise(size: int, mean: float = 0.0, std: float = 1.0, seed: int = 42) -> np.ndarray:
    """
    Generates noise signal using Gaussian distribution.

    Parameters:
    size (int): Length of the signal.
    mean (float): Average of the values.
    std (float): Standard deviation.
    seed (int): For reproducibility.

    Returns:
    (np.ndarray): Generated noise signal.
    """

    # Check input type
    if type(size) is not int:
        raise TypeError("'size' must be an integer type.")
    if type(mean) is not float:
        raise TypeError("'mean' must be a float type.")
    if type(std) is not float:
        raise TypeError("'std' must be a float type.")
    if type(seed) is not int:
        raise TypeError("'seed' must be an integer type.")

    # Check value
    if size <= 0:
        raise ValueError("'size' must be greater than 0.")
    if std < 0:
        raise ValueError("'std' must be a positive float.")

    np.random.seed(seed=seed)
    return np.random.normal(mean, std, size)
