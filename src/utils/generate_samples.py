"""
Module for generating sample data.
"""
import numpy as np

from src.generate import (
    generate_gaussian_noise,
    generate_signal
)

# Default signal settings
FREQUENCY = 60.0
SAMPLING_RATE = 1000.0
DURATION = 5.0


def generate_healthy_signal() -> np.ndarray:
    """
    Generates a health signal

    Returns:
    (np.ndarray): Health signal.
    """
    signal = generate_signal(
        freq=FREQUENCY, fs=SAMPLING_RATE, duration=DURATION)
    noise = generate_gaussian_noise(signal.size)
    return signal + noise


def generate_faulty_signal(label: int, seed: int = 42) -> np.ndarray:
    """
    Generates faulty signal based on label.

    Paramters:
    label (int): Label for faulty signal (1, 2, 3, 4).
    seed (int): For reproducibility.

    Returns:
    (np.ndarray): Faulty signal
    """

    # Check type
    if type(label) is not int:
        raise TypeError("'label' must be an integer type.")
    if type(seed) is not int:
        raise TypeError("'seed' must be an integer type.")

    if label not in (1, 2, 3, 4):
        raise ValueError("'label' must be in (1, 2, 3, 4)")

    np.random.seed(seed=seed)
    healthy = generate_healthy_signal()
    length = healthy.size

    match label:
        case 1:
            # Amplitude fault
            gain = np.random.uniform(1.5, 3.0)
            return gain * healthy

        case 2:
            # Impulsive noise
            faulty = healthy.copy()
            n_spikes = length // 50
            idx = np.random.choice(length, n_spikes, replace=False)
            faulty[idx] += np.random.uniform(-5.0, 5.0, size=n_spikes)
            return faulty

        case 3:
            # Frequency shift
            shifted = generate_signal(
                freq=FREQUENCY * np.random.uniform(1.2, 1.5),
                fs=SAMPLING_RATE,
                duration=DURATION
            )
            noise = generate_gaussian_noise(shifted.size)
            return shifted + noise

        case 4:
            # Signal dropout
            faulty = healthy.copy()
            start = np.random.randint(length // 4, length // 2)
            width = length // 10
            faulty[start:start + width] = 0.0
            return faulty
