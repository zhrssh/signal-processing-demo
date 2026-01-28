"""
Testing generate_samples.py
"""
import numpy as np
from pathlib import Path
import pytest

from src.generate import (
    generate_gaussian_noise,
    generate_signal
)
from src.utils.generate_samples import (
    generate_healthy_signal,
    generate_faulty_signal,
    generate_data
)
from src.utils.generate_samples import (
    FREQUENCY,
    SAMPLING_RATE,
    DURATION
)


def test_generate_healthy_signal():
    """
    Test for generating a healthy signal.
    """
    sample_signal = generate_signal(FREQUENCY, SAMPLING_RATE, DURATION)
    sample_noise = generate_gaussian_noise(len(sample_signal))
    test_signal = sample_signal + sample_noise

    result = generate_healthy_signal()

    assert result.size == test_signal.size
    assert np.array_equal(result, test_signal)


@pytest.mark.parametrize("label, seed", [
    (2.0, 42),
    ('4', 42),
    (False, 42),
    (1, '0'),
    (2, None),
    (3, False),
    (None, None)
])
def test_generate_faulty_signal_invalid_types(label, seed):
    """
    Test for generating a faulty signal with invalid types.
    """
    with pytest.raises(TypeError):
        generate_faulty_signal(label, seed)


@pytest.mark.parametrize("label", [
    5,
    -1,
])
def test_generate_faulty_signal_invalid_values(label):
    """
    Test for generating a faulty signal with invalid values.
    """
    with pytest.raises(ValueError):
        generate_faulty_signal(label)


@pytest.mark.parametrize("label", [1, 2, 3, 4])
def test_generate_faulty_signal_shape(label):
    """
    Test for generating a faulty signal with good values.
    """
    sig = generate_faulty_signal(label, seed=0)
    assert sig.shape == generate_healthy_signal().shape
