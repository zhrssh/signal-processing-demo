"""
Unit test for signal.generate module
"""
import pytest
import numpy as np

from src.generate import generate_signal


@pytest.mark.parametrize("freq, fs, duration", [
    ("10", 1000.0, 1.0),
    (10.0, "1000", 1.0),
    (10.0, 1000.0, "1"),
    (None, 1000.0, 1.0),
])
def test_generate_signal_invalid_types(freq, fs, duration):
    with pytest.raises(TypeError):
        generate_signal(freq, fs, duration)


@pytest.mark.parametrize("freq, fs, duration", [
    (-60.0, 1000.0, 1.0),
    (60.0, -1000.0, 1.0),
    (60.0, 1000.0, -1.0),
])
def test_generate_signal_invalid_values(freq, fs, duration):
    with pytest.raises(ValueError):
        generate_signal(freq, fs, duration)


def test_generate_signal():
    # test settings
    freq = 60.0
    fs = 1000.0
    duration = 1.0

    # invoke function
    signal = generate_signal(freq, fs, duration)

    # assert
    assert signal.size == duration * fs
    assert isinstance(signal, np.ndarray)
    assert signal.shape == (duration * fs,)
