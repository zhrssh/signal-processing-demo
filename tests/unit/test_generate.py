"""
Unit test for signal.generate module
"""
import pytest

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
