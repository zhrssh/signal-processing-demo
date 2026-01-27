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
    with pytest.raises(TypeError):
        generate_faulty_signal(label, seed)


@pytest.mark.parametrize("label", [
    5,
    -1,
])
def test_generate_faulty_signal_invalid_values(label):
    with pytest.raises(ValueError):
        generate_faulty_signal(label)


@pytest.mark.parametrize("label", [1, 2, 3, 4])
def test_generate_faulty_signal_shape(label):
    sig = generate_faulty_signal(label, seed=0)
    assert sig.shape == generate_healthy_signal().shape


def test_generate_data_creates_structure(tmp_path: Path):
    generate_data(
        path=tmp_path,
        n_healthy=10,
        n_faulty_per_class=5,
        seed=123
    )

    assert (tmp_path / "healthy").exists()

    for label in (1, 2, 3, 4):
        assert (tmp_path / f"faulty_{label}").exists()


def test_generate_data_file_counts(tmp_path: Path):
    n_healthy = 8
    n_faulty = 6

    generate_data(
        path=tmp_path,
        n_healthy=n_healthy,
        n_faulty_per_class=n_faulty,
        seed=0
    )

    assert len(list((tmp_path / "healthy").glob("*.npy"))) == n_healthy

    for label in (1, 2, 3, 4):
        files = list((tmp_path / f"faulty_{label}").glob("*.npy"))
        assert len(files) == n_faulty


def test_saved_arrays_are_valid(tmp_path: Path):
    generate_data(tmp_path, n_healthy=2, n_faulty_per_class=2)

    for file in tmp_path.rglob("*.npy"):
        arr = np.load(file)
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 1
        assert arr.size > 0
        assert np.isfinite(arr).all()


def test_faulty_differs_from_healthy():
    healthy = generate_healthy_signal()
    faulty = generate_faulty_signal(label=1, seed=1)

    # Not equal elementwise
    assert not np.allclose(healthy, faulty)


def test_faulty_signal_is_deterministic():
    x1 = generate_faulty_signal(label=2, seed=123)
    x2 = generate_faulty_signal(label=2, seed=123)

    assert np.allclose(x1, x2)
