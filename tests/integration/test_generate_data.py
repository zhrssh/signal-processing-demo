import numpy as np

from pathlib import Path

from src.utils.generate_samples import (
    generate_data,
    generate_healthy_signal,
    generate_faulty_signal
)


def test_generate_data_creates_structure(tmp_path: Path):
    """
    Test for generate_data creating structure.
    """
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
    """
    Test for generate_data generating correct number of files.
    """
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
    """
    Test for generate_data checking if .npy files are valid.
    """
    generate_data(tmp_path, n_healthy=2, n_faulty_per_class=2)

    for file in tmp_path.rglob("*.npy"):
        arr = np.load(file)
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 1
        assert arr.size > 0
        assert np.isfinite(arr).all()


def test_faulty_differs_from_healthy():
    """
    Test for generate_data checking if healthy and faulty signal are different.
    """
    healthy = generate_healthy_signal()
    faulty = generate_faulty_signal(label=1, seed=1)

    # Not equal elementwise
    assert not np.allclose(healthy, faulty)


def test_faulty_signal_is_deterministic():
    """
    Test for generate_data's reproducibility.
    """
    x1 = generate_faulty_signal(label=2, seed=123)
    x2 = generate_faulty_signal(label=2, seed=123)

    assert np.allclose(x1, x2)
