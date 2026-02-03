from tqdm import tqdm
from pathlib import Path
import torchaudio.transforms as T
import torchaudio
from matplotlib import pyplot as plt
import logging
import os
import sys

import librosa
import matplotlib
matplotlib.use('Agg')


logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def save_spectrogram(specgram, save_path=None):
    """Save spectrogram to path"""
    fig, ax = plt.subplots()

    ax.imshow(
        librosa.power_to_db(specgram),
        origin="lower",
        aspect="auto",
        interpolation="nearest"
    )

    # Remove everything that looks like a label
    ax.axis("off")

    # Remove padding around the image
    plt.tight_layout(pad=0)

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0
        )
        plt.close(fig)
    else:
        raise ValueError("'save_path' should not be None.")


def main():
    log.info("Searching for files in the `data/audio/recordings` folder")

    # Prepare paths
    cwd = Path(os.getcwd())
    datadir = Path(cwd, 'data', 'audio')
    rawdir = Path(datadir, 'recordings')
    specdir = Path(datadir, 'spec')

    log.debug(f"Prepared paths:\n1. {rawdir}\n2. {specdir}")

    # Search files
    raw_audios = []
    for filepath in rawdir.glob("*.wav"):
        raw_audios.append(filepath)

    N = len(raw_audios)
    if N <= 0:
        log.fatal(
            "No files found. Make sure to download the dataset and copy '.wav' files in the 'data/audio/recordings' directory.")
        exit(1)

    log.info(f"Found: {N} '.wav' files")

    # Preprocess files
    log.info(f"Preprocessing {N} files...")
    for filepath in tqdm(raw_audios, desc="Converting to Mel Spectrogram"):
        log.debug(f"Loading {filepath.name}.")
        data, fs = torchaudio.load(filepath)

        # Initialize Mel Spectrogram
        log.debug("Generating Mel Spectrogram object.")
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=fs,
            n_fft=1024,
            win_length=None,
            hop_length=256,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=128,
            mel_scale="htk",
        )

        # Process audio
        log.debug("Processing data...")
        mel_spec = mel_spectrogram(data)

        # Output file to directory
        out = Path(specdir, f"{filepath.stem}.png")
        log.debug("Saving to")
        save_spectrogram(mel_spec[0], out)

    log.info("Done!")
    exit(0)


if __name__ == "__main__":
    main()
