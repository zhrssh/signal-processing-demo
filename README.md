# Signal Processing Demo

This repository contains the source code and notebooks used in the signal processing presentation. It demonstrates preprocessing raw audio into Mel spectrograms and experimenting with them in notebooks.

## Installation

### Prerequisites

- uv must be installed. It manages Python and project dependencies for this repository.
  - Installation instructions:
    [`uv` Official Website](https://docs.astral.sh/uv/)
- Dataset download
  - Free Spoken Digits dataset:
    [Kaggle](https://www.kaggle.com/datasets/alanchn31/free-spoken-digits)

## Setup Instructions

1. Prepare the dataset
    - Extract the recordings directory from the downloaded dataset.
    - Copy it into:
        `data/audio/recordings` directory
    - After this step, the directory should contain 3000 `.wav` files.

2. Install dependencies
    `uv sync`

3. Preprocess audio data
    `uv run main.py`
    - This converts raw audio into Mel spectrogram images.
    - Generated images are saved in:
        - data/audio/spec

4. Launch Jupyter Notebook
    `uv run jupyter notebook`

5. Explore
    - Open the notebooks and experiment with the preprocessing and models.

## Notes

- The src/utils directory contains helper code for generating sample synthetic data.

