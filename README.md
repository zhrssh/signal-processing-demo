# Signal Processing Demo

This repository contains the source code and notebooks used for the presentation.

## How to install

### Prerequisites:

- You must have `uv` installed in the system. It will handle most of the dependencies needed for this project.
- Download the dataset from this [link](https://www.kaggle.com/datasets/alanchn31/free-spoken-digits).

### How to install

1. Once you've downloaded the dataset. Extract the contents of the `recordings` directory containing the `.wav` files and paste it into the `data/audio/recordings` directory.
2. You should now have a `data/audio/recordings` directory containing 3000+ files.
3. Next, run `uv sync` to synchronize the dependencies needed for this project.
4. After that, run `uv run main.py` to preprocess the raw data into Mel Spectrogram images.
5. Finally, run `uv run jupyter notebook` to run the Jupyter Notebook server.
6. You can now explore the notebooks and try experimenting with the code.

## Notes:

- The `src/utils` directory contains the code to generate some sample data.
