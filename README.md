# Sitting Posture Analysis

This project analyzes human sitting posture from images and uses keypoint extraction and classification techniques to assess ergonomics. It leverages TensorFlow for deep learning, [DVC](https://dvc.org/) for data versioning, and has several Jupyter Notebooks for interactive exploration.

This branch is focus on adding cross-validation and statistical analysis to the proposed model.

## Project Structure

- **.dvc/**
  Contains DVC configuration and remote settings (see [`.dvc/config`](.dvc/config)).
- **data/**
  Holds original data and generated outputs (CSV files, processed images). Note that some files are ignored by [`.gitignore`](data/.gitignore) (e.g. `/data.csv`, `/poses_images_out`, `/processed`).
- **dvclive/**
  Contains subdirectories for different model experiments (e.g. `adaboost/`, `nn/`, `xgb/`). Each folder has its respective output plots and metrics.
- **notebooks/**
  Jupyter notebooks for interactive experiments:
  - `classification.ipynb` – performs pose classification using extracted landmarks.
  - `keypoints_extraction.ipynb` – demonstrates keypoint detection and CSV saving (see excerpts starting at [line 32](notebooks/keypoints_extraction.ipynb) and [line 663](notebooks/keypoints_extraction.ipynb)).
- **src/**
  Python scripts and modules:
  - `featurize.py` – extracts features from the original images for further analysis.
  - `prepare.py` – processes the extracted data and writes train, validation, and test CSVs (refer to [src/prepare.py](src/prepare.py) around line 106).
  - `data.py` and `evaluate.py` – contain data handling and evaluation routines.

## Setup and Installation

1. **Clone the Repository:**

   ```sh
   git clone <repository-url>
   cd sitting-posture
   ```

2. **Environment Setup:**

   Install [uv](https://docs.astral.sh/uv)

   ```sh
   uv sync
   ```

3. **DVC Setup:**

   To pull the data from the remote, run:

   ```sh
   uv run dvc pull
   ```

## Running the Pipelines and Notebooks

- **Data Featurization:**

  Run the featurization stage as defined in dvc.lock:

  ```sh
  uv run dvc repro featurize
  ```

- **Prepare Data:**

  After featurization, the prepare.py script writes CSVs into `data/processed` that are used for training and evaluation:

  ```sh
  uv run dvc repro prepare
  ```

- **Interactive Notebook Execution:**

  Open notebooks such as classification.ipynb and keypoints_extraction.ipynb in Visual Studio Code or Jupyter to explore the extraction and classification workflows. These notebooks show how images are processed, CSVs are generated, and how the classification model is defined (see excerpt around line 250).

## Model Training and Evaluation

The classification model uses TensorFlow:
- The input layer is defined with shape `39` for the landmark embedding.
- Dense layers with dropout are used for robustness.
- Model summary information is printed upon instantiation (refer to classification.ipynb starting at line 250).
