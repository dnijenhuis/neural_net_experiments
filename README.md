# Cats vs Dogs — Educational CNN (PyTorch)

A minimal PyTorch project for educational purposes. It trains a binary cats-versus-dogs classifier with toggles for BatchNorm, Dropout, and WeightNorm. 
Per-epoch metrics are logged to a CSV. Written purely for educational purposes.

---

## Dataset

This script was written for the following dataset (but it can also be used for other binary image datasets):

- <https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset>

Expected folder layout (update the two paths in the script):

```
cats_dogs/
  train/
    cat/...
    dog/...
  val/
    cat/...
    dog/...
```
## Setup

1) Create environment and install dependencies (torch, torchvision, numpy).
2) Adjust the data and CSV directories in the code.
3) Download dataset and move data to the correct folders.
4) Split the data into a training set and validation set. During the creation of this script, an 80/20 split was used. 

## Usage

By default it runs all 4 experiments:
- Baseline.
- WeightNorm.
- BatchNorm.
- Dropout.

To try other experiments, adjust the common parameters and experiment parameters at the end of the script.

Then run the script:

```bash
main.py
```


## Logging

The script creates a CSV containing run configuration and epoch metrics per row.

---

## Future Improvements

- Optional GPU support.
- Automatic dataset downloading and unpacking.
- Automatic splitting the data into a training- and validation set.
- Exporting a '.pth file' with the weights.

---
