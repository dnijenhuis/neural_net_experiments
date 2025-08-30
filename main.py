# This module trains a binary cats-versus-dogs classifier on CPU using PyTorch.
# Its purpose is purely to serve as an educational exercise for its author.
# It allows to experiment with training a CNN using various normalization-techniques, parameters, etc.
#
# Data is loaded from train/val folders, resized to 128×128, and processed in batches.
# Training uses Adam with BCEWithLogitsLoss. For every epoch, the parameters and metrics are logged to CSV.
# This allows for easy conversion to graphic representations.
#
# The four experiments are:
# 1. Baseline;
# 2. WeightNorm;
# 3. BatchNorm;
# 4. Dropout.
#
# Other parameters that can be adjusted are batch size, number of epochs, dropout chance, learning rate and number
# of channels/feature maps.

import os # For file/directory management.
import csv # For logging into the CSV.
from datetime import datetime # For timestamping the log file.
from typing import List

import numpy as np # Numerical operations.
import torch # Various Torch imports needed for models and training.
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.parametrizations import weight_norm
from torchvision import datasets, transforms # Specific for image handling (computer vision).
from torch.utils.data import DataLoader


# Build the CNN using nn.Sequential.
# Returns raw logits (no Sigmoid); pair with BCEWithLogitsLoss.
# Bias is disabled in convs when BN is used.
def build_model(use_bn: bool, use_dropout: bool, use_wn: bool,
                channels: List[int], dropout_p: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_ch = 3  # RGB input.
    for out_ch in channels:
        # 3×3 size, 1 padding. Disable bias if BN is on (BN provides its own shift/scale).
        conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn)
        if use_wn:
            conv = weight_norm(conv)
        layers += [
            conv,
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]
        in_ch = out_ch

    # GAP + classifier head.
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]

    fc1: nn.Module = nn.Linear(in_ch, 128)
    if use_wn:
        fc1 = weight_norm(fc1)
    layers += [fc1, nn.ReLU(inplace=True)]
    if use_dropout:
        layers += [nn.Dropout(p=dropout_p)]

    final_fc: nn.Module = nn.Linear(128, 1)  # single logit for binary classification.
    if use_wn:
        final_fc = weight_norm(final_fc)
    layers += [final_fc]

    return nn.Sequential(*layers)


def run_experiment(
    batch_size: int,        # Batch size per training steps. By default set to 32, resulting in 0.8*25000/32=625 steps.
    epochs: int,            # Number of epochs. By default set to 5.
    use_bn: bool,           # Batch Normalization used or not.
    use_dropout: bool,      # Dropout used or not.
    use_wn: bool,           # Weight Normalization used or not.
    dropout_p: float,       # Dropout rate (if used at all).
    learning_rate: float,   # Learning rate.
    channels_per_block: List[int], # Used feature maps, i.e. [32, 64, 128, 256, 512].
    log_suffix: str,    # Takes a hardcoded file name for each experiment.
    seed: int,          # Seed. By default fixed for reproducability, but mainly for experiment comparability.
    log_csv_path: str,  # Takes the name and path of the log file.
):
    """This function builds and trains a CNN using the parameters of the experiment(s). It takes various model
    parameters as arguments. It evaluates losses/accuracies per epoch, and logs metrics with the used parameters
    into a CSV.
    """

    # Determine seed for function (fixed seed).
    np.random.seed(seed)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)  # deterministic shuffling for DataLoader

    # Resizes data to 128*128, and converts to multidimensional vector.
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Training and validation directories. These should be manually adjusted by user and they should contain the
    # respective sets of images. Code for this functionality is not included.
    train_dir = "PLACEHOLDER"
    val_dir   = "PLACEHOLDER"

    # Read and wrap the data.
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    val_data   = datasets.ImageFolder(val_dir,   transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  generator=g, num_workers=0)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False,                 num_workers=0)

    # Device setup (CPU), I could not get GPU to work.
    device = torch.device("cpu")

    # Model setup per experiment.
    model = build_model(use_bn, use_dropout, use_wn, channels_per_block, dropout_p).to(device)

    # Use BCEWithLogits() for binary loss. Adam optimizer, set learning rate.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # This code manages the CSV for logging. The CSV-format allows for easy conversion to a graph.
    def init_csv(path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True) # Make sure directory exists.
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0 # Write header if file does not exist.
        f = open(path, "a", newline="", encoding="utf-8") # Open the file.
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                # Column headers.
                "run",
                "use_bn", "use_dropout", "use_wn", "dropout_p",
                "learning_rate", "batch_size", "epochs", "channels_per_block", "seed",
                "epoch", "train_loss", "train_acc", "val_loss", "val_acc"
            ])
            f.flush(); os.fsync(f.fileno())
        return f, writer

    run_name = log_suffix

    # Printing run to the terminal.
    print(
        f"[{run_name}] "
        f"BN={use_bn} | Dropout={use_dropout} (p={dropout_p}) | "
        f"WN={use_wn} | LR={learning_rate} | Batch={batch_size} | Epochs={epochs} | "
        f"Channels={channels_per_block} | Seed={seed}"
    )
    print(f"[{run_name}] logging to CSV: {log_csv_path}")

    # Initialize CSV file and writer.
    csv_file, csv_writer = init_csv(log_csv_path)

    # This for-loop performs the actual training. Every epoch the model is trained and then validated.
    # The metrics are both printed to the console, and logged to the CSV.
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            logits = model(inputs)                # Produces raw logits.
            loss = criterion(logits, labels)      # BCEWithLogitsLoss.
            loss.backward()
            optimizer.step()

            bs = inputs.size(0)
            running_loss += loss.item() * bs
            preds = (logits > 0).int()
            correct += (preds == labels.int()).sum().item()
            total += bs

        # Calculating training metrics and storing to variables.
        train_loss = running_loss / total if total else 0.0
        train_acc = correct / total if total else 0.0

        # Validation.
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                logits = model(inputs)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * inputs.size(0)
                preds = (logits > 0).int()
                val_correct += (preds == labels.int()).sum().item()
                val_total += inputs.size(0)

        # Calculating validation metrics and storing to variables.
        val_loss = val_loss_sum / val_total if val_total else 0.0
        val_acc = val_correct / val_total if val_total else 0.0

        # Printing to console and logging to CSV.
        print(
            f"[{run_name}] Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )
        csv_writer.writerow([
            run_name,
            use_bn, use_dropout, use_wn, dropout_p,
            learning_rate, batch_size, epochs, str(channels_per_block), seed,
            epoch, f"{train_loss:.6f}", f"{train_acc:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}"
        ])
        csv_file.flush()
        os.fsync(csv_file.fileno())

    csv_file.close()


# Below are first the parameters that will be equal in all of the experiments. They can be adjusted though.
if __name__ == "__main__":
    # Common parameters for all experiments.
    batch_size = 32
    epochs = 5
    dropout_p = 0.2
    learning_rate = 0.001
    channels = [32, 64, 128, 256, 512]
    seed = 20250822

    # CSV file.
    log_dir = "PLACEHOLDER"
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_csv_path = os.path.join(log_dir, f"catsdogs_logs_{ts}.csv")

    # Below are the four experiments I ran for my presentation.The parameters can be adjusted per experiment so.
    # 1) Baseline (no norm)
    run_experiment(batch_size, epochs, False, False, False, dropout_p, learning_rate,
                   channels, "exp1_baseline", seed, log_csv_path)

    # 2) WeightNorm only
    run_experiment(batch_size, epochs, False, False, True,  dropout_p, learning_rate,
                   channels, "exp2_wn", seed, log_csv_path)

    # 3) BatchNorm only
    run_experiment(batch_size, epochs, True,  False, False, dropout_p, learning_rate,
                   channels, "exp3_bn", seed, log_csv_path)

    # 4) Dropout
    run_experiment(batch_size, epochs, False, True,  False, dropout_p, learning_rate,
                   channels, "exp4_dropout", seed, log_csv_path)




