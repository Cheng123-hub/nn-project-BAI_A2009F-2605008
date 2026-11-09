"""Utility functions and global configurations for the emotion classification project.

Provides directory paths, label mappings, deterministic random seeding,
file I/O helpers, and preflight safety checks to ensure experimental reproducibility
and prevent accidental overwriting of trained artifacts.
"""

import json
import platform
import random
from pathlib import Path
import numpy as np
import torch

# Canonical 6-class emotion label list in strict index order (0 to 5)
# 0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise
# Neural network output logits directly map to these indices
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Project root directory: resolved two levels up from this file (src/ -> root)
ROOT = Path(__file__).resolve().parents[1]


# Read and parse a JSON file from the specified path using UTF-8 encoding
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# Serialize and write a Python object to a JSON file with pretty formatting
def write_json(path, value):
    path = Path(path)
    # Create parent directories automatically if they do not exist
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize to JSON with 2-space indentation and disable non-standard NaN
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


# Set random seeds across all libraries to enforce strict experimental reproducibility
def seed_all(seed):
    # Seed Python built-in pseudo-random generator
    random.seed(seed)
    # Seed NumPy random generator
    np.random.seed(seed)
    # Seed PyTorch CPU/GPU operations
    torch.manual_seed(seed)
    # Restrict CPU execution threads to avoid race conditions in parallel matrix multiplication
    torch.set_num_threads(4)
    # Force deterministic algorithms in PyTorch kernels
    torch.use_deterministic_algorithms(True)


# Collect runtime environment metadata for tracking and provenance
def environment():
    import importlib.metadata

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["torch", "numpy", "scipy", "scikit-learn", "joblib"]
        },
    }


# Preserve full numeric precision in machine-readable reports
def write_csv(path, headers, rows):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        # Write header row
        writer.writerow(headers)
        # Write data rows
        writer.writerows(rows)


# Format experiment configuration and hyperparameters into a readable console banner
def format_header(config):
    return (
        f"=== Experiment: {config['name']} ===\n"
        f"Model: MLP | Hidden Layers: {config['hidden_dims']} | Activation: ReLU | Dropout: {config['dropout']}\n"
        f"Optimizer: Adam | Learning Rate: {config['learning_rate']} | Weight Decay: {float(config['weight_decay'])}\n"
        f"Batch Size: {config['batch_size']} | Seed: {config['seed']} | Epochs: {config['epochs']}\n"
    )


# Friendly training accuracy always means frozen full-set eval, not minibatches
def format_epoch(row, epochs):
    training, validation = row["train_eval"], row["validation"]
    return (
        f"epoch {row['epoch']}/{epochs} training accuracy={training['accuracy']:.4f} "
        f"validation accuracy={validation['accuracy']:.4f} "
        f"validation-macro-F1={validation['macro_f1']:.4f} "
        f"training loss={training['loss']:.6f} validation loss={validation['loss']:.6f}"
    )


# Format final experiment summary including best validation Macro-F1 and checkpoint path
def format_summary(result, checkpoint):
    return (
        f"Training Accuracy: {result['train_eval']['accuracy']:.6f} | "
        f"Best epoch: {result['best_epoch']} | Checkpoint: {checkpoint} | "
        f"Validation Macro-F1: {result['validation']['macro_f1']:.6f}"
    )


# Standard filenames generated for each trained model bundle
MODEL_FILES = (
    "model.pt",
    "architecture.json",
)


# Check the complete command's destinations before any training or writes
def preflight(directories, overwrite=False, extra_files=()):
    # Scan target directories for existing model weights or architecture files
    conflicts = [
        directory / name
        for directory in directories
        for name in MODEL_FILES
        if (directory / name).exists()
    ] + [path for path in extra_files if path.exists()]
    
    # Abort if artifacts exist and overwrite was not explicitly enabled
    if conflicts and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite saved model outputs:\n"
            + "\n".join(str(p) for p in conflicts)
            + "\nUse --overwrite only to deliberately replace these model artifacts."
        )
    if conflicts:
        print(
            "WARNING: --overwrite replaces only the selected model artifacts. "
            "Historical Test scores are NOT verified for replacement weights.",
            flush=True,
        )
    return conflicts


# Only explicitly authorized model files may lose their read-only flag
def writable_outputs(directory, overwrite):
    if overwrite:
        for name in MODEL_FILES:
            path = directory / name
            # Restore write permission on existing files
            if path.is_file():
                path.chmod(0o666)
