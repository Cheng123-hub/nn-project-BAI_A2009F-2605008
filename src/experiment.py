"""Two training modes with complete preflight and Validation-only selection.

Orchestrates multi-experiment execution, controlled ablation studies,
and unbiased model selection based strictly on Validation set performance.
"""

import shutil
import joblib
import numpy as np
from .utils import ROOT, LABELS, read_json, preflight, writable_outputs


# Require exactly the five agreed controlled configurations
def configurations():
    # Load historical experiment hyperparameter list
    configs = read_json(ROOT / "training_config/experiments.json")["experiments"]
    
    # Expected ablation configurations:
    # 1. baseline: [256, 64] hidden units, dropout=0.2, weight_decay=0
    # 2. experiment1: [128, 64] hidden units (smaller capacity ablation)
    # 3. experiment2: dropout=0.4 (stronger dropout regularization ablation)
    # 4. experiment3: weight_decay=0.0001 (mild L2 regularization ablation)
    # 5. experiment4: weight_decay=0.001 (stronger L2 regularization ablation, chosen final model)
    expected = [
        ("baseline", [256, 64], 0.2, 0),
        ("experiment1", [128, 64], 0.2, 0),
        ("experiment2", [256, 64], 0.4, 0),
        ("experiment3", [256, 64], 0.2, 0.0001),
        ("experiment4", [256, 64], 0.2, 0.001),
    ]
    if len(configs) != 5:
        raise ValueError("Five historical configs required")
        
    # Verify that each configuration strictly matches the frozen parameters
    for c, (name, hidden, dropout, decay) in zip(configs, expected):
        fields = {
            "name": name,
            "hidden_dims": hidden,
            "dropout": dropout,
            "weight_decay": decay,
            "learning_rate": 0.001,
            "batch_size": 128,
            "seed": 42,
            "epochs": 25,
        }
        if any(c[k] != v for k, v in fields.items()):
            raise ValueError("Frozen config mismatch")
            
    return configs


# Rank measured Macro-F1, breaking ties with lower Cross-Entropy
def rank_candidates(results):
    # Primary key: Validation Macro-F1 (higher is better)
    # Secondary key: -validation loss (lower loss is better)
    return sorted(
        results,
        key=lambda r: (r["validation"]["macro_f1"], -r["validation"]["loss"]),
        reverse=True,
    )


# Load only Training/Validation; reuse the fitted Training vocabulary
def training_inputs():
    from .data import SparseDataset

    processed = ROOT / "data/processed"
    # Ensure processed label ordering matches the 6 canonical classes
    if read_json(processed / "labels.json") != LABELS:
        raise ValueError("Processed label order differs from the six original classes")
        
    # Load pre-fitted TF-IDF vectorizer and training/validation sparse datasets
    vectorizer = joblib.load(processed / "vectorizer.joblib")
    datasets = {
        split: SparseDataset(processed, split) for split in ("train", "validation")
    }
    
    # Verify feature dimension is exactly 10,000 across datasets
    dimension = len(vectorizer.vocabulary_)
    if dimension != 10000 or any(d.x.shape[1] != dimension for d in datasets.values()):
        raise ValueError("Processed matrices and fitted TF-IDF dimensions do not match")
        
    return datasets, dimension


# Save one shared fitted vectorizer next to the experiment/model folders
def save_bundle(directory):
    destination = directory.parent.parent / "vectorizer.joblib"
    source = ROOT / "data/processed/vectorizer.joblib"
    
    # If a vectorizer already exists at destination, ensure it is completely identical
    if destination.exists():
        expected = joblib.load(source)
        existing = joblib.load(destination)
        if existing.vocabulary_ != expected.vocabulary_ or not np.array_equal(
            existing.idf_, expected.idf_
        ):
            raise ValueError(f"Incompatible shared vectorizer: {destination}")
    else:
        # Copy the pre-fitted vectorizer alongside model checkpoints for self-contained deployment
        shutil.copyfile(source, destination)


# Train to the mounted model folder and print metrics; do not write reports
def train_models(mode, overwrite=False):
    from .train import train

    configs = configurations()
    
    # Select which configurations to execute based on mode
    if mode == "train-final":
        # Train only Experiment 4 directly
        configs = [next(c for c in configs if c["name"] == "experiment4")]
    elif mode != "train-all":
        raise ValueError("Expected train-final or train-all")
        
    final = ROOT / "models/final"
    shared_vectorizer = ROOT / "models/vectorizer.joblib"
    
    # Determine all target directories for safety preflight check
    destinations = (
        [ROOT / "models/experiments" / c["name"] for c in configs]
        if mode == "train-all"
        else []
    ) + [final]
    
    print(
        f"Mode: {mode} | overwrite={'authorized' if overwrite else 'disabled'}",
        flush=True,
    )
    
    # Check for file collisions to prevent accidental overwrite without --overwrite flag
    preflight(destinations, overwrite, extra_files=[shared_vectorizer])
    if overwrite and shared_vectorizer.is_file():
        shared_vectorizer.chmod(0o666)
        
    # Load training and validation datasets
    datasets, dimension = training_inputs()
    results = []
    
    # Train each configuration sequentially
    for config in configs:
        directory = (
            final
            if mode == "train-final"
            else ROOT / "models/experiments" / config["name"]
        )
        result = train(
            config, datasets, dimension, directory, overwrite=overwrite
        )
        save_bundle(directory)
        results.append(result)
        
    # Select best performing model using Validation Macro-F1 (Validation-only selection)
    winner = rank_candidates(results)[0]
    
    # If all experiments were trained, copy the winning candidate into models/final/
    if mode == "train-all":
        writable_outputs(final, overwrite)
        final.mkdir(parents=True, exist_ok=True)
        selected = ROOT / "models/experiments" / winner["name"]
        
        # Every copied artifact belongs to the selected candidate, not a preferred ID.
        for name in (
            "model.pt",
            "architecture.json",
        ):
            shutil.copyfile(selected / name, final / name)
        save_bundle(final)
        
    print(
        f"Selected {winner['name']} | Training Accuracy "
        f"{winner['train_eval']['accuracy']:.6f} | best epoch {winner['best_epoch']} | "
        f"Validation Macro-F1 {winner['validation']['macro_f1']:.6f} | saved {final / 'model.pt'}"
    )
