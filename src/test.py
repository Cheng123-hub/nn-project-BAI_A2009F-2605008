"""Explicit final-model Test evaluation; alternate models are prediction-only.

Conducts the official, unbiased evaluation of the final selected model
on the held-out Test split. Enforces strict experimental hygiene by prohibiting
intermediate experimental models from evaluating against the Test set.
"""

import numpy as np
from .utils import ROOT, LABELS
from .predict import DEFAULT_MODEL, resolve_model, load_bundle
from .data import SparseDataset
from .evaluate import evaluate_model, classification_metrics


# Evaluate the official final model on the held-out Test dataset
def evaluate_test(model_path=DEFAULT_MODEL):
    path = resolve_model(model_path)
    
    # Enforce test set reservation policy: only the final model can be evaluated on test
    # This prevents adaptive data snooping / p-hacking from running multiple iterations on test
    if path != (ROOT / DEFAULT_MODEL).resolve():
        raise ValueError(
            "Dataset Test is reserved for models/final/model.pt. Use predict --text with alternate models; do not select experiments using Test."
        )
        
    # Load model architecture, weights, and matched fitted TF-IDF vectorizer
    model, _ = load_bundle(path)
    
    # Load held-out test split SparseDataset
    dataset = SparseDataset(ROOT / "data/processed", "test")
    
    # Verify test feature dimensions match neural network input dimension (10,000)
    if dataset.x.shape[1] != model.layers[0].in_features:
        raise ValueError("Test feature dimensions do not match the saved model")
        
    print(f"Model: {path} | Test samples: {len(dataset)}", flush=True)
    
    # Compute test classification metrics in frozen evaluation mode (Dropout OFF, inference_mode ON)
    metrics, actual, _, _ = evaluate_model(model, dataset)
    
    # Display test accuracy
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    
    # Display unweighted class-averaged precision and recall
    print(
        f"Macro Precision: {metrics['macro_precision']:.6f} | Macro Recall: {metrics['macro_recall']:.6f}"
    )
    
    # Display balanced Macro F1 and support-weighted F1
    print(
        f"Macro F1: {metrics['macro_f1']:.6f} | Weighted F1: {metrics['weighted_f1']:.6f}"
    )
    
    # Display formatted confusion matrix showing true labels vs predicted labels
    print("Confusion Matrix (true -> predicted counts):")
    for label, row in zip(LABELS, metrics["confusion_matrix"]):
        cells = " | ".join(
            f"{predicted} {count}" for predicted, count in zip(LABELS, row)
        )
        print(f"  {label}: {cells}")
        
    # Find majority class from training labels for baseline reference (Zero-Rule baseline)
    majority = int(
        np.bincount(
            np.load(ROOT / "data/processed/train_labels.npy"), minlength=len(LABELS)
        ).argmax()
    )


if __name__ == "__main__":
    import sys
    from .main import main

    main(["test", *sys.argv[1:]])
