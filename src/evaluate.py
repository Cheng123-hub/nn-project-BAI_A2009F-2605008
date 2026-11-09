"""Model evaluation and classification performance metrics computation.

Computes multi-class evaluation metrics including Accuracy, Macro/Weighted Precision,
Recall, F1-scores, Confusion Matrix, and Cross-Entropy loss under deterministic evaluation mode.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from .data import loader
from .utils import LABELS


# Include all six classes even if a prediction omits one minority class
def classification_metrics(actual, predicted):
    # Generate full scikit-learn classification report across all 6 predefined classes
    # zero_division=0 sets score to 0.0 without throwing warnings when a class has 0 predictions
    report = classification_report(
        actual,
        predicted,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        zero_division=0,
        output_dict=True,
    )
    
    # Compile metrics dictionary:
    # - accuracy: fraction of correctly classified samples across the dataset
    # - macro_*: unweighted average across classes; treats minority classes equally
    # - weighted_f1: class-weighted F1 by true sample support
    # - confusion_matrix: C[i, j] counts samples from class i predicted as class j
    # - predicted_counts: distribution of predictions to detect class collapse/bias
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=list(range(len(LABELS)))
        ).tolist(),
        "predicted_counts": np.bincount(predicted, minlength=len(LABELS)).tolist(),
    }


# Evaluate a trained model over a full dataset split in frozen evaluation mode
def evaluate_model(model, dataset, batch_size=128):
    # Disable Dropout and gradient tracking for comparable frozen-set metrics.
    model.eval()
    actual, predicted, confidence = [], [], []
    total_loss = 0.0

    # Execute evaluation without autograd tracking to save memory and CPU cycles
    with torch.inference_mode():
        # Iterate over mini-batches using the sparse-to-dense DataLoader
        for features, target in loader(dataset, batch_size):
            # Compute raw unnormalized logits [batch_size, 6]
            logits = model(features)
            
            # Accumulate unweighted Cross-Entropy loss across all samples in the batch
            total_loss += torch.nn.functional.cross_entropy(
                logits, target, reduction="sum"
            ).item()
            
            # Apply Softmax along class axis (dim=1) to obtain normalized probabilities
            probabilities = logits.softmax(1)
            
            # Record ground-truth labels, predicted class indices (argmax), and confidence scores (max prob)
            actual.extend(target.tolist())
            predicted.extend(logits.argmax(1).tolist())
            confidence.extend(probabilities.max(1).values.tolist())

    # Compute comprehensive classification metrics (accuracy, macro F1, confusion matrix)
    result = classification_metrics(actual, predicted)
    
    # Unweighted, per-example CE; does not include the Adam L2 penalty.
    # Divide total accumulated loss by total number of evaluated samples
    result["loss"] = total_loss / len(actual)
    return result, actual, predicted, confidence
