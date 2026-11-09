"""Data visualization and report figure generation.

Generates exploratory data analysis (EDA) charts, training loss and accuracy curves,
annotated confusion matrix heatmaps, and cross-experiment comparison summaries.
"""

import os
import tempfile
import csv
from pathlib import Path
import numpy as np

# Configure non-interactive Matplotlib backend and temporary config directory
# 'Agg' backend renders figures directly to raster image files without requiring a GUI window server
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nnb-emotion-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .utils import LABELS


# Save the current Matplotlib figure to disk with clean layout and close it
def save(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# Generate Exploratory Data Analysis (EDA) visualizations for dataset diagnostics
def eda(rows, audit, directory):
    # Plot 1: Class distribution comparison across splits (Original vs Cleaned)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    x = np.arange(len(LABELS))
    for ax, split in zip(axes, ["train", "validation", "test"]):
        # Bar for original valid samples before cleaning
        ax.bar(
            x - 0.18,
            [audit[split]["original_classes"][label] for label in LABELS],
            0.36,
            label="Original valid",
        )
        # Bar for retained samples after cleaning and deduplication
        ax.bar(
            x + 0.18,
            [audit[split]["retained_classes"][label] for label in LABELS],
            0.36,
            label="Cleaned",
        )
        ax.set_xticks(x, LABELS, rotation=40, ha="right")
        ax.set(title=split, ylabel="Samples")
        ax.legend(fontsize=8)
    save(directory / "class_distribution.png")

    # Plot 2: Histogram of token lengths in cleaned training texts
    plt.figure(figsize=(8, 4.5))
    plt.hist([len(row["text"].split()) for row in rows["train"]], bins=30)
    plt.xlabel("Whitespace-separated words")
    plt.ylabel("Training samples")
    plt.title("Cleaned training text lengths")
    save(directory / "text_lengths.png")

    # Plot 3: Horizontal bar chart summarizing rows pruned during data cleaning
    plt.figure(figsize=(9, 4.5))
    reasons = [
        "missing_or_empty",
        "invalid_label",
        "invalid_text_type",
        "conflicting_label_rows_removed",
        "cross_split_rows_removed",
        "within_split_rows_removed",
    ]
    totals = [sum(audit[s][k] for s in audit) for k in reasons]
    plt.barh(
        [
            "Missing/empty",
            "Invalid label",
            "Invalid text",
            "Conflicting labels",
            "Cross-split overlap",
            "Repeated text",
        ],
        totals,
    )
    # Overlay count values next to each bar
    for i, value in enumerate(totals):
        plt.text(value, i, f" {value}", va="center")
    plt.xlabel("Removed rows across all splits")
    plt.title("Predetermined data cleaning audit")
    plt.xlim(0, max(totals + [1]) * 1.15)
    save(directory / "cleaning_audit.png")


# Plot learning trajectories over training epochs for Loss and Accuracy
def training_curves(history, name, directory):
    epochs = [row["epoch"] for row in history]
    # Identify epoch with the best validation performance (highest Macro-F1, lowest CE loss)
    best = max(
        history, key=lambda r: (r["validation"]["macro_f1"], -r["validation"]["loss"])
    )["epoch"]
    
    # Generate separate curve plots for loss and accuracy
    for key in ["loss", "accuracy"]:
        plt.figure(figsize=(8, 4.5))
        # Plot train online (with Dropout), train eval (frozen), and validation curves
        for split, label in [
            ("train_online", "Train online (dropout on)"),
            ("train_eval", "Train eval"),
            ("validation", "Validation"),
        ]:
            plt.plot(epochs, [row[split][key] for row in history], label=label)
            
        # Add vertical dashed marker showing the selected best epoch checkpoint
        plt.axvline(
            best, color="black", linestyle="--", alpha=0.6, label=f"Best epoch {best}"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy (per example)" if key == "loss" else "Accuracy (0–1)")
        plt.title(name)
        plt.legend()
        save(directory / f"{key}_curves.png")


# Render and save an annotated Confusion Matrix heatmap
def confusion(metrics, title, path):
    cm = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(7.5, 6.5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar(label="Samples")
    plt.xticks(range(len(LABELS)), LABELS, rotation=40, ha="right")
    plt.yticks(range(len(LABELS)), LABELS)
    
    # Overlay numerical sample counts inside each cell with dynamic contrast coloring
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    save(path)


# Rebuild logs and loss/accuracy curves from the retained epoch CSV
def experiment_reports(config):
    from .utils import (
        ROOT,
        read_json,
        format_header,
        format_epoch,
        format_summary,
    )

    name = config["name"]
    directory = ROOT / "report/experiments" / name
    history = []
    
    # Ingest per-epoch metrics recorded in CSV
    with (directory / "epoch_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            history.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_online": {
                        "loss": float(row["train_online_ce_loss"]),
                        "accuracy": float(row["train_online_accuracy"]),
                    },
                    "train_eval": {
                        "loss": float(row["train_eval_ce_loss"]),
                        "accuracy": float(row["train_eval_accuracy"]),
                    },
                    "validation": {
                        "loss": float(row["val_ce_loss"]),
                        "accuracy": float(row["val_accuracy"]),
                        "macro_f1": float(row["val_macro_f1"]),
                    },
                }
            )
    if not history:
        raise ValueError(f"No saved epoch rows in {directory / 'epoch_metrics.csv'}")
        
    best = read_json(directory / "final_validation_metrics.json")
    
    # Reconstruct human-readable training log text file
    lines = [format_header(config)]
    lines.extend(format_epoch(row, config["epochs"]) for row in history)
    lines.append(format_summary(best, f"models/experiments/{name}/model.pt"))
    (directory / "training.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    
    # Generate training curves and validation confusion matrix heatmap
    training_curves(history, name, directory)
    confusion(
        best["validation"],
        f"{name}: historical validation",
        directory / "validation_confusion_matrix.png",
    )
    return best


# Separate five-run Validation comparisons from the single official Test
def summary_reports(configs, results):
    from .utils import ROOT, read_json, write_csv

    tables = ROOT / "report/summary/tables"
    figures = ROOT / "report/summary/figures"
    
    # Table 1: Export hyperparameter configurations of all 5 controlled experiments
    write_csv(
        tables / "experiment_config_comparison.csv",
        [
            "experiment",
            "hidden_layers",
            "dropout",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "seed",
            "max_epochs",
        ],
        [
            [
                c["name"],
                str(c["hidden_dims"]),
                c["dropout"],
                "Adam",
                c["learning_rate"],
                c["weight_decay"],
                c["batch_size"],
                c["seed"],
                c["epochs"],
            ]
            for c in configs
        ],
    )
    
    # Table 2: Export quantitative performance comparison across all 5 experiments
    write_csv(
        tables / "experiment_comparison.csv",
        [
            "experiment",
            "best_epoch",
            "train_eval_accuracy",
            "validation_accuracy",
            "validation_macro_f1",
            "validation_weighted_f1",
            "accuracy_gap_pp",
        ],
        [
            [
                r["name"],
                r["best_epoch"],
                r["train_eval"]["accuracy"],
                r["validation"]["accuracy"],
                r["validation"]["macro_f1"],
                r["validation"]["weighted_f1"],
                r["accuracy_gap_pp"],
            ]
            for r in results
        ],
    )
    
    # Figure 1: Multi-panel bar chart comparing Validation Accuracy, Macro-F1, and Generalization Gap
    positions = np.arange(len(results))
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top panel: Validation Accuracy vs Validation Macro-F1
    for index, (key, label) in enumerate(
        (("accuracy", "Validation Accuracy"), ("macro_f1", "Validation Macro-F1"))
    ):
        bars = axes[0].bar(
            positions + (index - 0.5) * 0.36,
            [r["validation"][key] for r in results],
            0.36,
            label=label,
        )
        axes[0].bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    axes[0].set(
        ylim=(0, 1.12),
        ylabel="Score (0–1)",
        title="Five historical experiments: best Validation checkpoints",
    )
    axes[0].legend(loc="upper right")
    
    # Bottom panel: Generalization gap (Train Accuracy minus Validation Accuracy)
    bars = axes[1].bar(
        positions, [r["accuracy_gap_pp"] for r in results], color="#598e72"
    )
    axes[1].bar_label(bars, fmt="%.2f pp", padding=3)
    axes[1].set(
        ylabel="Accuracy gap (percentage points)",
        title="Full Train-eval Accuracy minus Validation Accuracy",
    )
    axes[1].margins(y=0.2)
    for ax in axes:
        ax.set_xticks(positions, [r["name"] for r in results])
    save(figures / "validation_experiments_comparison.png")
    
    # Read saved Test metrics only; never derive predictions or select a candidate here.
    # These plots describe the recorded official evaluation, never replacement weights.
    recorded = read_json(
        ROOT / "report/experiments/experiment4/final_validation_metrics.json"
    )
    release = {"experiment_id": recorded["name"], "validation": recorded["validation"]}
    
    with (ROOT / "report/test/tables/summary_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        test = {
            row["metric"]: float(row["test_value"])
            for row in csv.DictReader(stream)
        }
    validation = release["validation"]
    keys = ["accuracy", "macro_f1", "weighted_f1"]
    
    # Table 3: Side-by-side comparison of official model's Validation vs Test scores
    write_csv(
        tables / "final_validation_vs_test.csv",
        ["experiment", "split", "samples"] + keys,
        [
            [release["experiment_id"], split, count] + [metrics[key] for key in keys]
            for split, count, metrics in [
                ("Validation", 1993, validation),
                ("Historical Test", 2000, test),
            ]
        ],
    )
    
    # Figure 2: Bar chart contrasting final model Validation vs Test performance
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(3)
    for index, (label, metrics) in enumerate(
        (("Validation (1993)", validation), ("Historical Test (2000)", test))
    ):
        bars = ax.bar(
            positions + (index - 0.5) * 0.36,
            [metrics[k] for k in keys],
            0.36,
            label=label,
        )
        ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_xticks(positions, ["Accuracy", "Macro-F1", "Weighted-F1"])
    ax.set(
        ylim=(0, 1.12),
        ylabel="Score (0–1)",
        title="Official final model: recorded Validation vs Test",
    )
    ax.legend()
    save(figures / "final_results_overview.png")


# Rebuild experiment/summary outputs; retain original EDA and official Test files
def generate_reports():
    from .experiment import configurations

    configs = configurations()
    results = [experiment_reports(config) for config in configs]
    summary_reports(configs, results)
    print(
        "Historical experiment and summary reports rebuilt. EDA and official Test files preserved. No training or evaluation."
    )
