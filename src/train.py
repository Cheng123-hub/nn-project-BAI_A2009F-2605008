"""Fresh initialization, unweighted CE and Validation-selected checkpoints.

Core training loop for the Emotion Multi-Layer Perceptron (MLP).
Implements mini-batch gradient descent with the Adam optimizer, L2 weight decay,
per-epoch frozen evaluation, and validation checkpoint selection.
"""

import time
import torch
from .data import loader
from .model import EmotionMLP
from .evaluate import evaluate_model
from .utils import (
    seed_all,
    write_json,
    preflight,
    writable_outputs,
    format_header,
    format_epoch,
    format_summary,
)

# Train the EmotionMLP model according to a hyperparameter configuration.
def train(config, datasets, input_dim, model_dir, overwrite=False):
    
    # Verify destination files do not collide unless overwrite was explicitly authorized
    preflight([model_dir], overwrite)
    writable_outputs(model_dir, overwrite)
    
    # Enforce strict random seed for reproducibility
    seed_all(config["seed"])
    
    # Initialize the Multi-Layer Perceptron architecture
    model_config = {
        "input_dim": input_dim,
        "hidden_dims": config["hidden_dims"],
        "dropout": config["dropout"],
        "num_classes": 6,
    }
    model = EmotionMLP(**model_config)
    
    # Save model architecture metadata for inference compatibility and transparency
    write_json(model_dir / "architecture.json", model.summary())
    print(format_header(config), flush=True)
    
    # Adam's coupled L2 penalty is configured here; reported CE excludes that penalty.
    # Optimizer computes parameter updates using gradients, momentum, and weight decay
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    
    # Create the training DataLoader with sample shuffling and isolated random generator
    batches = loader(
        datasets["train"], config["batch_size"], shuffle=True, seed=config["seed"]
    )
    
    history, best_key, best_epoch = [], None, None
    start = time.monotonic()
    
    # Main training loop across epochs
    for epoch in range(1, config["epochs"] + 1):
        # Enable Dropout for minibatches; frozen full-set evaluation below disables it.
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        # Mini-batch gradient descent pass
        for x, y in batches:
            # 1. Forward pass: compute raw class logits [batch_size, 6]
            logits = model(x)
            
            # 2. Compute multi-class Cross-Entropy Loss: -log(softmax(logits)[y])
            loss = torch.nn.functional.cross_entropy(logits, y)
            
            # CE consumes logits and integer labels; backpropagate then update parameters.
            # 3. Clear gradients from previous iteration
            optimizer.zero_grad()
            # 4. Backward pass: compute gradients via autograd chain rule
            loss.backward()
            # 5. Optimizer step: update model parameters based on Adam equations and weight decay
            optimizer.step()
            
            # Accumulate running metrics for online monitoring
            total_loss += loss.item() * len(y)
            correct += int((logits.argmax(1) == y).sum())
            total += len(y)
            
        # Post-epoch frozen evaluation: disable Dropout to compute unbiased full-set metrics
        train_eval, _, _, _ = evaluate_model(
            model, datasets["train"], config["batch_size"]
        )
        validation, _, _, _ = evaluate_model(
            model, datasets["validation"], config["batch_size"]
        )
        
        # Record complete epoch trajectory
        history.append(
            {
                "epoch": epoch,
                "train_online": {
                    "loss": total_loss / total,
                    "accuracy": correct / total,
                },
                "train_eval": train_eval,
                "validation": validation,
            }
        )
        
        # Model Selection Criterion:
        # Primary key: Validation Macro-F1 (maximizes balanced class performance)
        # Secondary key: -validation["loss"] (minimizes Cross-Entropy loss as tie-breaker)
        key = (validation["macro_f1"], -validation["loss"])
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            # Save the current best model weights
            torch.save(model.state_dict(), model_dir / "model.pt")
            
        print(format_epoch(history[-1], config["epochs"]), flush=True)
        
    # Verification phase: Reload the best saved checkpoint from disk
    model.load_state_dict(
        torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    )
    
    # Re-evaluate loaded checkpoint to verify strict numerical reproducibility
    train_eval, _, _, _ = evaluate_model(model, datasets["train"], config["batch_size"])
    validation, _, _, _ = evaluate_model(
        model, datasets["validation"], config["batch_size"]
    )
    if (
        train_eval != history[best_epoch - 1]["train_eval"]
        or validation != history[best_epoch - 1]["validation"]
    ):
        raise RuntimeError("Best checkpoint does not reproduce its recorded metrics")
        
    # Compile final result summary including generalization gap
    # Generalization gap = (Train Accuracy - Validation Accuracy) in percentage points
    result = {
        "name": config["name"],
        "best_epoch": best_epoch,
        "config": config,
        "train_eval": train_eval,
        "validation": validation,
        "accuracy_gap_pp": 100 * (train_eval["accuracy"] - validation["accuracy"]),
        "seconds": time.monotonic() - start,
    }
    print(format_summary(result, model_dir / "model.pt"), flush=True)
    return result
