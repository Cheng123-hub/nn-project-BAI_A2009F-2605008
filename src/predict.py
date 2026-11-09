"""CPU inference from the unchanged frozen Experiment 4 bundle.

Provides single-sentence emotion inference, model bundle loading with LRU caching,
input validation, out-of-vocabulary handling, and an interactive terminal demo.
"""

from functools import lru_cache
import json
import torch
import joblib
from pathlib import Path
import numpy as np
from .utils import ROOT, LABELS, read_json, seed_all
from .model import EmotionMLP

# Default checkpoint path pointing to the official final model
DEFAULT_MODEL = "models/final/model.pt"


# Resolve relative or absolute model path to a canonical absolute Path
def resolve_model(path=DEFAULT_MODEL):
    path = Path(path)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found: {path}. Mount/copy the saved models folder."
        )
    return path


# Load model architecture, weights, and matched TF-IDF vectorizer bundle with LRU caching
@lru_cache(maxsize=6)
def load_bundle(path=DEFAULT_MODEL):
    path = resolve_model(path)
    bundle = path.parent
    architecture_path = bundle / "architecture.json"
    
    # Locate shared vectorizer joblib in the models directory hierarchy
    model_root = (
        bundle.parent.parent
        if bundle.parent.name == "experiments"
        else bundle.parent
    )
    vectorizer_path = model_root / "vectorizer.joblib"
    
    # Ensure both architecture JSON and fitted vectorizer exist on disk
    for required in (architecture_path, vectorizer_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"Missing {required}. Copy the model architecture and shared fitted vectorizer."
            )
            
    # Read saved architecture specification
    architecture = read_json(architecture_path)
    linear = [
        row for row in architecture.get("layers", []) if row.get("type") == "Linear"
    ]
    dropout = next(
        (
            row.get("dropout")
            for row in architecture.get("layers", [])
            if row.get("type") == "Dropout"
        ),
        None,
    )
    if len(linear) != 3 or dropout is None:
        raise ValueError(f"Invalid MLP architecture: {architecture_path}")
        
    # Reconstruct configuration parameters for EmotionMLP from architecture metadata
    config = {
        "input_dim": linear[0]["input_dim"],
        "hidden_dims": [linear[0]["output_dim"], linear[1]["output_dim"]],
        "dropout": dropout,
        "num_classes": linear[-1]["output_dim"],
    }
    if config["num_classes"] != len(LABELS):
        raise ValueError("Saved architecture does not match the six-label mapping")
        
    # Load scikit-learn TF-IDF vectorizer
    vectorizer = joblib.load(vectorizer_path)
    if (
        len(vectorizer.vocabulary_) != config["input_dim"]
        or len(vectorizer.idf_) != config["input_dim"]
    ):
        raise ValueError("Model dimensions and fitted vectorizer do not match")
        
    # Verify TF-IDF extraction hyperparameters match expectations
    params = vectorizer.get_params()
    expected = {
        "ngram_range": (1, 2),
        "stop_words": None,
        "sublinear_tf": True,
        "lowercase": True,
    }
    if any(params.get(key) != value for key, value in expected.items()):
        raise ValueError("Fitted vectorizer configuration is incompatible")
    if vectorizer.dtype != np.float32:
        raise ValueError("Expected the original float32 TF-IDF representation")
        
    # Seed deterministic state and instantiate network
    seed_all(42)
    model = EmotionMLP(**config)
    
    # Load state dict weights safely onto CPU (weights_only=True prevents arbitrary code execution)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    # Deactivate Dropout for deterministic inference
    model.eval()
    return model, vectorizer


# Predict the emotion class and confidence score for an input English sentence
def predict(text, model_path=DEFAULT_MODEL):
    # Guard against whitespace-only or empty strings
    if not text.strip():
        return {
            "input": text,
            "warning": "Enter a nonempty English sentence.",
            "emotion": None,
        }
        
    # Guard against excessively long input texts
    if len(text) > 10000:
        return {
            "warning": "Input exceeds 10000 characters. Please shorten it; nothing was predicted.",
            "emotion": None,
        }
        
    # Load cached model and vectorizer
    model, vectorizer = load_bundle(model_path)
    
    # Transform raw text into sparse TF-IDF feature row using frozen training vocabulary
    x = vectorizer.transform([text])
    
    # Check if input text generated any known TF-IDF features (Out-Of-Vocabulary detection)
    if x.nnz == 0:
        return {
            "input": text,
            "emotion": None,
            "warning": "No known TF-IDF features. Use an English sentence; no reliable prediction is available.",
        }
        
    # Execute neural network forward pass without autograd tracking
    with torch.inference_mode():
        # Densify sparse slice -> forward pass -> softmax normalization along class axis
        probs = model(torch.from_numpy(x.toarray())).softmax(1)[0]
        
    # Rank predicted probabilities in descending order
    order = probs.argsort(descending=True)[:3].tolist()
    
    # Return predicted top-1 emotion and confidence score
    result = {
        "input": text,
        "emotion": LABELS[order[0]],
        "confidence": float(probs[order[0]]),
    }
    return result


# Run an interactive command-line Read-Eval-Print Loop (REPL) for real-time text prediction
def demo(model_path=DEFAULT_MODEL):
    print(
        "Saved-model CPU demo. English text; /quit to exit. Confidence is uncalibrated."
    )
    while True:
        try:
            text = input("Text> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if text.strip().lower() in ["/quit", "quit", "exit"]:
            break
        # Print JSON prediction output
        print(json.dumps(predict(text, model_path), ensure_ascii=False, indent=2))
