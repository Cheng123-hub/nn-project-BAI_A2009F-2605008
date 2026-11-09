# NNB Emotion Text Classification

This project uses a PyTorch multilayer perceptron (MLP) to classify English sentences into six emotions: sadness, joy, love, anger, fear, and surprise. It is the coding component of the Neural Network Basics assignment and demonstrates text preprocessing, model training, checkpoint selection with Validation data, Docker-based evaluation, and prediction for new sentences.

There is no Neutral class. The model transforms text with a fitted TF-IDF vectorizer and predicts one of the six emotion labels. The project contains five historical experiments and the official Final Model. Historical scores apply only to the weights that were evaluated at the time: Validation Accuracy 0.8846 and Macro-F1 0.8539; official Test Accuracy 0.8600, Macro-F1 0.8099, and Weighted-F1 0.8591. Retrained models have new weights and must not be presented with these historical scores.

Each model folder contains its checkpoint and architecture. The fitted vectorizer is stored once at the root of each model collection. The six-label order is fixed in the source code.

## Copying the project

### Copy to another folder or computer

The simplest option is to copy the entire `source-code/` folder. Do not copy only `model.pt`: inference also needs the matching `architecture.json` and the root-level `vectorizer.joblib`. Data and models may not be included in the Git repository, so cloning the repository alone may not provide everything needed to run the project.

For Test evaluation on another computer, copy at least:

- `Dockerfile.test`, `requirements-test.txt`, `src/`, and `.dockerignore`
- `models/final/model.pt`, `models/final/architecture.json`, and `models/vectorizer.joblib`
- `data/processed/test.npz`, `data/processed/test_labels.npy`, and `data/processed/train_labels.npy`

`train_labels.npy` is used to calculate the Training-majority baseline. Dataset files are not needed for single-sentence prediction.

For training on another computer, also copy `Dockerfile.train`, `requirements-train.txt`, `training_config/experiments.json`, and the processed Training and Validation features and labels, `labels.json`, and `vectorizer.joblib`. You do not need to install Python, Conda, or `.venv` on the host to run the Docker containers.

## Using Docker

The commands below are for Docker Desktop using Linux containers. Open PowerShell in the project root. The first build downloads the base image and dependencies.

### 1. Build the Train and Test images

```powershell
docker build -f Dockerfile.train -t nn-train .
docker build -f Dockerfile.test -t nn-test .
```

Train and Test use separate Dockerfiles. Data and models are bind-mounted at runtime and are not packaged into the images. `./data:/app/data:ro` makes data read-only. Mount models as writable for training and read-only for testing.

### 2. Train a model

Running `nn-train` without a subcommand runs `train-final`, which trains only the Experiment 4 configuration. `train-all` trains the five configured experiments and selects the final model by Validation Macro-F1. Neither mode uses Test data for model selection.

```powershell
# Train only Experiment 4 and write to models/final/
docker run --rm -v "./data:/app/data:ro" -v "./models:/app/models" nn-train

# Train all five experiments and select the final model using Validation data
docker run --rm -v "./data:/app/data:ro" -v "./models:/app/models" nn-train train-all
```

The current `models/` folder already contains saved models, so these commands will refuse to overwrite them before training starts. To create a new model version without changing the existing models, mount a new `models-new/` folder. Do not add `--overwrite`:

```powershell
# Create a new Final Model

docker run --rm -v "./data:/app/data:ro" -v "./models-new:/app/models" nn-train train-final

# Train five experiments and select a new Final Model

docker run --rm -v "./data:/app/data:ro" -v "./models-new:/app/models" nn-train train-all
```

Training prints epoch metrics to the terminal. It writes checkpoints and architecture files to the mounted models folder and copies the shared fitted vectorizer to its root. The container does not create or modify files under `report/`. If the target folder already contains model files, training refuses to overwrite them.

### 3. Evaluate the Final Model on Test data

By default, `nn-test` loads `models/final/model.pt` from the mounted folder, reads the processed Test split, and prints Accuracy, Precision, Recall, Macro/Weighted F1, the confusion matrix, and the Training-majority baseline:

```powershell
docker run --rm -v "./data:/app/data:ro" -v "./models:/app/models:ro" nn-test
```

To evaluate a newly trained Final Model in `models-new/`, mount that folder instead:

```powershell
docker run --rm -v "./data:/app/data:ro" -v "./models-new:/app/models:ro" nn-test
```

This Test split has already been used for the historical evaluation. Re-running it reproduces an evaluation; it is not a new independent test. Do not use it to tune parameters or choose a model. Results for new weights do not update or inherit the historical report.

### 4. Predict a new sentence or choose a model

Single-sentence prediction does not read the dataset. The default is the Final Model:

```powershell
docker run --rm -v "./models:/app/models:ro" nn-test --text "I am excited about tomorrow"
```

You can specify an experiment model explicitly:

```powershell
docker run --rm -v "./models:/app/models:ro" nn-test --model models/experiments/experiment1/model.pt --text "I feel hopeful today"
```

To predict with a model in `models-new/`, mount that folder at `/app/models`:

```powershell
docker run --rm -v "./models-new:/app/models:ro" nn-test --text "I feel hopeful today"
```

A single sentence has no known ground-truth label, so this command predicts an emotion but does not calculate Accuracy.

## Notes

- Inputs are English sentences; outputs are one of six emotions. The model has no Neutral class.
- TF-IDF is fit using Training data only. Validation, Test, and new sentences are transformed with the fitted vectorizer.
- New checkpoints are selected using Validation data. Test data is not used for tuning or model selection.
