"""Neural network model definition for emotion classification.

Defines the EmotionMLP architecture: a Multi-Layer Perceptron with two hidden
layers, ReLU non-linearities, and Dropout regularization for 6 emotion classes.
"""

from torch import nn
from .utils import LABELS


# Emotion Multi-Layer Perceptron (MLP) for 6-class classification.
class EmotionMLP(nn.Module):

    # Initialize two-hidden-layer MLP with ReLU and Dropout regularization
    def __init__(self, input_dim, hidden_dims=(256, 64), dropout=0.2, num_classes=6):
        super().__init__()
        
        # Validate hidden layer count and output classes against canonical 6 emotion labels
        if (
            len(hidden_dims) != 2
            or min(input_dim, *hidden_dims) <= 0
            or num_classes != len(LABELS)
        ):
            raise ValueError(
                "Require two positive hidden layers and six output classes"
            )
            
        # Keep layer ordering and names stable for historical state-dict compatibility.
        # Layer 0: Project 10,000 TF-IDF features to first hidden representation (e.g. 256)
        # Layer 1: ReLU non-linear activation f(x) = max(0, x)
        # Layer 2: Inverted Dropout with probability p; randomly drops neurons during training
        #          to prevent co-adaptation and overfitting; acts as identity during eval
        # Layer 3: Project to second hidden bottleneck (e.g. 64)
        # Layer 4: Second ReLU non-linear activation
        # Layer 5: Output linear classifier emitting 6 raw unnormalized logits
        # Note: Softmax is omitted here because CrossEntropyLoss applies LogSoftmax internally
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], num_classes),
        )

    # Forward pass: pass input features through sequential layers to obtain raw logits
    def forward(self, features):
        return self.layers(features)

    # Inspect network layers and count trainable parameters for architecture logging
    def summary(self):
        dimension = self.layers[0].in_features
        rows = []
        
        # Iterate over layers to record shapes, parameter counts, and dropout rates
        for layer in self.layers:
            # Linear layers change feature dimension; activation and dropout preserve it
            output = layer.out_features if isinstance(layer, nn.Linear) else dimension
            rows.append(
                {
                    "type": type(layer).__name__,
                    "input_dim": dimension,
                    "output_dim": output,
                    "parameters": sum(p.numel() for p in layer.parameters()),
                    "dropout": layer.p if isinstance(layer, nn.Dropout) else None,
                }
            )
            dimension = output
            
        # Return complete layer breakdown and total trainable parameter count
        return {
            "layers": rows,
            "trainable_parameters": sum(p.numel() for p in self.parameters()),
        }
