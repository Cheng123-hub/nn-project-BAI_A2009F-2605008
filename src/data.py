"""Data loading and batch generation for sparse TF-IDF text features.

Defines a memory-efficient PyTorch Dataset that maintains features in SciPy
CSR sparse format and densifies batches on-the-fly, plus a seeded DataLoader factory.
"""

import numpy as np
import torch
from scipy.sparse import load_npz
from torch.utils.data import Dataset, DataLoader


# Sparse PyTorch Dataset that keeps TF-IDF on disk and densifies only the current mini-batch.
class SparseDataset(Dataset):

    # Load sparse features and label arrays for a specific dataset split
    def __init__(self, directory, split):
        # Load sparse feature matrix stored as .npz and ensure CSR representation for fast row slicing
        self.x = load_npz(directory / f"{split}.npz").tocsr()
        # Load 1D NumPy array of integer class labels (0..5)
        self.y = np.load(directory / f"{split}_labels.npy")

        # Validate label array integrity: must be 1D integer within valid emotion class bounds [0, 5]
        if (
            self.y.ndim != 1
            or not np.issubdtype(self.y.dtype, np.integer)
            or np.any((self.y < 0) | (self.y > 5))
        ):
            raise ValueError("Invalid labels; expected original IDs 0..5")

        # Ensure feature sample count matches label count exactly
        if len(self.y) != self.x.shape[0]:
            raise ValueError("Feature / label count mismatch")

    # Return the total number of samples in this dataset split
    def __len__(self):
        return len(self.y)

    # Return sample index; batch densification is deferred to the collate function
    def __getitem__(self, index):
        return index

    # Convert a mini-batch of indices into dense PyTorch tensors
    def collate(self, indices):
        # 1. Slice the CSR sparse matrix by batch indices: self.x[indices]
        # 2. Convert sparse slice to dense NumPy float32 array: .toarray()
        # 3. Wrap into PyTorch FloatTensor [batch_size, 10000]
        features_tensor = torch.from_numpy(self.x[indices].toarray().astype(np.float32))
        
        # Convert target labels to 64-bit signed integers (torch.long) required by CrossEntropyLoss
        targets_tensor = torch.from_numpy(self.y[indices]).long()

        return features_tensor, targets_tensor


# Create a PyTorch DataLoader configured with custom sparse collation and reproducible seeding
def loader(dataset, batch_size, shuffle=False, seed=42):
    # A private generator avoids consuming the model/Dropout random stream.
    # num_workers=0 runs synchronous loading in main process for cross-platform determinism
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
