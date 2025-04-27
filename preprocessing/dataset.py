# New File: data_handling/dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset

def create_sequences(input_data: np.ndarray, target_data: np.ndarray, sequence_length: int, precision):
    """
    Creates sequences for time series model input.

    Args:
        input_data (np.ndarray): Array of features (num_samples, num_features).
        target_data (np.ndarray): Array of labels (num_samples,).
        sequence_length (int): Desired length of each sequence (T).

    Returns:
        tuple: (np.ndarray, np.ndarray) - Sequences (X) and corresponding labels (y).
               X shape: (num_sequences, sequence_length, num_features)
               y shape: (num_sequences,)
    """
    X, y = [], []
    if len(input_data) <= sequence_length:
        print(f"Warning: Data length ({len(input_data)}) is less than or equal to sequence length ({sequence_length}). Cannot create sequences.")
        return np.array(X), np.array(y)

    for i in range(len(input_data) - sequence_length):
        # Sequence: features from index i to i+T-1
        X.append(input_data[i:(i + sequence_length), :])
        # Label: corresponds to the time step *after* the sequence ends (index i+T)
        # Adjust if your label definition differs (e.g., label for last element: target_data[i + sequence_length - 1])
        # y.append(target_data[i + sequence_length]) # Predict step immediately after sequence
        y.append(target_data[i + sequence_length -1]) # Label corresponds to last element of sequence

    # Check if sequences were actually created before converting to numpy array
    if not X:
        return np.array(X), np.array(y)

    # Assuming float32 for features and int64 for labels based on typical PyTorch usage
    return np.array(X, dtype=precision), np.array(y, dtype=np.int8)


class LOBSequenceDataset(Dataset):
    """
    PyTorch Dataset for LOB sequences.
    Handles conversion to tensor and permutation for Conv1D.
    """
    def __init__(self, features_seq: np.ndarray, labels_seq: np.ndarray, device=torch.device("cpu")):
        """
        Args:
            features_seq (np.ndarray): Sequences of features (num_sequences, seq_len, num_features).
            labels_seq (np.ndarray): Corresponding labels (num_sequences,).
            device: The torch device to move tensors to ('cuda', 'cpu', or None).
        """
        self.features = features_seq
        self.labels = labels_seq
        self.device = device
        self.seq_len = features_seq.shape[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Get sequence and label
        x_np = self.features[idx] # Shape: (seq_len, num_features)
        y_np = self.labels[idx]   # Shape: ()

        # Convert to tensor
        # Permute features for Conv1D: (seq_len, num_features) -> (num_features, seq_len)
        x_tensor = torch.tensor(x_np).permute(1, 0)
        y_tensor = torch.tensor(y_np)

        # Move to device if specified
        if self.device:
            x_tensor = x_tensor.to(self.device)
            y_tensor = y_tensor.to(self.device, dtype=torch.long)
            # at least on mps, f1 score requires conversion to long / int64

        return x_tensor, y_tensor


# ------------------------------------------------------------------------
# Memory-efficient sliding-window Dataset for LOB data
class LOBLazySequenceDataset(torch.utils.data.Dataset):
    """
    A memory‑efficient Dataset that generates sliding‑window sequences on‑the‑fly.

    Parameters
    ----------
    features : np.ndarray
        Raw feature matrix of shape (N, F).
    labels : np.ndarray
        Label vector of shape (N,).
    seq_len : int
        Length T of each sliding window.
    stride : int, default 1
        Step size between consecutive windows.  stride=1 ⇒ fully overlapping windows.
    device : torch.device or None
        If given, tensors are moved to this device during __getitem__.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels:   np.ndarray,
        seq_len:  int,
        stride:   int = 1,
        device:   torch.device | None = None,
    ):
        assert features.ndim == 2, "features must have shape (N, F)"
        assert len(features) == len(labels), "features and labels length mismatch"
        assert seq_len > 0, "seq_len must be positive"
        assert stride > 0, "stride must be positive"
        assert len(features) >= seq_len + 1, "Need at least seq_len+1 rows"

        self.x = features.astype(np.float32, copy=False)   # ensure fp32 but avoid copy if already
        self.y = labels.astype(np.int64,   copy=False)     # int64 for CE/F1
        self.seq_len = seq_len
        self.stride  = stride
        self.device  = device

        # pre‑compute length to avoid recalculation
        self._n_samples = (len(self.x) - seq_len) // stride + 1

    def __len__(self):
        return self._n_samples

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self._n_samples:
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self._n_samples}")

        start = idx * self.stride
        end   = start + self.seq_len

        x_win = torch.as_tensor(self.x[start:end], dtype=torch.float32)
        # (T, F) -> (F, T) for Conv1d channel‑first
        x_win = x_win.permute(1, 0)

        # label corresponds to the *last* element of the window
        y_lab = torch.as_tensor(self.y[end - 1], dtype=torch.long)

        if self.device is not None:
            x_win = x_win.to(self.device)
            y_lab = y_lab.to(self.device)

        return x_win, y_lab