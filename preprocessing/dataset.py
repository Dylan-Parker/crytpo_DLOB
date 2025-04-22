# New File: data_handling/dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset

def create_sequences(input_data: np.ndarray, target_data: np.ndarray, sequence_length: int):
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
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


class LOBSequenceDataset(Dataset):
    """
    PyTorch Dataset for LOB sequences.
    Handles conversion to tensor and permutation for Conv1D.
    """
    def __init__(self, features_seq: np.ndarray, labels_seq: np.ndarray, device=None):
        """
        Args:
            features_seq (np.ndarray): Sequences of features (num_sequences, seq_len, num_features).
            labels_seq (np.ndarray): Corresponding labels (num_sequences,).
            device: The torch device to move tensors to ('cuda', 'cpu', or None).
        """
        self.features = features_seq
        self.labels = labels_seq
        self.device = device

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Get sequence and label
        x_np = self.features[idx] # Shape: (seq_len, num_features)
        y_np = self.labels[idx]   # Shape: ()

        # Convert to tensor
        # Permute features for Conv1D: (seq_len, num_features) -> (num_features, seq_len)
        x_tensor = torch.tensor(x_np, dtype=torch.float32).permute(1, 0)
        y_tensor = torch.tensor(y_np, dtype=torch.long)

        # Move to device if specified
        if self.device:
            x_tensor = x_tensor.to(self.device)
            y_tensor = y_tensor.to(self.device)

        return x_tensor, y_tensor