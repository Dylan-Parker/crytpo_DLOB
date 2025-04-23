# models/base_model.py
from abc import ABC, abstractmethod
import torch.nn as nn # Or tensorflow.keras.Model

class BaseModel(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x): # -> model_output (logits/predictions)
        # x: Input batch tensor (batch_size, time_steps, features)
        pass

    def train(self):
        # special training logic / considerations
        pass

    def eval(self):
        # special eval logic / considerations
        pass



# models/base_model.py

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

# Depending on your chosen deep learning framework and other needs, import necessary types
# Example using PyTorch types, adjust if using TensorFlow or others
import torch
from torch.utils.data import DataLoader

class BaseModelWrapper(ABC):
    """
    Abstract Base Class for model wrappers.
    Provides a consistent interface for training, prediction, evaluation,
    saving, and loading different types of models (PyTorch, Sklearn, etc.).
    """
    def __init__(self, config: Dict[str, Any], model_dir: str = './trained_models'):
        """
        Initializes the model wrapper.

        Args:
            config (Dict[str, Any]): Dictionary containing model-specific hyperparameters
                                     and architecture configurations.
            model_dir (str): Directory to save/load trained models. Creates if not exists.
        """
        self.config = config
        self.model: Optional[Any] = None # Holds the actual model (e.g., nn.Module, Sklearn estimator)
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        # Define a default filename pattern for saved models
        self._model_filename_pattern = f"{self.__class__.__name__}_model.pt" # Example for PyTorch

    @abstractmethod
    def build_model(self):
        """
        Builds the underlying model architecture based on self.config.
        Sets the self.model attribute. This method should handle device placement
        if applicable (e.g., self.model.to(device)).
        """
        pass

    @abstractmethod
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader], device: torch.device) -> Dict[str, list]:
        """
        Trains the model using the provided data loaders.
        This method should contain the framework-specific training loop
        (epochs, batches, forward pass, loss, backprop, optimizer step, validation).

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data (optional, can be None).
            device: The device to train on (e.g., torch.device('cuda')).

        Returns:
            Dict[str, list]: Training history (e.g., {'train_loss': [...], 'val_loss': [...], 'val_f1': [...]}).
        """
        pass

    @abstractmethod
    def predict(self, data_loader: DataLoader, device: torch.device) -> Any:
        """
        Makes predictions on the given data.

        Args:
            data_loader: DataLoader for the data to predict on.
            device: The device for inference.

        Returns:
            Any: Predictions in a suitable format (e.g., numpy array of labels or probabilities,
                 torch tensor). Should handle model.eval() mode and moving data to device.
        """
        pass

    @abstractmethod
    def evaluate(self, data_loader: DataLoader, device: torch.device) -> Dict[str, float]:
        """
        Evaluates the model on the given data.

        Args:
            data_loader: DataLoader for the evaluation data.
            device: The device for evaluation.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
                              (e.g., {'loss': 0.5, 'accuracy': 0.8, 'f1': 0.75}).
                              Should handle model.eval() mode.
        """
        pass

    @abstractmethod
    def save(self, epoch: int = -1):
        """
        Saves the model's state (e.g., weights, optimizer state if needed).

        Args:
            epoch (int): Optional epoch number to include in the filename for checkpoints.
                         If -1 (default), saves as the final/best model using the default pattern.
        """
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() before saving.")
        # Implementation depends on the framework (e.g., torch.save for PyTorch)
        pass

    @abstractmethod
    def load(self, filename: Optional[str] = None):
        """
        Loads the model's state from a file.

        Args:
            filename (Optional[str]): Specific model filename (relative to model_dir) to load.
                                      If None, loads the default model file pattern.
        """
        # Ensure the model architecture is built first
        if self.model is None:
            print("Model not built, building default model before loading state...")
            self.build_model()
        # Implementation depends on the framework (e.g., model.load_state_dict for PyTorch)
        pass

    @staticmethod
    @abstractmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Returns a dictionary containing the default hyperparameters and configuration
        settings for this specific model type. Useful for baseline training and
        understanding required parameters.
        """
        pass

    def get_model_path(self, epoch: int = -1) -> str:
        """
        Helper method to construct the full path for saving/loading the model file.

        Args:
            epoch (int): Epoch number for checkpointing. -1 indicates final/best model.

        Returns:
            str: The full file path for the model file.
        """
        if epoch == -1:
            filename = self._model_filename_pattern
        else:
            # Example checkpoint naming: CNNModelWrapper_model_epoch_10.pt
            base, ext = os.path.splitext(self._model_filename_pattern)
            filename = f"{base}_epoch_{epoch}{ext}"
        return os.path.join(self.model_dir, filename)