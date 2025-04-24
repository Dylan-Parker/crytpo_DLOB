# training/trainer.py
import os
import torch
import random
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional
from models.mlp_basic_model import BasicMLPModel
from torcheval.metrics.functional import multiclass_f1_score

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)


class Trainer:
    def __init__(
            self,
            config : Config,
            train_dataset: torch.utils.data.Dataset,
            val_dataset: torch.utils.data.Dataset,
            test_dataset: Optional[torch.utils.data.Dataset] = None,
            device: Optional[torch.device] = None,
            output_dir: Optional[str] = None
    ):
        self.device = device or self._get_device()
        self.output_dir = output_dir
        self.config = config
        self.batch_size = config.train.batch_size
        self.num_workers = config.train.num_workers
        self.lr = config.train.lr
        self.n_epochs = config.train.n_epochs
        self.train_ds = train_dataset
        self.val_ds = val_dataset
        self.test_ds = None or test_dataset
        self._set_seed((config.seed))

        # data loaders
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )
        if self.test_ds is not None:
            self.test_loader = DataLoader(
                self.test_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False
            )
        print(f"\nDatasets and DataLoaders created.")
        print(f"Number of training batches: {len(self.train_loader)}")
        print(f"Number of validation batches: {len(self.val_loader)}")
        if self.test_ds is not None:
            print(f"Number of testing batches: {len(self.test_loader)}")

        self.train_loss = []
        self.val_loss = []
        self.train_score = []
        self.val_score = []

        # model / criterion / optimizer
        self.model = self._build_model().to(self.device)
        self.criterion = self._init_criterion()
        self.optimizer = self._init_optimizer(self.model)

    @staticmethod
    def _get_device():
        # if you want to default to cuda first change order.
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS.")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA (gpu: {torch.cuda.get_device_name(0)}).")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _build_model(self):
        if self.config.model.type == "mlp_basic_model":
            return BasicMLPModel(self.config, self.device)

    def _init_criterion(self):
        # Override if you need a different loss
        return nn.CrossEntropyLoss()

    def _init_optimizer(self, net: nn.Module):
        opt_cfg = self.config.optimizer
        if opt_cfg.type.lower() == 'sgd':
            return torch.optim.SGD(
                net.parameters(),
                lr=self.lr,
                momentum=opt_cfg.momentum,
                weight_decay=opt_cfg.weight_decay
            )
        elif opt_cfg.type.lower() == 'adamw':
            return torch.optim.AdamW(
                net.parameters(),
                lr=self.lr,
                betas=opt_cfg.betas,
                weight_decay=opt_cfg.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_cfg.type}")

    def evaluate(self): # -> metrics_dict
        # Evaluate model on validation set
        self.model.eval()
        running_loss = 0.0
        total = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model.forward(x)
                loss = self.criterion(logits, y)
                running_loss += loss.item() * x.size(0)
                total += x.size(0)

                preds = logits.argmax(dim=1)
                all_preds.append(preds)
                all_targets.append(y)
        val_loss = running_loss / total
        self.val_loss.append(val_loss)

        preds_tensor = torch.cat(all_preds)
        target_tensor = torch.cat(all_targets)
        val_f1 = multiclass_f1_score(
            input=preds_tensor,
            target=target_tensor,
            num_classes=3,
            average="macro"
        )
        self.val_score.append(val_f1)
        print(f"  ↳ Val loss: {val_loss:.4f} — Val F₁: {val_f1:.4f}")

    def train(self): # -> training_history
        # Main training loop
        self.train_loss = []
        self.val_loss = []
        self.train_score = []
        self.val_score = []
        for epoch in range(1, self.n_epochs + 1):
            self.model.train()
            running_loss = 0.0
            total = 0
            all_preds = []
            all_targets = []

            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model.forward(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * x.size(0)
                total += x.size(0)

                preds = logits.argmax(dim=1)
                all_preds.append(preds)
                all_targets.append(y)

            epoch_loss = running_loss / total
            self.train_loss.append(epoch_loss)

            preds_tensor = torch.cat(all_preds)
            target_tensor = torch.cat(all_targets)
            train_f1 = multiclass_f1_score(
                input=preds_tensor,
                target=target_tensor,
                num_classes=3,
                average="macro"
            )
            self.train_score.append(train_f1)
            print(f"Epoch {epoch}/{self.n_epochs} — "
                  f"Train loss: {epoch_loss:.4f} — Train F₁: {train_f1:.4f}")
            self.evaluate()


    def save_model(self, name: str = "model.pt"):
        path = os.path.join(self.output_dir, name)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")