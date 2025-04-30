# training/trainer.py
import os
import torch
import random
import time
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional
from models.mlp_basic_model import BasicMLPModel
from models.cnn_model import CNNClassifier
from models.base_model import BaseModel
from models.linear_model import LinearModel
from models.mlplob_model import MLPLOB
from torcheval.metrics.functional import multiclass_f1_score
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_tcn import TCN
from models.tcn_model import TCNClassifier

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

class AverageMeter(object):
    """Computes and stores the average and current value. Code credit:CS7643 A2"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        self.device = device or self._get_device()
        # Store the provided output directory and fall back to config if not set
        self.output_dir = output_dir
        self.output_path = output_dir if output_dir is not None else config.output_path
        self.config = config
        self.batch_size = config.train.batch_size
        self.num_workers = config.train.num_workers
        self.lr = config.train.lr
        self.n_epochs = config.train.n_epochs
        self.verbose = config.verbose
        self.train_ds = train_dataset
        self.val_ds = val_dataset
        self.test_ds = None or test_dataset
        self._set_seed((config.seed))
        self.train_loss_meter = AverageMeter()
        self.train_score_meter = AverageMeter()
        self.iter_meter = AverageMeter()
        self.early_stop_patience = self.config.train.early_stop_patience
        self.early_stop_threshold = self.config.train.early_stop_threshold
        #self.use_amp = (self.device.type == "cuda")
        self.use_amp = False
        self.scaler = GradScaler() if self.use_amp else None
        self.output_name = self.config.output_name
        # data loaders
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0 if self.device == "mps" else self.num_workers,
            pin_memory=(self.device == "cuda")
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0 if self.device == "mps" else self.num_workers,
            pin_memory=(self.device == "cuda")
        )
        if self.test_ds is not None:
            self.test_loader = DataLoader(
                self.test_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0 if self.device == "mps" else self.num_workers,
                pin_memory=(self.device == "cuda")
            )

        sample_batch = next(iter(self.train_loader))[0]
        print("One batch shape:", sample_batch.shape)
        # Default: preserve per-sample shape (works for MLP, CNN, etc.)
        self.input_size = sample_batch.shape
        # MLPM will flatten to config.model.in_features, so ensure it matches
        if self.config.model.type == "mlp_basic_model":
            self.input_size = (1, self.config.model.in_features)

        print(f"\nDatasets and DataLoaders created.")
        print(f"Number of training batches: {len(self.train_loader)}")
        print(f"Number of validation batches: {len(self.val_loader)}")
        if self.test_ds is not None:
            print(f"Number of testing batches: {len(self.test_loader)}")
        print(f"Input Size: {self.input_size}")

        self.train_loss = []
        self.train_score = []
        self.val_loss = []
        self.val_score = []
        self.test_loss = []
        self.test_score = []

        # model / criterion / optimizer
        self.model = self._build_model().to(self.device)
        self.criterion = self._init_criterion()
        self.optimizer = self._init_optimizer(self.model)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.scheduler.factor,
            patience=self.config.scheduler.patience,
            threshold=self.config.scheduler.threshold,
            min_lr= 1e-6,
            verbose=True,
        )
    @staticmethod
    def _get_device():
      # Check for TPU (TPU support via torch_xla)
      try:
        import torch_xla.core.xla_model as xm
        print("Using TPU (xla:0).")
        return xm.xla_device()
      except ImportError:
        print("torch_xla not found, TPU is not available.")

        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("Using MPS.")
            return "mps"
        elif torch.cuda.is_available():
            print(f"Using CUDA (gpu: {torch.cuda.get_device_name(0)}).")
            return "cuda"
        else:
            print("Using CPU")
            return "cpu"

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
        if self.config.model.type == "base_model":
            return BaseModel(self.config, self.device, self.input_size)
        elif self.config.model.type == "mlp_basic_model":
            return BasicMLPModel(self.config, self.device, self.input_size)
        elif self.config.model.type == "cnn_model":
            batch_size, num_features, _ = self.input_size
            seq_len = self.train_ds.seq_len
            # Build a dummy input_shape tuple same form as old features.shape=(N_windows, T, F)
            # use len(self.train_ds) for N_windows so that idx-based logic in CNN still works
            input_shape = (len(self.train_ds), seq_len, num_features)
            return CNNClassifier(self.config, input_shape, self.device, self.input_size)
        elif self.config.model.type == "tcn_model":
            #print("input_size:", self.input_size)
            batch_size, num_features, _ = self.input_size
            seq_len = self.config.sequence_length #self.train_ds.seq_len
            #print("batch_size:", batch_size)
            #print("num_features:", num_features)
            # Build a dummy input_shape tuple same form as old features.shape=(N_windows, T, F)
            # use len(self.train_ds) for N_windows so that idx-based logic in CNN still works
            input_shape = (len(self.train_ds), seq_len, num_features)
            return TCNClassifier(self.config, input_shape, self.device, self.input_size)
        elif self.config.model.type == "linear_model":
            return LinearModel(self.config, self.device, self.input_size)
        elif self.config.model.type == "mlplob_model":
            return MLPLOB(self.config, self.device, self.input_size)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model.type}")

    def _init_criterion(self):
        #Ridge regression
        #if loss is not an attribute of config, then use default
        try:
            loss_type = self.config.loss.type.lower()
            if loss_type == "cross_entropy":
                return nn.CrossEntropyLoss()
            elif loss_type == 'ridge':
                l2_lambda = self.config.loss.l2_lambda
                assert l2_lambda > 0, "L2 lambda must be greater than 0 for Ridge regression"
                return nn.MSELoss() + l2_lambda * torch.sum(torch.square(self.model.parameters()))
            ##Lasso regression
            elif self.config.loss.type.lower() == 'lasso':
                l1_lambda = self.config.loss.l1_lambda
                assert l1_lambda > 0, "L1 lambda must be greater than 0 for Lasso regression"
                return nn.MSELoss() + l1_lambda * torch.sum(torch.abs(self.model.parameters()))
        except AttributeError:
            print("Warning: Loss not specified in config, using default CrossEntropyLoss")
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
                weight_decay=opt_cfg.weight_decay,
                eps=opt_cfg.eps
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_cfg.type}")

    def evaluate(self, data_loader): # -> metrics_dict
        # Evaluate model on validation set
        self.model.set_eval_mode()
        running_loss = 0.0
        total = 0

        all_preds = []
        all_targets = []
        score = []
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device, dtype=torch.float32, non_blocking=True), y.to(self.device, dtype=torch.long, non_blocking=True)

                # quick data check
                assert torch.isfinite(x).all(), "NaN/Inf in inputs!"
                assert torch.isfinite(y).all(), "NaN/Inf in labels!"

                with autocast(enabled=self.use_amp):
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                running_loss += loss.item() * x.size(0)
                total += x.size(0)

                preds = logits.argmax(dim=1)
                all_preds.append(preds)
                all_targets.append(y)
        loss = running_loss / total

        preds_tensor = torch.cat(all_preds)
        target_tensor = torch.cat(all_targets)
        f1 = multiclass_f1_score(
            input=preds_tensor,
            target=target_tensor,
            num_classes=3,
            average="macro"
        )
        score = f1.cpu().numpy()
        print(f"  ↳ Val loss: {loss:.4f} — Val F₁: {f1:.4f}")
        return loss, score

    def train(self): # -> training_history
        # Main training loop
        self.train_loss = []
        self.val_loss = []
        self.train_score = []
        self.val_score = []
        self.train_loss_meter.reset()
        self.train_score_meter.reset()
        wait = 0
        best_val = float('inf')
        print("Starting Training")
        print(f"Epochs: {self.n_epochs} | Num Batches: {len(self.train_loader)}")

        for epoch in range(1, self.n_epochs + 1):
            t1 = time.perf_counter()
            self.model.set_train_model()
            all_preds = []
            all_targets = []
        #
            for i, (x, y) in enumerate(self.train_loader):
                start_time = time.perf_counter()
                x, y = x.to(self.device, dtype=torch.float32, non_blocking=True), y.to(self.device, dtype=torch.long, non_blocking=True)
                self.optimizer.zero_grad()
                if self.use_amp:
                    # CUDA AMP branch
                    with autocast():
                        logits = self.model(x)
                        loss = self.criterion(logits, y)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # plain float32 branch
                    #logits = self.model(x)
                    #loss = self.criterion(logits, y)
                    #loss.backward()
                    #self.optimizer.step()

                    with autocast():
                        logits = self.model(x)
                        loss = self.criterion(logits, y)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                self.train_loss_meter.update(loss.item(), x.size(0))
                preds = logits.argmax(dim=1)
                all_preds.append(preds)
                all_targets.append(y)
                self.iter_meter.update(time.perf_counter() - start_time)
                                
                print(f'Epoch[Batch]: [{epoch}][{i}/{len(self.train_loader)}]\t',
                      f'Time {self.iter_meter.val:.3f} ({self.iter_meter.avg:.3f})\t')
                #print("batch=",i+1)
                if i % int(len(self.train_loader)/10) == 0:
                    print(
                        f'Epoch[Batch]: [{epoch}][{i}/{len(self.train_loader)}]\t'
                        f'Avg_Loss {self.train_loss_meter.val:.3f} ({self.train_loss_meter.avg:.3f})\t',
                        f'Time {self.iter_meter.val:.3f} ({self.iter_meter.avg:.3f})\t'
                    )

            epoch_loss = self.train_loss_meter.avg
            self.train_loss.append(epoch_loss)
            preds_tensor = torch.cat(all_preds)
            target_tensor = torch.cat(all_targets)
            train_f1 = multiclass_f1_score(
                input=preds_tensor,
                target=target_tensor,
                num_classes=3,
                average="macro"
            )
            self.train_score_meter.update(train_f1.cpu().numpy(), 1)
            self.train_score.append(train_f1.cpu().numpy())
            print(f"Epoch {epoch}/{self.n_epochs} — "
                  f"Train loss: {epoch_loss:.4f} — Train F₁: {train_f1:.4f}")

            val_loss, val_score = self.evaluate(self.val_loader)
            self.val_loss.append(val_loss)
            self.val_score.append(val_score)
            self.scheduler.step(val_loss)
            t2 = time.perf_counter()
            print(f"Epoch finished in {t2-t1} seconds")
            if val_loss < best_val - self.early_stop_threshold:
                best_val = val_loss
                wait = 0
                path = os.path.join(self.output_path, self.output_name + "_best_model.pt")
                torch.save(self.model.state_dict(), path)
                print(f"  ↳ New best model (val_loss={val_loss:.4f}), checkpoint saved.")
            else:
                wait += 1
                if wait >= self.early_stop_patience:
                    print(f"Stopping early at epoch {epoch} (no improvement in { self.early_stop_patience} epochs).")
                    break

    def test(self):
        loss, score = self.evaluate(self.test_loader)
        self.test_loss.append(loss)
        self.test_score.append(score)
        return loss, score

    def save_model(self, name: str = "model.pt"):
        path = os.path.join(self.output_path, name)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


    def test_train(self):
        print("testing train function wil work at all")
