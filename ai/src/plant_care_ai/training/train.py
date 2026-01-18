"""Training module for plant classification models.

Copyright 2025 Plant Care Assistant
"""

import json
import random
import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from plant_care_ai.data.dataset import PlantNetDataset
from plant_care_ai.data.preprocessing import get_inference_pipeline, get_training_pipeline
from plant_care_ai.models.efficientnetv2 import create_efficientnetv2
from plant_care_ai.models.resnet18 import Resnet18


class PlantTrainer:
    """Training class for plant classification models."""

    REQUIRED_CONFIG_KEYS: ClassVar[list[str]] = [
        "data_dir",
        "checkpoint_dir",
        "subset_classes",
        "train_samples_per_class",
        "val_samples_per_class",
        "model",
        "img_size",
        "batch_size",
        "epochs",
        "lr",
        "weight_decay",
        "augm_strength",
    ]

    def __init__(
        self,
        config: dict[str, Any],
        model: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        scheduler: _LRScheduler | None = None,
        *,
        verbose: bool = True,
    ) -> None:
        """Initialize trainer.

        Args:
            config: Configuration dictionary with training parameters.
            model: Optional pre-initialized model. If None, will be built from config.
            optimizer: Optional pre-initialized optimizer. If None, will be created.
            scheduler: Optional pre-initialized scheduler. If None, will be created.
            verbose: Whether to print progress information.

        """
        self._validate_config(config)
        self.config = config
        self.verbose = verbose
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.scaler = torch.amp.GradScaler("cuda")
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.num_classes = None

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_top5": [],
            "lr": [],
        }
        self.best_acc = 0.0
        self.best_epoch = 0

        self.class_to_idx = None  # plant_id (str) -> model_output_idx (int)
        self.idx_to_class = None  # model_output_idx (int) -> plant_id (str)
        self.id_to_name = None  # plant_id (str) -> plant_name (str)

        seed = 42
        random.seed(seed)
        np.random.Generator(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _validate_config(self, config: dict[str, Any]) -> None:
        """Validate that all required config keys are present.

        Args:
            config: Configuration dictionary to validate.

        Raises:
            ValueError: If required keys are missing.

        """
        missing_keys = [key for key in self.REQUIRED_CONFIG_KEYS if key not in config]
        if missing_keys:
            msg = f"Missing required config keys: {missing_keys}"
            raise ValueError(msg)

    def prepare_data(self) -> None:
        """Load and prepare data with subset creation."""
        train_pipeline = get_training_pipeline(
            img_size=self.config["img_size"],
            augm_strength=self.config["augm_strength"],
        )
        val_pipeline = get_inference_pipeline(self.config["img_size"])

        train_dataset = PlantNetDataset(
            self.config["data_dir"],
            split="train",
            transform=train_pipeline,
        )
        val_dataset = PlantNetDataset(
            self.config["data_dir"],
            split="val",
            transform=val_pipeline,
        )

        selected_classes = self.config["subset_classes"]
        self.num_classes = len(selected_classes)

        self.class_to_idx = {species_id: i for i, species_id in enumerate(selected_classes)}
        self.idx_to_class = dict(enumerate(selected_classes))

        if self.verbose:
            print(f"Total train samples in dataset: {len(train_dataset.paths)}")
            print(f"Selected classes: {len(selected_classes)}")

        train_indices = []
        train_class_counts = {}

        for idx, (_, species_id) in enumerate(train_dataset.paths):
            if species_id in self.class_to_idx:
                count = train_class_counts.get(species_id, 0)
                if count < self.config["train_samples_per_class"]:
                    train_indices.append(idx)
                    train_class_counts[species_id] = count + 1

        val_indices = []
        val_class_counts = {}

        for idx, (_, species_id) in enumerate(val_dataset.paths):
            if species_id in self.class_to_idx:
                count = val_class_counts.get(species_id, 0)
                if count < self.config["val_samples_per_class"]:
                    val_indices.append(idx)
                    val_class_counts[species_id] = count + 1

        train_dataset.class_to_idx = self.class_to_idx
        val_dataset.class_to_idx = self.class_to_idx

        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)

        self.train_loader = DataLoader(
            train_subset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config.get("num_workers", 2),
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            val_subset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config.get("num_workers", 2),
            pin_memory=True,
        )

        if self.verbose:
            print(f"Train samples: {len(train_indices)}")
            print(f"Val samples: {len(val_indices)}")

    def build_model(self) -> None:
        """Build model if not provided from outside."""
        if self.model is not None:
            self.model = self.model.to(self.device)
            return

        model_type = self.config["model"]
        if model_type == "resnet18":
            self.model = Resnet18(num_classes=self.num_classes)
        elif model_type == "efficientnetv2":
            self.model = create_efficientnetv2(
                variant=self.config["variant"],
                num_classes=self.num_classes,
            )

        self.model = self.model.to(self.device)

    def setup_training(self) -> None:
        """Set up loss, optimizer, and scheduler (if not provided)."""
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.get("label_smoothing", 0.1)
        )

        if self.optimizer is None:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
            )

        if self.scheduler is None:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["epochs"],
                eta_min=self.config.get("min_lr", 1e-6),
            )

        self.checkpoint_dir = Path(self.config["checkpoint_dir"]) / self.config.get(
            "experiment_name", "exp"
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        with Path(self.checkpoint_dir / "config.json").open("w", encoding="utf-8") as f:
            clean_cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in self.config.items()}
            json.dump(clean_cfg, f, indent=2)

        if self.verbose:
            print(f"\nCheckpoint dir: {self.checkpoint_dir}")

    def train_epoch(self) -> tuple[float, float]:
        """Train one epoch.

        Returns:
            Tuple of (average_loss, accuracy)

        """
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        pbar = tqdm(
            self.train_loader,
            desc="Train",
            leave=False,
            disable=not self.verbose,
        )
        for batch_images, batch_labels in pbar:
            images = batch_images.to(self.device)
            labels = batch_labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            with torch.amp.autocast("cuda"):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

            if self.verbose:
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{100.0 * correct / total:.1f}%",
                    }
                )

        return total_loss / len(self.train_loader), 100.0 * correct / total

    def train(self) -> dict[str, Any]:
        """Run full training loop.

        Returns:
            Training history dictionary

        """
        if self.verbose:
            print(f"Epochs: {self.config['epochs']}")
            print(f"Train samples: {len(self.train_loader.dataset)}")
            print(f"Val samples: {len(self.val_loader.dataset)}")

        start_time = time.time()

        for epoch in range(1, self.config["epochs"] + 1):
            if self.verbose:
                print(f"Epoch {epoch}/{self.config['epochs']}")

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_top5 = self.validate()

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_top5"].append(val_top5)
            self.history["lr"].append(current_lr)

            print("\nResults:")
            print(f"\tTrain: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"\tVal:   Loss={val_loss:.4f}, Top-1={val_acc:.2f}%, Top-5={val_top5:.2f}%")
            print(f"\tLR: {current_lr:.6f}")

            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                self.best_epoch = epoch

            self.save_checkpoint(epoch, val_acc, val_top5, is_best=is_best)

        with Path(self.checkpoint_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        elapsed_time = time.time() - start_time

        print("\nTraining evaluation:\n")
        print(f"\tTime: {elapsed_time / 60:.1f} minutes")
        print(f"\tBest Val Accuracy: {self.best_acc:.2f}% (epoch {self.best_epoch})")
        print(f"\tCheckpoints: {self.checkpoint_dir}")

        return self.history

    def validate(self) -> tuple[float, float, float]:
        """Validate model.

        Returns:
            Tuple of (average_loss, top1_accuracy, top5_accuracy)

        """
        self.model.eval()

        total_loss, correct_top1, correct_top5, total = 0, 0, 0, 0

        with torch.no_grad():
            pbar = tqdm(
                self.val_loader,
                desc="Val",
                leave=False,
                disable=not self.verbose,
            )
            for batch_images, batch_labels in pbar:
                images = batch_images.to(self.device)
                labels = batch_labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                _, pred = outputs.max(1)
                total += labels.size(0)
                correct_top1 += pred.eq(labels).sum().item()

                _, top5 = outputs.topk(
                    min(5, outputs.size(1)),
                    dim=1,
                    largest=True,
                    sorted=True,
                )
                correct_top5 += top5.eq(labels.view(-1, 1).expand_as(top5)).sum().item()

                if self.verbose:
                    pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "acc": f"{100.0 * correct_top1 / total:.1f}%",
                        }
                    )

        avg_loss = total_loss / len(self.val_loader)
        top1_acc = 100.0 * correct_top1 / total
        top5_acc = 100.0 * correct_top5 / total
        return avg_loss, top1_acc, top5_acc

    def save_checkpoint(
        self,
        epoch: int,
        val_acc: float,
        val_top5: float,
        *,
        is_best: bool = False,
    ) -> None:
        """Save checkpoint.

        Args:
            epoch: Current epoch number.
            val_acc: Validation top-1 accuracy.
            val_top5: Validation top-5 accuracy.
            is_best: Whether this is the best model so far.

        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_acc": val_acc,
            "val_top5": val_top5,
            "best_acc": self.best_acc,
            "config": self.config,
            "history": self.history,
            "class_to_idx": self.class_to_idx,
            "idx_to_class": self.idx_to_class,  # model_idx -> plant_id
            "num_classes": self.num_classes,
        }

        torch.save(checkpoint, self.checkpoint_dir / "last.pth")

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
            print(f"Best model saved (acc: {val_acc:.2f}%)")
