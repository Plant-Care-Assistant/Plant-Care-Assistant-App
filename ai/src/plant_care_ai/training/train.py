"""Training module for plant classification models.

Copyright 2025 Plant Care Assistant
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from plant_care_ai.data.dataset import PlantNetDataset
from plant_care_ai.data.preprocessing import get_training_pipeline, get_inference_pipeline
from plant_care_ai.models.resnet18 import Resnet18
from plant_care_ai.models.efficientnetv2 import create_efficientnetv2

class PlantTrainer:
    """Training class for plant classification models."""

    def __init__(
        self,
        config: Dict[str, Any],
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        verbose: bool = True,
    ) -> None:
        """Initialize trainer.

        Args:
            config: configuration dictionary with training parameters
            verbose: whether to print progress information
        """
        self.config = config
        self.verbose = verbose
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
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
        
        self.checkpoint_dir = None

    def prepare_data(self) -> None:
        """Load and prepare data with subset creation."""

        train_pipeline = get_training_pipeline(
            img_size=self.config["img_size"],
            augm_strength=self.config["augm_strength"]
        )
        val_pipeline = get_inference_pipeline(self.config["img_size"])
        
        train_dataset = PlantNetDataset(
            self.config["data_dir"],
            split="train",
            transform=train_pipeline
        )
        val_dataset = PlantNetDataset(
            self.config["data_dir"],
            split="val",
            transform=val_pipeline
        )
        
        selected_classes = self.config["subset_classes"]
        self.num_classes = len(selected_classes)

        self.class_to_idx = {species_id: i for i, species_id in enumerate(selected_classes)}
        #train_dataset.class_to_idx = self.class_to_idx
        #val_dataset.class_to_idx = self.class_to_idx

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
    
    def build_model(self) -> None:
        """Build model if not provided from outside."""
        if self.model is not None:
            self.model = self.model.to(self.device)
            return

        model_type = self.config["model"]
        if model_type == "resnet18":
            self.model = Resnet18(num_classes=self.num_classes)
        elif model_type == "efficientnetv2":
            self.model = create_efficientnetv2(variant=self.config["variant"], num_classes=self.num_classes)
        
        self.model = self.model.to(self.device)
    
    def setup_training(self) -> None:
        """Setup loss, optimizer, and scheduler (if not provided)."""
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.get("label_smoothing", 0.1)
        )
        
        if self.optimizer is None:
            self.optimizer = AdamW(
                self.model.parameters(), 
                lr=self.config["lr"], 
                weight_decay=self.config["weight_decay"]
            )
        
        if self.scheduler is None:
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config["epochs"], 
                eta_min=self.config.get("min_lr", 1e-6)
            )
        
        self.checkpoint_dir = Path(self.config["checkpoint_dir"]) / self.config.get("experiment_name", "exp")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.checkpoint_dir / "config.json", "w") as f:
            clean_cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in self.config.items()}
            json.dump(clean_cfg, f, indent=2)
        
        if self.verbose:
            print(f"\nCheckpoint dir: {self.checkpoint_dir}")
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        pbar = tqdm(self.train_loader, desc="Train", leave=False, disable=not self.verbose)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

            if self.verbose:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100.*correct/total:.1f}%"
                })

        return total_loss / len(self.train_loader), 100.0 * correct / total

    def train(self) -> Dict[str, Any]:
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
            
            print(f"\nResults:")
            print(f"\tTrain: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"\tVal:   Loss={val_loss:.4f}, Top-1={val_acc:.2f}%, Top-5={val_top5:.2f}%")
            print(f"\tLR: {current_lr:.6f}")
            
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                self.best_epoch = epoch
            
            self.save_checkpoint(epoch, val_acc, val_top5, is_best)
        
        with open(self.checkpoint_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        
        elapsed_time = time.time() - start_time
        
        print(f"\nTraining evaluation:\n")
        print(f"\tTime: {elapsed_time/60:.1f} minutes")
        print(f"\tBest Val Accuracy: {self.best_acc:.2f}% (epoch {self.best_epoch})")
        print(f"\tCheckpoints: {self.checkpoint_dir}")

        return self.history

    def validate(self) -> Tuple[float, float, float]:
        """Validate model.
        
        Returns:
            Tuple of (average_loss, top1_accuracy, top5_accuracy)
        """
        self.model.eval()

        total_loss, correct_top1, correct_top5, total = 0, 0, 0, 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Val", leave=False, disable=not self.verbose)
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct_top1 += pred.eq(labels).sum().item()
                
                _, top5 = outputs.topk(min(5, outputs.size(1)), 1, True, True)
                correct_top5 += top5.eq(labels.view(-1, 1).expand_as(top5)).sum().item()
                
                if self.verbose:
                    pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{100.*correct_top1/total:.1f}%"
                    })
        
        return total_loss / len(self.val_loader), 100.0 * correct_top1 / total, 100.0 * correct_top5 / total
    
    def save_checkpoint(self, epoch: int, val_acc: float, val_top5: float, is_best: bool = False) -> None:
        """Save checkpoint.
        
        Args:
            epoch: Current epoch number
            val_acc: Validation top-1 accuracy
            val_top5: Validation top-5 accuracy
            is_best: Whether this is the best model so far
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
        }
        
        torch.save(checkpoint, self.checkpoint_dir / "last.pth")
        
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
            print(f"Best model saved (acc: {val_acc:.2f}%)")