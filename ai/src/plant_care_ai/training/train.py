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
            config: Configuration dictionary with training parameters
            verbose: Whether to print progress information
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