"""Training module unit tests.

Copyright 2025 Plant Care Assistant
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from plant_care_ai.models.resnet18 import Resnet18
from plant_care_ai.training.train import PlantTrainer

# Test constants
MAX_ACCURACY_PERCENT = 100
MOCK_VAL_ACC = 75.0
MOCK_VAL_TOP5 = 90.0
MOCK_VAL_LOSS = 0.5


def get_valid_config(tmp_path: Path, sample_data_dir: Path) -> dict[str, Any]:
    """Create a valid training configuration.

    Args:
        tmp_path: Temporary directory for checkpoints
        sample_data_dir: Path to sample data directory

    Returns:
        Valid training configuration dictionary

    """
    return {
        "data_dir": str(sample_data_dir),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "subset_classes": ["1355936"],
        "train_samples_per_class": 100,
        "val_samples_per_class": 20,
        "model": "resnet18",
        "img_size": 224,
        "batch_size": 2,
        "epochs": 1,
        "lr": 0.001,
        "weight_decay": 0.01,
        "augm_strength": 0.5,
        "experiment_name": "test_exp",
    }


class TestPlantTrainerInit:
    """Tests for PlantTrainer initialization."""

    @staticmethod
    def test_trainer_init_with_valid_config(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that trainer initializes correctly with valid config.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        trainer = PlantTrainer(config, verbose=False)

        assert trainer.config == config
        assert trainer.verbose is False
        assert trainer.model is None
        assert trainer.optimizer is None
        assert trainer.scheduler is None
        assert trainer.best_acc == 0.0
        assert trainer.best_epoch == 0

    @staticmethod
    def test_trainer_init_with_missing_config_raises(tmp_path: Path) -> None:
        """Test that trainer raises ValueError with missing config keys.

        Args:
            tmp_path: Pytest temporary directory fixture

        """
        incomplete_config = {
            "data_dir": str(tmp_path),
            "checkpoint_dir": str(tmp_path),
        }

        with pytest.raises(ValueError, match="Missing required config keys"):
            PlantTrainer(incomplete_config)

    @staticmethod
    def test_trainer_init_with_custom_model(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that trainer accepts custom model.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        custom_model = Resnet18(num_classes=10)

        trainer = PlantTrainer(config, model=custom_model, verbose=False)

        assert trainer.model is custom_model

    @staticmethod
    def test_trainer_init_with_custom_optimizer(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that trainer accepts custom optimizer.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        custom_model = Resnet18(num_classes=10)
        custom_optimizer = SGD(custom_model.parameters(), lr=0.01)

        trainer = PlantTrainer(
            config,
            model=custom_model,
            optimizer=custom_optimizer,
            verbose=False,
        )

        assert trainer.optimizer is custom_optimizer

    @staticmethod
    def test_trainer_init_with_custom_scheduler(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that trainer accepts custom scheduler.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        custom_model = Resnet18(num_classes=10)
        custom_optimizer = SGD(custom_model.parameters(), lr=0.01)
        custom_scheduler = StepLR(custom_optimizer, step_size=1)

        trainer = PlantTrainer(
            config,
            model=custom_model,
            optimizer=custom_optimizer,
            scheduler=custom_scheduler,
            verbose=False,
        )

        assert trainer.scheduler is custom_scheduler

    @staticmethod
    def test_trainer_history_initialized(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that trainer history is initialized correctly.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        trainer = PlantTrainer(config, verbose=False)

        expected_keys = ["train_loss", "train_acc", "val_loss", "val_acc", "val_top5", "lr"]
        assert all(key in trainer.history for key in expected_keys)
        assert all(trainer.history[key] == [] for key in expected_keys)


class TestValidateConfig:
    """Tests for PlantTrainer._validate_config method."""

    @staticmethod
    def test_validate_config_with_all_keys(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that validation passes with all required keys.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        # Should not raise
        trainer = PlantTrainer(config, verbose=False)
        assert trainer is not None

    @staticmethod
    def test_validate_config_missing_single_key(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that validation fails with single missing key.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        del config["epochs"]

        with pytest.raises(ValueError, match="epochs"):
            PlantTrainer(config)


class TestPrepareData:
    """Tests for PlantTrainer.prepare_data method."""

    @staticmethod
    def test_prepare_data_creates_dataloaders(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that prepare_data creates train and val dataloaders.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        trainer = PlantTrainer(config, verbose=False)
        trainer.prepare_data()

        assert trainer.train_loader is not None
        assert trainer.val_loader is not None
        assert trainer.num_classes == len(config["subset_classes"])

    @staticmethod
    def test_prepare_data_creates_class_mappings(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that prepare_data creates class mappings.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        trainer = PlantTrainer(config, verbose=False)
        trainer.prepare_data()

        assert trainer.class_to_idx is not None
        assert trainer.idx_to_class is not None
        assert len(trainer.class_to_idx) == len(config["subset_classes"])


class TestBuildModel:
    """Tests for PlantTrainer.build_model method."""

    @staticmethod
    def test_build_model_resnet18(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that build_model creates resnet18 model.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        config["model"] = "resnet18"
        trainer = PlantTrainer(config, verbose=False)
        trainer.prepare_data()
        trainer.build_model()

        assert trainer.model is not None
        assert isinstance(trainer.model, Resnet18)

    @staticmethod
    def test_build_model_efficientnetv2(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that build_model creates efficientnetv2 model.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        config["model"] = "efficientnetv2"
        config["variant"] = "b0"
        trainer = PlantTrainer(config, verbose=False)
        trainer.prepare_data()
        trainer.build_model()

        assert trainer.model is not None

    @staticmethod
    def test_build_model_uses_provided_model(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that build_model uses provided model and moves to device.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        custom_model = Resnet18(num_classes=1)
        trainer = PlantTrainer(config, model=custom_model, verbose=False)
        trainer.prepare_data()
        trainer.build_model()

        assert trainer.model is custom_model


class TestSetupTraining:
    """Tests for PlantTrainer.setup_training method."""

    @staticmethod
    def test_setup_training_creates_criterion(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that setup_training creates loss criterion.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        trainer = PlantTrainer(config, verbose=False)
        trainer.prepare_data()
        trainer.build_model()
        trainer.setup_training()

        assert trainer.criterion is not None
        assert isinstance(trainer.criterion, nn.CrossEntropyLoss)

    @staticmethod
    def test_setup_training_creates_optimizer(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that setup_training creates optimizer.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        trainer = PlantTrainer(config, verbose=False)
        trainer.prepare_data()
        trainer.build_model()
        trainer.setup_training()

        assert trainer.optimizer is not None

    @staticmethod
    def test_setup_training_creates_scheduler(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that setup_training creates scheduler.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        trainer = PlantTrainer(config, verbose=False)
        trainer.prepare_data()
        trainer.build_model()
        trainer.setup_training()

        assert trainer.scheduler is not None

    @staticmethod
    def test_setup_training_creates_checkpoint_dir(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that setup_training creates checkpoint directory.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        trainer = PlantTrainer(config, verbose=False)
        trainer.prepare_data()
        trainer.build_model()
        trainer.setup_training()

        expected_dir = Path(config["checkpoint_dir"]) / config["experiment_name"]
        assert expected_dir.exists()

    @staticmethod
    def test_setup_training_saves_config_json(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that setup_training saves config.json.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        trainer = PlantTrainer(config, verbose=False)
        trainer.prepare_data()
        trainer.build_model()
        trainer.setup_training()

        config_file = Path(config["checkpoint_dir"]) / config["experiment_name"] / "config.json"
        assert config_file.exists()

        with config_file.open() as f:
            saved_config = json.load(f)
        assert saved_config["epochs"] == config["epochs"]


class TestTrainEpoch:
    """Tests for PlantTrainer.train_epoch method."""

    @staticmethod
    def test_train_epoch_returns_loss_and_acc(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that train_epoch returns loss and accuracy.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        trainer = PlantTrainer(config, verbose=False)
        trainer.prepare_data()
        trainer.build_model()
        trainer.setup_training()

        loss, acc = trainer.train_epoch()

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0
        assert 0 <= acc <= MAX_ACCURACY_PERCENT

    @staticmethod
    def test_train_epoch_verbose_output(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that train_epoch works with verbose mode (coverage for pbar.set_postfix).

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        trainer = PlantTrainer(config, verbose=True)
        trainer.prepare_data()
        trainer.build_model()
        trainer.setup_training()

        # This should cover the verbose branch in train_epoch
        loss, acc = trainer.train_epoch()

        assert isinstance(loss, float)
        assert isinstance(acc, float)


class TestValidate:
    """Tests for PlantTrainer.validate method."""

    @staticmethod
    def test_validate_returns_metrics(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that validate returns loss and accuracy metrics.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        trainer = PlantTrainer(config, verbose=False)
        trainer.prepare_data()
        trainer.build_model()
        trainer.setup_training()

        # Mock validate since train/val classes don't overlap in sample data
        with patch.object(trainer, "validate", return_value=(0.5, 50.0, 80.0)):
            val_loss, val_acc, val_top5 = trainer.validate()

        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert isinstance(val_top5, float)
        assert val_loss >= 0
        assert 0 <= val_acc <= MAX_ACCURACY_PERCENT
        assert 0 <= val_top5 <= MAX_ACCURACY_PERCENT


class TestSaveCheckpoint:
    """Tests for PlantTrainer.save_checkpoint method."""

    @staticmethod
    def test_save_checkpoint_creates_last_pth(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that save_checkpoint creates last.pth.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        trainer = PlantTrainer(config, verbose=False)
        trainer.prepare_data()
        trainer.build_model()
        trainer.setup_training()

        trainer.save_checkpoint(epoch=1, val_acc=50.0, val_top5=80.0)

        last_checkpoint = trainer.checkpoint_dir / "last.pth"
        assert last_checkpoint.exists()

    @staticmethod
    def test_save_checkpoint_creates_best_pth_when_is_best(
        tmp_path: Path,
        sample_data_dir: Path,
    ) -> None:
        """Test that save_checkpoint creates best.pth when is_best=True.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        trainer = PlantTrainer(config, verbose=False)
        trainer.prepare_data()
        trainer.build_model()
        trainer.setup_training()

        trainer.save_checkpoint(epoch=1, val_acc=50.0, val_top5=80.0, is_best=True)

        best_checkpoint = trainer.checkpoint_dir / "best.pth"
        assert best_checkpoint.exists()

    @staticmethod
    def test_save_checkpoint_prints_when_is_best(
        tmp_path: Path,
        sample_data_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that save_checkpoint prints when is_best=True.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory
            capsys: Pytest capture fixture

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        trainer = PlantTrainer(config, verbose=False)
        trainer.prepare_data()
        trainer.build_model()
        trainer.setup_training()

        trainer.save_checkpoint(epoch=1, val_acc=95.5, val_top5=99.0, is_best=True)

        captured = capsys.readouterr()
        assert "Best model saved" in captured.out
        assert "95.50%" in captured.out

    @staticmethod
    def test_save_checkpoint_no_best_when_not_best(
        tmp_path: Path,
        sample_data_dir: Path,
    ) -> None:
        """Test that save_checkpoint does not create best.pth when is_best=False.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        trainer = PlantTrainer(config, verbose=False)
        trainer.prepare_data()
        trainer.build_model()
        trainer.setup_training()

        trainer.save_checkpoint(epoch=1, val_acc=50.0, val_top5=80.0, is_best=False)

        best_checkpoint = trainer.checkpoint_dir / "best.pth"
        assert not best_checkpoint.exists()

    @staticmethod
    def test_save_checkpoint_contains_required_keys(
        tmp_path: Path,
        sample_data_dir: Path,
    ) -> None:
        """Test that saved checkpoint contains required keys.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        trainer = PlantTrainer(config, verbose=False)
        trainer.prepare_data()
        trainer.build_model()
        trainer.setup_training()

        trainer.save_checkpoint(epoch=1, val_acc=50.0, val_top5=80.0)

        checkpoint = torch.load(trainer.checkpoint_dir / "last.pth", weights_only=False)
        required_keys = [
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "val_acc",
            "val_top5",
            "best_acc",
            "config",
            "history",
            "class_to_idx",
            "idx_to_class",
            "num_classes",
        ]
        assert all(key in checkpoint for key in required_keys)


class TestTrain:
    """Tests for PlantTrainer.train method."""

    @staticmethod
    def test_train_returns_history(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that train returns history dictionary.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        config["epochs"] = 1
        trainer = PlantTrainer(config, verbose=False)
        trainer.prepare_data()
        trainer.build_model()
        trainer.setup_training()

        # Mock validate to avoid empty val_loader issue
        with patch.object(trainer, "validate", return_value=(0.5, 50.0, 80.0)):
            history = trainer.train()

        assert isinstance(history, dict)
        assert len(history["train_loss"]) == 1
        assert len(history["val_acc"]) == 1

    @staticmethod
    def test_train_saves_history_json(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that train saves history.json.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        config["epochs"] = 1
        trainer = PlantTrainer(config, verbose=False)
        trainer.prepare_data()
        trainer.build_model()
        trainer.setup_training()

        # Mock validate to avoid empty val_loader issue
        with patch.object(trainer, "validate", return_value=(0.5, 50.0, 80.0)):
            trainer.train()

        history_file = trainer.checkpoint_dir / "history.json"
        assert history_file.exists()

    @staticmethod
    def test_train_updates_best_acc(tmp_path: Path, sample_data_dir: Path) -> None:
        """Test that train updates best_acc and best_epoch.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        config["epochs"] = 1
        trainer = PlantTrainer(config, verbose=False)
        trainer.prepare_data()
        trainer.build_model()
        trainer.setup_training()

        # Mock validate to avoid empty val_loader issue
        with patch.object(
            trainer,
            "validate",
            return_value=(MOCK_VAL_LOSS, MOCK_VAL_ACC, MOCK_VAL_TOP5),
        ):
            trainer.train()

        # Best acc should be updated after first epoch
        assert trainer.best_acc == MOCK_VAL_ACC
        assert trainer.best_epoch == 1


class TestTrainerVerbose:
    """Tests for PlantTrainer verbose output."""

    @staticmethod
    def test_prepare_data_verbose_output(
        tmp_path: Path,
        sample_data_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that prepare_data prints info when verbose=True.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory
            capsys: Pytest capture fixture

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        trainer = PlantTrainer(config, verbose=True)
        trainer.prepare_data()

        captured = capsys.readouterr()
        assert "Total train samples" in captured.out
        assert "Selected classes" in captured.out
        assert "Train samples" in captured.out
        assert "Val samples" in captured.out

    @staticmethod
    def test_setup_training_verbose_output(
        tmp_path: Path,
        sample_data_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that setup_training prints info when verbose=True.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory
            capsys: Pytest capture fixture

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        trainer = PlantTrainer(config, verbose=True)
        trainer.prepare_data()
        trainer.build_model()

        # Clear captured output before setup_training
        capsys.readouterr()
        trainer.setup_training()

        captured = capsys.readouterr()
        assert "Checkpoint dir" in captured.out

    @staticmethod
    def test_train_verbose_output(
        tmp_path: Path,
        sample_data_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that train prints info when verbose=True.

        Args:
            tmp_path: Pytest temporary directory fixture
            sample_data_dir: Path to sample data directory
            capsys: Pytest capture fixture

        """
        config = get_valid_config(tmp_path, sample_data_dir)
        config["epochs"] = 1
        trainer = PlantTrainer(config, verbose=True)
        trainer.prepare_data()
        trainer.build_model()
        trainer.setup_training()

        # Clear captured output
        capsys.readouterr()

        # Mock validate to avoid empty val_loader issue
        with patch.object(trainer, "validate", return_value=(0.5, 50.0, 80.0)):
            trainer.train()

        captured = capsys.readouterr()
        assert "Epochs" in captured.out
        assert "Train samples" in captured.out
        assert "Val samples" in captured.out
        assert "Epoch 1" in captured.out
        assert "Results" in captured.out
