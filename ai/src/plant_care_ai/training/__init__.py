"""Training module for plant classification.

Copyright 2025 Plant Care Assistant
"""

from .class_selection import get_most_popular_classes
from .train import PlantTrainer

__all__ = ["PlantTrainer", "get_most_popular_classes"]
