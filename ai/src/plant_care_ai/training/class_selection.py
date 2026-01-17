"""Utility function to select most popular classes from PlantNet-300K.

Copyright 2025 Plant Care Assistant
"""

import sys
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict

from plant_care_ai.data.dataset import PlantNetDataset
from plant_care_ai.data.preprocessing import get_inference_pipeline

def get_most_popular_classes(
    data_dir: str, 
    top_k: int = 100, 
    split: str = "train"
) -> Tuple[List[str], Dict[str, int]]:
    """
    Identifies the most frequent classes in the dataset.
    
    Returns:
        Tuple containing:
        - List of the top_k class IDs (sorted by popularity)
        - Dictionary of {class_id: count} for all classes in the split
    """
    
    transform = get_inference_pipeline(224)
    dataset = PlantNetDataset(data_dir, split=split, transform=transform)
    
    class_counts = Counter()
    for _, species_id in dataset.paths:
        class_counts[species_id] += 1
    
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    top_classes = [class_id for class_id, _ in sorted_classes[:top_k]]
    
    return top_classes, dict(class_counts)