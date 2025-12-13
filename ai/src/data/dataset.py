from PIL import Image
from torch.utils.data import Dataset

from pathlib import Path
import json

class PlantNetDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.split = split
        self.transform = transform

        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images" / split

        self.paths = self._load_paths()

        self.classes = sorted(list(set(label for _, label in self.paths)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        print(f"Loaded {len(self.paths)} samples, {len(self.classes)} classes")

    def _load_paths(self):
        paths = []

        for species_dir in self.images_dir.iterdir():
            if species_dir.is_dir():
                species_id = species_dir.name

                for img_path in species_dir.glob('*.jpg'):
                    paths.append((img_path, species_id))

        return paths
    
    def __getitem__(self, idx):
        img_path, species_id = self.paths[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, self.class_to_idx[species_id]

    def __len__(self):
        return len(self.paths)