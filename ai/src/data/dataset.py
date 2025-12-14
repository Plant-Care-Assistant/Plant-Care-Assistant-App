from PIL import Image
from torch.utils.data import Dataset

from pathlib import Path
import json

class PlantNetDataset(Dataset):
    """
    Custom sub-dataset for PlantNet data.
    
    Loads images from a structured directory:
        data_dir/images/{split}/{species_id}/*.jpg
    """

    def __init__(self, data_dir, split='train', transform=None):
        """Initialize dataset.
        
        Args:
            data_dir: root dir with 'images' dir
            split: dataset split ('train', 'val', or 'test') -- 'images' subdirectory
            transform: image transformations to apply
        """

        self.split = split
        self.transform = transform

        # setup paths (main and img- one)
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images" / split

        # connect plants images (paths) with their labels (parents directories)
        self.paths = self._load_paths()

        # class (species) mappings
        self.classes = sorted(list(set(label for _, label in self.paths)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        print(f"Loaded {len(self.paths)} samples, {len(self.classes)} classes")

    def _load_paths(self):
        """
        Find all .jpg images in the split directory
        """
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