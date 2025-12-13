from torch.utils.data import DataLoader, random_split
from .dataset import PlantNetDataset


class PlantNetDataLoader:
    def __init__(self, data_dir, batch_size=32, train_transform=None, val_transform=None):
        self.batch_size = batch_size
        
        self.train_dataset = PlantNetDataset(data_dir, 'train', train_transform)
        self.val_dataset = PlantNetDataset(data_dir, 'val', val_transform)
        self.test_dataset = PlantNetDataset(data_dir, 'test', val_transform)
        
        self.num_classes = len(self.train_dataset.classes)
    
    def get_train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def get_val_loader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def get_test_loader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size)
    