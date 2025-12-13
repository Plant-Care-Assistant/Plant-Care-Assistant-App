import torchvision.transforms as T

class PlantNetPreprocessor:
    def __init__(self, img_size=224, normalize=True, augm_strength=0.0):
        self.img_size = img_size
        self.normalize = normalize
        self.augm_strength = max(0.0, min(1.0, augm_strength))  # [0;1]
        
        # normalization stats of ImageNet stats
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def get_train_transform(self):
        transforms = [T.Resize(256), T.RandomCrop(self.img_size)]
        
        # apply augmentation
        if self.augm_strength > 0:
            transforms.append(T.RandomHorizontalFlip(p=0.5))
            
            if self.augm_strength >= 0.3:
                rotation_deg = int(30*self.augm_strength)
                transforms.append(T.RandomRotation(rotation_deg))
            
            if self.augm_strength >= 0.5:
                transforms.append(T.ColorJitter(
                    brightness=0.3*self.augm_strength,
                    contrast=0.3*self.augm_strength,
                    saturation=0.3*self.augm_strength,
                    hue=0.1*self.augm_strength
                ))
            
            if self.augm_strength >= 0.7:
                transforms.append(T.RandomAffine(
                    degrees=0,
                    translate=(0.1*self.augm_strength, 0.1*self.augm_strength),
                    scale=(1-0.1*self.augm_strength, 1+0.1*self.augm_strength)
                ))
        
        transforms.append(T.ToTensor())
        
        # normalization
        if self.normalize:
            transforms.append(T.Normalize(mean=self.mean, std=self.std))
        
        return T.Compose(transforms)
    
    def get_val_transform(self):
        transforms = [T.Resize(256), T.CenterCrop(self.img_size), T.ToTensor()]

        # .....test and vals skip augmentations 
        
        if self.normalize:
            transforms.append(T.Normalize(mean=self.mean, std=self.std))
        
        return T.Compose(transforms)
    
    def get_transform(self, train=True):
        return self.get_train_transform() if train else self.get_val_transform()