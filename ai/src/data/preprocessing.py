import torchvision.transforms as T

from PIL import Image

class PlantNetPreprocessor:
    """
    Preprocessor - with optional augmentation - pipeline.
    """

    def __init__(self, img_size=224, normalize=True, augm_strength=0.0):
        """Initialize preprocessor.
        
        Args:
            img_size: target image size (sqr)
            normalize: whether to apply ImageNet norm
            augm_strength: augmentation intensity [0.0 to 1.0]
        """

        self.img_size = img_size
        self.normalize = normalize
        self.augm_strength = max(0.0, min(1.0, augm_strength))  # [0;1]
        
        # normalization stats of ImageNet
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def get_full_transform(self):
        """
        Training pipeline (with augmentation)
        """

        # resize, do a random crop
        transforms = [
            T.Resize(256),
            T.RandomCrop(self.img_size),
        ]
        
        # apply augmentation
        if self.augm_strength > 0:
            transforms.append(T.RandomHorizontalFlip(p=0.5))
            
            # random rotations (up to 30 deg)
            if self.augm_strength >= 0.3:
                rotation_deg = int(30*self.augm_strength)
                transforms.append(T.RandomRotation(rotation_deg))
            
            # random brightness, contrast etc
            if self.augm_strength >= 0.5:
                transforms.append(T.ColorJitter(
                    brightness=0.3*self.augm_strength,
                    contrast=0.3*self.augm_strength,
                    saturation=0.3*self.augm_strength,
                    hue=0.1*self.augm_strength
                ))
            
            # random translation, scalling
            if self.augm_strength >= 0.7:
                transforms.append(T.RandomAffine(
                    degrees=0, # done before
                    translate=(0.1*self.augm_strength, 0.1*self.augm_strength),
                    scale=(1-0.1*self.augm_strength, 1+0.1*self.augm_strength)
                ))
        
        transforms.append(T.ToTensor())
        
        if self.normalize:
            transforms.append(T.Normalize(mean=self.mean, std=self.std))
        
        return T.Compose(transforms) # (1, C, H, W)
    
    def get_interference_transform(self):
        """
        Interference pipeline (excluding the augmentation)
        """

        transforms = [
            T.Resize(256),
            T.RandomCrop(self.img_size), # central...? hmm
            T.ToTensor(),
        ]

        # .....test and vals skip augmentations 
        
        if self.normalize:
            transforms.append(T.Normalize(mean=self.mean, std=self.std))
        
        return T.Compose(transforms)
    
    def get_transform(self, train=True):
        """
        train=True: with augmentation
        train=False: no augmentation (for interference, test or validation mode)
        """
        return self.get_full_transform() if train else self.get_interference_transform()
    

def get_training_pipeline(img_size=224, augm_strength=0.5):
    preprocessor = PlantNetPreprocessor(
        img_size=img_size,
        normalize=True,
        augm_strength=augm_strength
    )
    return preprocessor.get_train_transform()


def get_inference_pipeline(img_size=224):
    preprocessor = PlantNetPreprocessor(
        img_size=img_size,
        normalize=True
    )
    return preprocessor.get_interference_transform()

def preprocess_single_image(image, img_size=224, normalize=True):
    """
    Preprocess single image for inference
    """

    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    
    transform = get_inference_pipeline(img_size)
    tensor = transform(image)
    
    return tensor.unsqueeze(0)