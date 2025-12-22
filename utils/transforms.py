import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
import numpy as np
import cv2

def _get_train_transforms(imgsz=512):
    train_aug = A.Compose([
        A.LongestMaxSize(max_size=imgsz),
        A.PadIfNeeded(
            min_height=imgsz,
            min_width=imgsz,
            border_mode=0,      
            value=0            
        ),

        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
            p=0.5
        ),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])
    return train_aug

def _get_test_transforms(imgsz=512):
    test_aug = A.Compose(
        [
            A.LongestMaxSize(max_size=imgsz),
            A.PadIfNeeded(
                min_height=imgsz,
                min_width=imgsz,
                border_mode=0,
                value=0          
            ),
            ToTensorV2()
        ]
    )
    return test_aug
