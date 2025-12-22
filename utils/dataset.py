import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from glob import glob


class ImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = sorted(glob(os.path.join(root_dir, "*", "*.*")))

        self.class_names = sorted(
            [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

        self.labels = [
            self.class_to_idx[os.path.basename(os.path.dirname(p))]
            for p in self.image_paths
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    from utils.transforms import _get_train_transforms

    dataset = ImageFolder(
        root_dir="data/train", transform=_get_train_transforms(imgsz=224)
    )
    print(f"Dataset size: {len(dataset)}")
    img, label = dataset[0]
    cv2.imshow("img", img.permute(1, 2, 0).numpy())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Image shape: {img.shape}, Label: {label}")
