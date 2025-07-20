import os
import numpy as np
import pandas as pd

import cv2
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


class BirdsDataset(Dataset):

    def __init__(self, images_dir, classes_csv=None, target_size=256, transform=None, device=None):
        """
        :param images_dir: string, root directory for images
        :param keypoints_csv: string, keypoints csv path
        :param target_size: int, size of a square trainable image
        :param transform: callable | optional, transforms applied to columns

        :return: dataset
        """

        default_transofrm = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        self.images_dir = images_dir
        self.classes_csv = pd.read_csv(classes_csv) if classes_csv else None
        self.target_size = target_size
        self.transform = transform if transform is not None else default_transofrm
        self.device = device

        if self.classes_csv is not None:
            self.image_files = self.classes_csv.iloc[:, 0].tolist()
        else:
            self.image_files = [file for file in os.listdir(images_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            self.image_files.sort()
            assert len(self.image_files) > 0, f"No images found in {images_dir}"


    def __len__(self):
        return len(self.image_files)
    

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.images_dir, self.image_files[idx])
        if not os.path.exists(img_name):
            raise FileNotFoundError(f"Image not found: {img_name}")
        
        image = cv2.imread(img_name)
        if image is None:
            raise ValueError(f"Failed to load image: {img_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        class_ = self.classes_csv.iloc[idx, 1].astype(float) if self.classes_csv is not None else None
        image = self.smart_resize_with_padding(
            image, self.target_size
        )
        
        if class_ is not None:
            image = self.transform(image=image)['image'] if self.transform is not None else image
            return image, class_
        else:
            image = self.transform(image=image)['image'] if self.transform is not None else image
            return image
    
    
    @staticmethod
    def smart_resize_with_padding(image, target_size):
        h, w = image.shape[:2]
        scale = min(target_size/h, target_size/w)
        new_w, new_h = int(w*scale), int(h*scale)
        resized_image = cv2.resize(image, (new_w, new_h))

        padded_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        start_x = (target_size - new_w) // 2
        start_y = (target_size - new_h) // 2

        padded_image[start_y : start_y+new_h, start_x : start_x+new_w] = resized_image

        return padded_image