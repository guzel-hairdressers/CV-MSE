import os
import numpy as np
import pandas as pd

import cv2
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import glob
import random
from tqdm.notebook import tqdm 


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, train_recall_rare=None, val_recall_rare=None):
    
    def to_cpu_numpy(data):
        if torch.is_tensor(data[0]): return [d.cpu().numpy() for d in data]
        return data
    
    train_losses = to_cpu_numpy(train_losses)
    val_losses = to_cpu_numpy(val_losses)
    train_accuracies = to_cpu_numpy(train_accuracies)
    val_accuracies = to_cpu_numpy(val_accuracies)

    plot_recall_data = train_recall_rare is not None and val_recall_rare is not None
    if plot_recall_data:
        train_recall_rare = to_cpu_numpy(train_recall_rare)
        val_recall_rare = to_cpu_numpy(val_recall_rare)

    num_subplots = 3 if plot_recall_data else 2
    plt.figure(figsize=(6 * num_subplots, 5))

    plt.subplot(1, num_subplots, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.subplot(1, num_subplots, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Overall Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    if plot_recall_data:
        plt.subplot(1, num_subplots, 3)
        plt.plot(epochs, train_recall_rare, 'b-', label='Train Recall (Rare)')
        plt.plot(epochs, val_recall_rare, 'r-', label='Validation Recall (Rare)')
        plt.title('Recall on Rare Classes')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

    plt.tight_layout()
    plt.show()


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, features, labels):
        distances = torch.cdist(features, features, p=2)
        yi_eq_yj = labels.unsqueeze(1).eq(labels.unsqueeze(0))

        pos_loss = (distances**2 * yi_eq_yj).sum() / (yi_eq_yj.sum().clamp(min=1))

        neg_pairs = torch.clamp(self.margin - distances, min=0.0) ** 2
        neg_loss = (neg_pairs * ~yi_eq_yj).sum() / ((~yi_eq_yj).sum().clamp(min=1))

        return 0.5 * (pos_loss + neg_loss)



class RareTrafficSignsDataset(Dataset):
    def __init__(self, classes_json, images_dir=None, synthetic_data_dir=None, test_classes_csv=None, target_size=48, synthetic_p=0.0, transform=None):
        
        self.pretransform = A.Compose([
                A.LongestMaxSize(max_size=target_size),
                A.PadIfNeeded(min_height=target_size, min_width=target_size, border_mode=cv2.BORDER_CONSTANT, value=0),
        ])
        if transform is None:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = transform
        
        self.classes = pd.read_json(classes_json, orient='index')
        self.name_to_id = self.classes['id'].to_dict()

        self.synthetic_p = synthetic_p
        if synthetic_data_dir is None: self.synthetic_p = 0.0
        elif images_dir is None: self.synthetic_p = 1.0

        self.real_paths = []
        self.synthetic_paths = []

        if images_dir:
            for class_name in os.listdir(images_dir):
                class_path = os.path.join(images_dir, class_name)
                if os.path.isdir(class_path):
                    for img_file in os.listdir(class_path):
                        self.real_paths.append((os.path.join(class_path, img_file), class_name))

        if synthetic_data_dir:
            generated_files = glob.glob(os.path.join(synthetic_data_dir, '**', '*.png'), recursive=True)
            for file_path in generated_files:
                class_name = os.path.basename(os.path.dirname(file_path))
                self.synthetic_paths.append((file_path, class_name))
        
        if test_classes_csv is not None:
            df = pd.read_csv(test_classes_csv)
            csv_base_dir = os.path.dirname(test_classes_csv)
            for _, row in df.iterrows():
                full_path = './tests/smalltest/' + row['filename']
                self.real_paths.append((full_path, row['class']))

        if not self.real_paths:
            raise ValueError('No images found. Check the provided paths')

    def __len__(self):
        return max(len(self.real_paths), len(self.synthetic_paths))

    def __getitem__(self, idx):
        image_files_pool = self.synthetic_paths if random.random() < self.synthetic_p else self.real_paths
        img_path, class_name = image_files_pool[idx % len(image_files_pool)]
        
        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__(np.random.randint(len(image_files_pool)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        class_id = self.name_to_id.get(class_name, -1)
        image = self.pretransform(image=image)['image']
        image = self.transform(image=image)['image']

        return image, class_id



class PasteSign(A.DualTransform):
    """
    Paste a random sign (RGB) onto the background image using its alpha/binary mask.
    Additional targets expected:
        sign  : H_s × W_s × 3  uint8
        sign_mask : H_s × W_s  uint8 / bool
    """

    def __init__(self, scale_range=(0.9, 1), p:float=1.0):
        super().__init__(always_apply=False, p=p)
        self.scale_range = scale_range

    @property
    def targets_as_params(self):
        return ['sign', 'sign_mask']
    
    def get_params_dependent_on_data(self, params, data):
        if 'rows' in params and 'cols' in params:
            bg_h, bg_w = params['rows'], params['cols']
        elif 'image' in data:
            bg_h, bg_w = data['image'].shape[:2]
        else:
            bg_h = bg_w = max(sign_h, sign_w) * 4
        crop_size = min(bg_h, bg_w)
        
        sign = data['sign']
        sign_mask = data['sign_mask']

        sign_h, sign_w = sign.shape[:2]
        scale_factor = crop_size / max(sign_h, sign_w)
        
        h = int(sign_h * scale_factor)
        w = int(sign_w * scale_factor)

        paste_y0 = (crop_size - h) // 2
        paste_x0 = (crop_size - w) // 2

        return {
            'h': h, 'w': w,
            'crop_size': crop_size,
            'paste_y0': paste_y0, 'paste_x0': paste_x0,
            'sign': sign, 'sign_mask': sign_mask
        }
    
    def apply(self, bg, crop_size, h, w, paste_y0, paste_x0, sign=None, sign_mask=None, **params):
        if sign is None: return bg

        bg_h, bg_w = bg.shape[:2]
        crop_y_start = (bg_h - crop_size) // 2
        crop_x_start = (bg_w - crop_size) // 2
        canvas = bg[crop_y_start : crop_y_start + crop_size, crop_x_start : crop_x_start + crop_size].copy()

        sign_r = cv2.resize(sign, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_r = cv2.resize(sign_mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        roi = canvas[paste_y0 : paste_y0 + h, paste_x0 : paste_x0 + w]
        roi_h, roi_w = roi.shape[:2]
        if roi_h == 0 or roi_w == 0: return canvas

        final_sign_r = sign_r[:roi_h, :roi_w]
        final_mask_r = mask_r[:roi_h, :roi_w]

        roi[final_mask_r] = final_sign_r[final_mask_r]
        canvas[paste_y0 : paste_y0 + h, paste_x0 : paste_x0 + w] = roi

        return canvas


    def apply_to_mask(self, mask, crop_size, h, w, paste_y0, paste_x0, sign=None, sign_mask=None, **params):
        if sign_mask is None: return mask

        bg_h, bg_w = mask.shape[:2]
        crop_y_start = (bg_h - crop_size) // 2
        crop_x_start = (bg_w - crop_size) // 2
        canvas_mask = mask[crop_y_start : crop_y_start + crop_size, crop_x_start : crop_x_start + crop_size].copy()

        mask_r = cv2.resize(sign_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        roi_mask = canvas_mask[paste_y0 : paste_y0 + h, paste_x0 : paste_x0 + w]
        roi_h, roi_w = roi_mask.shape[:2]
        if roi_h == 0 or roi_w == 0: return canvas_mask

        final_mask_r = mask_r[:roi_h, :roi_w]
        
        canvas_mask[paste_y0 : paste_y0 + h, paste_x0 : paste_x0 + w] = np.maximum(roi_mask, final_mask_r)
        
        return canvas_mask


def generate_synthetic_data():

    NUM_SAMPLES_TO_GENERATE = 50000
    TARGET_SIZE = 48
    OUTPUT_DIR = './tests/generated_images'
    ICONS_DIR = './tests/icons'
    BACKGROUND_DIR = './tests/background_images'
    CLASSES_JSON = './tests/classes.json'

    synthetic_pipeline = A.Compose([
        PasteSign(scale_range=(0.9, 1.0)),
        A.Resize(height=TARGET_SIZE, width=TARGET_SIZE),
        A.Rotate(limit=10, p=0.7),
        A.MotionBlur(blur_limit=(3, 7), p=1),
        A.ColorJitter(brightness=(.5, 1.2), contrast=.3, saturation=(.0, 1.0), hue=.05, p=.8),
        A.RandomBrightnessContrast(p=0.5),
    ])

    print("Loading file paths...")
    sign_image_paths = [os.path.join(ICONS_DIR, f) for f in os.listdir(ICONS_DIR)]
    background_image_paths = [os.path.join(BACKGROUND_DIR, f) for f in os.listdir(BACKGROUND_DIR)]
    classes_df = pd.read_json(CLASSES_JSON, orient='index')
    class_names = classes_df.index.tolist()

    for class_name in class_names:
        os.makedirs(os.path.join(OUTPUT_DIR, class_name), exist_ok=True)

    print(f"Generation of {NUM_SAMPLES_TO_GENERATE} synthetic images...")
    for i in tqdm(range(NUM_SAMPLES_TO_GENERATE)):
        bg_path = np.random.choice(background_image_paths)
        sign_path = np.random.choice(sign_image_paths)
        
        class_name = os.path.splitext(os.path.basename(sign_path))[0]
        if class_name not in class_names:
            continue

        bg_image = cv2.imread(bg_path)
        icon_image = cv2.imread(sign_path, cv2.IMREAD_UNCHANGED)

        if bg_image is None or icon_image is None:
            print(f"Corrupt file skipped: {bg_path} or {sign_path}")
            continue

        overlay = {
            'sign': cv2.cvtColor(icon_image[:, :, :3], cv2.COLOR_BGR2RGB),
            'sign_mask': (icon_image[:, :, 3] > 127).astype(np.uint8) if icon_image.shape[2] == 4 else \
                         cv2.threshold(cv2.cvtColor(icon_image, cv2.COLOR_BGR2GRAY), 8, 1, cv2.THRESH_BINARY)[1]
        }
        
        generated_data = synthetic_pipeline(
            image=cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB),
            sign=overlay['sign'],
            sign_mask=overlay['sign_mask'],
            mask=np.zeros(bg_image.shape[:2], dtype=np.uint8)
        )
        final_image_rgb = generated_data['image']

        output_path = os.path.join(OUTPUT_DIR, class_name, f'synth_{i:06d}.png')
        final_image_bgr = cv2.cvtColor(final_image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, final_image_bgr)

    print("Generation Complete!")