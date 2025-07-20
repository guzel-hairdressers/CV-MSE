import os
import numpy as np
import pandas as pd

import cv2
import torch
from torch import nn, optim
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm 

from PIL import Image, ImageDraw, ImageFont


def plot_metrics(train_losses=[], val_losses=[], train_accuracies=[], val_accuracies=[]):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Train Accuracy')
    if val_accuracies:
        plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_model(model, train_loader, val_loader=None, epochs=50, lr=1e-3, lr_ratio=1e-2, model_path=None, train_graphs=True, device=None):
    epochs = epochs

    params_to_update = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(params_to_update, lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=lr_ratio**(1/epochs)
    )

    patience = int(epochs/2)
    best_val_accuracy = 0.0
    epochs_no_improve = 0

    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []

    for epoch in tqdm(range(epochs), position=0, desc='Epoch'):
        model.train()
        train_loss = train_correct = train_seen = 0
        val_loss = val_correct = val_seen = 0

        with tqdm(train_loader, position=1, leave=False, desc='Train') as pbar:
            for i, data in enumerate(pbar):
                inputs, targets = data[0].to(device), data[1].to(device).float()

                optimizer.zero_grad()

                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                # scheduler.step()

                train_loss += loss.item() * inputs.size(0)
                train_correct += torch.sum((outputs>0).float()==targets).item()
                train_seen += inputs.size(0)
                pbar.set_postfix(
                    loss=f'{train_loss/train_seen:.4f}',
                    accuracy=f'{train_correct/train_seen:.4f}',
                    lr=f'{optimizer.param_groups[0]["lr"]:.6f}'
                )

        train_loss /= train_seen
        train_accuracy = train_correct / train_seen
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        scheduler.step()

        if val_loader is None:
            if (train_accuracy > best_val_accuracy):
                tqdm.write(f'Train accuracy improved ({best_val_accuracy:.4f} --> {train_accuracy:.4f}). Saving model...')
                torch.save(model, model_path)
                best_val_accuracy = train_accuracy
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                tqdm.write(f'Train accuracy did not improve. Counter: {epochs_no_improve}/{patience}')
            tqdm.write(f'Epoch {epoch+1} - train accuracy: {train_accuracy:.4f} - train loss: {train_loss:.4f}')
            continue

        model.eval()
        with torch.no_grad():
            for data in val_loader:
                inputs, targets = data[0].to(device), data[1].to(device).float()
                outputs = model(inputs).squeeze(1)
                val_loss += criterion(outputs, targets).item() * inputs.size(0)
                val_correct += torch.sum((outputs>0).float()==targets).item()
                val_seen += inputs.size(0)
        
        val_loss /= val_seen
        val_accuracy = val_correct / val_seen
        
        tqdm.write(f'Epoch {epoch+1} - train accuracy: {train_accuracy:.4f} - train loss: {train_loss:.4f} - val accuracy: {val_accuracy:.4f} - val loss: {val_loss:.4f}')
        
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if (val_accuracy > best_val_accuracy):
            tqdm.write(f'Validation accuracy improved ({best_val_accuracy:.4f} --> {val_accuracy:.4f}). Saving model...')
            torch.save(model, model_path)
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            tqdm.write(f'Validation accuracy did not improve. Counter: {epochs_no_improve}/{patience}')

        if epochs_no_improve >= patience:
            tqdm.write('Early stopping triggered.')
            break


    if os.path.exists(model_path): model = torch.load(model_path, map_location=device)
    else: torch.save(model, model_path)
    print('Finished Training. Loaded best model.')

    if train_graphs:
        plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)


def evaluate_model(model, test_loader, device=None):
    model.eval()
    test_correct = test_seen = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data[0].to(device), data[1].to(device).float()
            outputs = model(inputs).squeeze(1)
            test_correct += torch.sum((outputs>0).float()==targets).item()
            test_seen += inputs.size(0)

    test_accuracy = test_correct / test_seen
    print(f'Test accuracy: {test_accuracy:.6f}')


class CarsClassifierDataset(Dataset):
    def __init__(self, images, classes, transform=None):
        
        if transform is None:
            self.transform = A.Compose([
                A.Normalize(mean=[.45], std=[0.27], max_pixel_value=1.0),
                ToTensorV2()
            ])
        else: self.transform = transform
        self.images = images
        self.classes = classes if classes is not None else None

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.transform(image=self.images[idx])['image'] if self. transform else self.images[idx]
        return (image, self.classes[idx]) if self.classes is not None else (image, None)



def get_detections(detection_model, dictionary_of_images, confidence_threshold=.65, heatmap_blur=0.0):

    img_mean, img_std = .45, .27
    image_shape = (220, 370)
    train_image_shape = (40, 100)

    device = next(detection_model.parameters()).device

    image_batch, shapes, keys = [], [], []
    for key, image_t in dictionary_of_images.items():
        keys.append(key)
        shapes.append(image_t.shape[:2])
        image = np.zeros(image_shape, dtype=np.float32)
        image[:image_t.shape[0], :image_t.shape[1]] = image_t[:, :, 0].astype(np.float32)

        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        image = (image/255.0 - img_mean) / img_std
        image_batch.append(image)
    
    image_batch = torch.cat(image_batch, dim=0).to(device)

    detection_model.eval()
    with torch.no_grad():
        logits = detection_model(image_batch)
        probs = torch.sigmoid(logits)
        if heatmap_blur > 0:
            from torchvision.transforms.functional import gaussian_blur
            probs = gaussian_blur(probs, kernel_size=5, sigma=heatmap_blur)
    stride_y, stride_x = np.array(image_shape) / logits.shape[2:]
    detections = {}

    for idx, key in enumerate(keys):
        heatmap = probs[idx, 0].cpu().numpy()
        ys, xs = np.indices(heatmap.shape)
        ys_strided, xs_strided = (ys + .5) * stride_y, (xs + .5) * stride_x
        bboxes = np.stack((xs_strided - train_image_shape[1]/2, ys_strided - train_image_shape[0]/2, \
                  xs_strided + train_image_shape[1]/2, ys_strided + train_image_shape[0]/2), axis=-1).round()
        
        mask_slices = (slice(int(train_image_shape[0]/2/stride_y), int((shapes[idx][0] - train_image_shape[0]/2)/stride_y)), \
                         slice(int(train_image_shape[1]/2/stride_x), int((shapes[idx][1] - train_image_shape[1]/2)/stride_x)))
        detection = np.hstack((bboxes[mask_slices].reshape(-1, 4), heatmap[mask_slices].reshape(-1, 1)))
        detections[key] = detection[detection[:, -1] >= confidence_threshold]
    return detections


def visualize_bboxes(image_np: np.ndarray, bounding_boxes, gt_bboxes=None):
    color_pred: str = 'red'
    color_gt: str = 'lime'
    width_pred: int = 2
    width_gt: int = 2
    show_scores: bool = False
    font_path: str | None = None
    font_size: int = 12

    if image_np.ndim == 2:
        img_pil = Image.fromarray(image_np).convert('RGB')
    else:
        img_pil = Image.fromarray(image_np)

    draw = ImageDraw.Draw(img_pil)

    font = None
    if show_scores:
        try:
            font = ImageFont.truetype(font_path or "arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
    for bbox in bounding_boxes:
        x1, y1, x2, y2, _ = bbox
        draw.rectangle([x1, y1, x2, y2],
                       outline=color_pred,
                       width=width_pred)

        if show_scores and score is not None:
            txt = f'{score:.2f}'
            draw.text((x1 + 2, y1 + 2), txt,
                      fill=color_pred, font=font)

    if gt_bboxes is not None:
        for x1, y1, x2, y2 in gt_bboxes:
            draw.rectangle([x1, y1, x2, y2],
                           outline=color_gt,
                           width=width_gt)
    return img_pil


def iou(bboxes_1, bboxes_2):
    bboxes_1, bboxes_2 = np.asarray(bboxes_1), np.asarray(bboxes_2) 
    lower_right_intersection = np.minimum(bboxes_1[:, None, 2:], bboxes_2[None, :, 2:])                         # n, m, 2
    upper_left_intersection = np.maximum(bboxes_1[:, None, :2], bboxes_2[None, :, :2])                          # n, m, 2
    intersection_diagonal = np.clip(lower_right_intersection - upper_left_intersection, 0, None)                # n, m, 2
    intersection_area = np.prod(intersection_diagonal, axis=-1)                                                 # n, m
    area_bboxes_1 = np.prod(bboxes_1[:, 2:] - bboxes_1[:, :2], axis=-1)                                         # n
    area_bboxes_2 = np.prod(bboxes_2[:, 2:] - bboxes_2[:, :2], axis=-1)                                         # m
    union_area = area_bboxes_1[:, None] + area_bboxes_2[None, :] - intersection_area                            # n, m
    return intersection_area / union_area 


def auc(detections, ground_truths, iou_threshold=.5):
    true_positive_confidences, all_confidences = [], []
    total_gt = sum(len(v) for v in ground_truths.values())

    for key, detection in detections.items():
        boxes_pred, confidences = detection[:, :4], detection[:, 4]
        gts = ground_truths.get(key, np.empty((0, 4), np.float32))
        if boxes_pred.size == 0:
            continue
        if gts.size == 0:
            all_confidences.extend(confidences)
            continue

        ious = iou(boxes_pred, gts)
        used_gt = np.zeros(len(gts), bool)

        for idx in confidences.argsort()[::-1]:
            best_gt = np.argmax(ious[idx])
            if (ious[idx, best_gt] >= iou_threshold) and (not used_gt[best_gt]):
                true_positive_confidences.append(confidences[idx])
                used_gt[best_gt] = True
        all_confidences.extend(confidences)

    if not all_confidences:
        return np.array([[1., 0.]], dtype=np.float32), 0.0

    all_confidences = np.asarray(all_confidences, dtype=np.float32)
    tp_flags = np.zeros_like(all_confidences, dtype=np.int8)


    if true_positive_confidences:
        tp_flags[np.in1d(all_confidences, true_positive_confidences,
                         assume_unique=False)] = 1

    order = all_confidences.argsort()[::-1]
    tp_flags = tp_flags[order]

    tp_cum = np.cumsum(tp_flags)
    fp_cum = np.cumsum(1 - tp_flags)

    recall    = np.concatenate(([0.], tp_cum / total_gt))
    precision = np.concatenate(([1.], tp_cum / (tp_cum + fp_cum + 1e-9)))

    auc_ = np.sum((recall[1:] - recall[:-1]) *
                  (precision[1:] + precision[:-1]) / 2)

    precision_recall_sorted = np.stack([precision, recall], axis=1)
    return precision_recall_sorted, float(auc_)
    

def nms(detections, iou_threshold=.5):

    filtered_detections = {}
    for key, detection in detections.items():
        bounding_boxes, confidences = detection[:, :-1], detection[:, -1]
        sorting_idxs = confidences.argsort()[::-1]
        bounding_boxes, confidences = bounding_boxes[sorting_idxs], confidences[sorting_idxs]
        
        ious = iou(bounding_boxes, bounding_boxes)

        is_removed = np.zeros(len(detection), dtype=bool)
        for i in range(len(detection)):
            if is_removed[i]: continue
            is_removed[(ious[i] >= iou_threshold) & (ious[i] != 1)] = True
        filtered_detections[key] = np.hstack((bounding_boxes[~is_removed], confidences[~is_removed, None]))
    
    return filtered_detections