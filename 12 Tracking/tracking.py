import os
import numpy as np
import torch
import cv2

from skimage.transform import resize
import matplotlib.pyplot as plt



@torch.no_grad()
def extract_detections(model, frames, confidence=0.3, labels=None):
    
    device = next(model.parameters()).device
    model.eval()

    labels_match = labels
    
    frames_tensor = torch.from_numpy(np.asarray(frames))\
        .permute(0, 3, 1, 2).float() / 255.0
    frames_tensor = frames_tensor.to(device)
    
    predictions = model(frames_tensor)
    
    detections = []
    for prediction in predictions:
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy() 
        labels = prediction['labels'].cpu().numpy()
        
        keep = scores >= confidence
        boxes = boxes[keep]
        scores = scores[keep] 
        labels = labels[keep]
        if labels_match is not None:
            labels = np.asarray(labels_match)[labels]
        
        if len(boxes) == 0:
            return np.array([]).reshape(0, 5)
        
        detections.append(np.column_stack([labels, boxes]))
    return np.asarray(detections)

    
def draw_detections(frame, detections, font_scale=.5, font_thickness=2, display=True):
    """
    Draw bounding boxes over `frame`.

    Parameters
    ----------
    frame : np.ndarray
    detections : np.ndarray(N, 5)
    """
    
    frame_with_boxes = frame.copy()
    
    for detection in detections:
        label_id, x1, y1, x2, y2 = detection

        x1, y1, x2, y2 = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))
        
        color = (0, 255, 0)
        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, font_thickness)
        cv2.putText(frame_with_boxes, str(label_id), (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
    
    if display:
        plt.imshow(frame_with_boxes)
        plt.axis('off')
        plt.show()
    return frame_with_boxes


def save_video(frames, output_folder, video_file, fps=10):

    if not frames:
        print("No frames")
        return

    os.makedirs(output_folder, exist_ok=True)
    height, width, _ = frames[0].shape
    output_path = os.path.join(output_folder, video_file)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        video_writer.write(frame[:,:,::-1])
    video_writer.release()
    print(f"Video saved to {os.path.join(output_folder, video_file)}")



def motp_mota(predicted_bboxes, gt_bboxes, iou_threshold=.5):

    def iou(bboxes_1, bboxes_2) -> np.ndarray:
        bboxes_1, bboxes_2 = np.asarray(bboxes_1), np.asarray(bboxes_2) 
        lower_right_intersection = np.minimum(bboxes_1[:, None, 2:], bboxes_2[None, :, 2:])
        upper_left_intersection = np.maximum(bboxes_1[:, None, :2], bboxes_2[None, :, :2])
        intersection_diagonal = np.clip(lower_right_intersection - upper_left_intersection, 0, None)
        intersection_area = np.prod(intersection_diagonal, axis=-1)
        area_bboxes_1 = np.prod(bboxes_1[:, 2:] - bboxes_1[:, :2], axis=-1)
        area_bboxes_2 = np.prod(bboxes_2[:, 2:] - bboxes_2[:, :2], axis=-1)
        union_area = area_bboxes_1[:, None] + area_bboxes_2[None, :] - intersection_area
        return intersection_area / np.clip(union_area, 1e-6, None)
    
    fp_count = 0
    fn_count = 0
    id_switch_count = 0
    gt_total = 0
    
    total_iou = 0.0
    match_count = 0
    
    prev_matches = {}
    predicted_bboxes = [np.asarray(x) for x in predicted_bboxes]
    gt_bboxes = [np.asarray(x) for x in gt_bboxes]

    for gt_frame, hyp_frame in zip(gt_bboxes, predicted_bboxes):
        gt_total += len(gt_frame)
        
        if gt_frame.size == 0 or hyp_frame.size == 0:
            fp_count += len(hyp_frame)
            fn_count += len(gt_frame)
            continue

        iou_matrix = iou(gt_frame[:, 1:], hyp_frame[:, 1:])
        current_matches = {}
        matched_gt_indices = set()
        matched_hyp_indices = set()

        while iou_matrix.max() > iou_threshold:
            gt_idx, hyp_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            
            total_iou += iou_matrix[gt_idx, hyp_idx]
            match_count += 1
            
            gt_id = int(gt_frame[gt_idx, 0])
            hyp_id = int(hyp_frame[hyp_idx, 0])
            
            current_matches[gt_id] = hyp_id
            matched_gt_indices.add(gt_idx)
            matched_hyp_indices.add(hyp_idx)

            if gt_id in prev_matches and prev_matches[gt_id] != hyp_id:
                id_switch_count += 1
            
            iou_matrix[gt_idx, :] = -1
            iou_matrix[:, hyp_idx] = -1
        
        fp_count += len(hyp_frame) - len(matched_hyp_indices)
        fn_count += len(gt_frame) - len(matched_gt_indices)
        prev_matches = current_matches

    motp_score = total_iou / match_count if match_count > 0 else 1.0
    
    if gt_total == 0:
        mota_score = 1.0
    else:
        mota_score = 1.0 - (fp_count + fn_count + id_switch_count) / gt_total
    
    return motp_score, mota_score