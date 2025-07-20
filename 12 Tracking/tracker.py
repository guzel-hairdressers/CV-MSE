import numpy as np
import torch
import cv2

from tracking import extract_detections, draw_detections 

class Tracker:
    def __init__(self, return_images=True, lookup_tail_size=80,
                 labels=None, confidence=.3, min_iou=.5, model_path='ssd_model.pt'):
        self.return_images = return_images
        self.lookup_tail_size = lookup_tail_size
        self.confidence = confidence
        self.min_iou = min_iou
        self.labels = labels
        
        self.frame_index = 0
        self.tracklet_count = 0
        self.last_detected = {}
        self.detection_history = []
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()

    @staticmethod
    def _iou(b1, b2):
        b1, b2 = np.asarray(b1, dtype=float), np.asarray(b2, dtype=float)
        inter_lr = np.minimum(b1[:, None, 2:], b2[None, :, 2:])
        inter_ul = np.maximum(b1[:, None, :2], b2[None, :, :2])
        inter_wh = np.clip(inter_lr - inter_ul, 0, None)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]
        area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
        area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
        union_area = area1[:, None] + area2[None, :] - inter_area
        return inter_area / np.clip(union_area, 1e-6, None)

    def _new_id(self):
        self.tracklet_count += 1
        return self.tracklet_count - 1

    def init_tracklet(self, frame):
        dets = extract_detections(self.model, [frame], confidence=self.confidence, labels=self.labels)
        dets = dets[0] if len(dets) > 0 else dets.reshape(0, 5)
        if dets.size == 0:
            return dets
        for i in range(len(dets)):
            dets[i, 0] = self._new_id()
        return dets.astype(int)

    @property
    def prev_detections(self):
        shots = []
        for tid, last_f_idx in self.last_detected.items():
            if self.frame_index - last_f_idx > self.lookup_tail_size:
                continue
            frame_dets = self.detection_history[last_f_idx]
            match = frame_dets[frame_dets[:, 0] == tid]
            if len(match) > 0:
                shots.append(match[0])
        return np.stack(shots) if shots else np.empty((0, 5), dtype=int)

    def bind_tracklet(self, detections):
        if detections.size == 0:
            return detections
        prev = self.prev_detections
        if prev.size == 0:
            for i in range(len(detections)):
                detections[i, 0] = self._new_id()
            return detections.astype(int)
        ious = self._iou(detections[:, 1:], prev[:, 1:])
        assigned_prev = np.full(len(prev), False)
        for i in range(len(detections)):
            if len(ious[i]) > 0:
                best_idx = np.argmax(ious[i])
                best_iou = ious[i, best_idx]
                if best_iou >= self.min_iou and not assigned_prev[best_idx]:
                    detections[i, 0] = prev[best_idx, 0]
                    assigned_prev[best_idx] = True
                else:
                    detections[i, 0] = self._new_id()
            else:
                detections[i, 0] = self._new_id()
        return detections.astype(int)

    def save_detections(self, detections):
        for detection in detections:
            track_id = int(detection[0])
            self.last_detected[track_id] = self.frame_index

    def update_frame(self, frame: np.ndarray):
        if self.frame_index == 0:
            detections = self.init_tracklet(frame)
        else:
            detections_raw = extract_detections(self.model, [frame],
                                                confidence=self.confidence,
                                                labels=self.labels)
            detections_raw = detections_raw[0] if len(detections_raw) > 0 else detections_raw.reshape(0, 5)
            detections = self.bind_tracklet(detections_raw)
        self.save_detections(detections)
        self.detection_history.append(detections)
        self.frame_index += 1

        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections


class CorrelationTracker(Tracker):

    def __init__(self, detection_rate=5, display_result=True, **kwargs):
        super().__init__(**kwargs)
        self.detection_rate = detection_rate
        self.display_result=display_result
        self.prev_frame = None

    def correlate_tracklet(self, frame: np.ndarray) -> np.ndarray:
        detections = self.detection_history[-1].copy()
        
        prev_frame_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for i, detection in enumerate(detections):
            track_id, x1, y1, x2, y2 = detection.astype(int)
            w, h = x2 - x1, y2 - y1

            template = prev_frame_gray[y1:y2, x1:x2]
            
            search_x1 = max(0, x1 - w // 2)
            search_y1 = max(0, y1 - h // 2)
            search_x2 = min(current_frame_gray.shape[1], x2 + w // 2)
            search_y2 = min(current_frame_gray.shape[0], y2 + h // 2)
            search_area = current_frame_gray[search_y1:search_y2, search_x1:search_x2]

            if template.size == 0 or search_area.size == 0:
                continue

            search_area = np.asarray(search_area, dtype=np.uint8)
            template = np.asarray(template, dtype=np.uint8)
            heatmap = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
            
            _, max_val, _, max_loc = cv2.minMaxLoc(heatmap)
            
            if max_val < 0.5:
                continue

            new_x1 = search_x1 + max_loc[0]
            new_y1 = search_y1 + max_loc[1]
            
            detections[i] = [track_id, new_x1, new_y1, new_x1 + w, new_y1 + h]

        return detections

    def update_frame(self, frame: np.ndarray):
        if self.frame_index == 0:
            detections = self.init_tracklet(frame)
        elif self.frame_index % self.detection_rate == 0:
            detections_raw = extract_detections(self.model, [frame],
                                                confidence=self.confidence,
                                                labels=self.labels)
            detections_raw = detections_raw[0] if len(detections_raw) > 0 else detections_raw.reshape(0, 5)
            detections = self.bind_tracklet(detections_raw)
        else:
            detections = self.correlate_tracklet(frame)
        self.save_detections(detections)
        self.detection_history.append(detections)
        self.prev_frame = frame.copy()
        self.frame_index += 1

        if self.return_images:
            return draw_detections(frame, detections, display=self.display_result)
        else:
            return detections
