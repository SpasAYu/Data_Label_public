from typing import List, Dict, Tuple
import cv2
import numpy as np
from pathlib import Path
from config import Config

class AnnotationUtils:
    @staticmethod
    def read_yolo_annotation(annotation_path: str) -> List[Tuple[int, float, float, float, float]]:
        """Read YOLO format annotation file"""
        annotations = []
        if Path(annotation_path).exists():
            with open(annotation_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        annotations.append((int(class_id), x_center, y_center, width, height))
        return annotations

    @staticmethod
    def write_yolo_annotation(annotation_path: str, annotations: List[Tuple[int, float, float, float, float]]):
        """Write annotations in YOLO format"""
        with open(annotation_path, "w") as f:
            for ann in annotations:
                f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

    @staticmethod
    def draw_bboxes(image: np.ndarray, annotations: List[Tuple[int, float, float, float, float]], 
                   class_names: List[str]) -> np.ndarray:
        """Draw bounding boxes on image with class labels"""
        if image is None:
            return None
            
        h, w = image.shape[:2]
        img_with_boxes = image.copy()
        
        for class_id, x_center, y_center, box_w, box_h in annotations:
            # Convert from YOLO format to pixel coordinates
            x1 = int((x_center - box_w/2) * w)
            y1 = int((y_center - box_h/2) * h)
            x2 = int((x_center + box_w/2) * w)
            y2 = int((y_center + box_h/2) * h)
            
            # Get class color
            color = Config.get_class_color(class_id)
            bgr_color = tuple(int(color[i:i+2], 16) for i in (5, 3, 1))
            
            # Draw rectangle
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), bgr_color, 2)
            
            # Draw class label
            class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
            label = f"{class_name}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(img_with_boxes, (x1, y1 - label_height - 10), (x1 + label_width, y1), bgr_color, -1)
            cv2.putText(img_with_boxes, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img_with_boxes