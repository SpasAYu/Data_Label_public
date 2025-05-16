from typing import List, Tuple, Optional
#import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class YOLOModel:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.class_names = self.model.names if hasattr(self.model, 'names') else []

    def predict(self, image_path: str, conf_threshold: float = 0.5) -> List[Tuple[int, float, float, float, float]]:
        """Run prediction on an image and return YOLO format annotations"""
        results = self.model(image_path, conf=conf_threshold)
        
        annotations = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                img_height, img_width = result.orig_shape
                for box in result.boxes:
                    class_id = int(box.cls)
                    xywh = box.xywhn.cpu().numpy()[0]  # Normalized xywh
                    annotations.append((class_id, xywh[0], xywh[1], xywh[2], xywh[3]))
        
        return annotations

    def get_class_names(self) -> List[str]:
        """Get class names that model was trained on"""
        return self.class_names