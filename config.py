import os
from pathlib import Path

class Config:
    # Path configurations
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    UPLOADS_DIR = DATA_DIR / "uploads"
    ANNOTATIONS_DIR = DATA_DIR / "annotations"
    MODELS_DIR = DATA_DIR / "models"
    
    # Create directories if they don't exist
    for dir_path in [UPLOADS_DIR, ANNOTATIONS_DIR, MODELS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    
    # Annotation format
    ANNOTATION_FORMAT = "yolo"  # Can be extended to other formats
    
    # Default colors for classes
    CLASS_COLORS = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
        "#00FFFF", "#FFA500", "#800080", "#008000", "#000080"
    ]
    
    @classmethod
    def get_class_color(cls, class_idx):
        return cls.CLASS_COLORS[class_idx % len(cls.CLASS_COLORS)]