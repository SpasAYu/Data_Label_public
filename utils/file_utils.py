import os
from pathlib import Path
from typing import List, Tuple
from config import Config

class FileUtils:
    @staticmethod
    def save_uploaded_files(uploaded_files) -> Tuple[int, List[str]]:
        """Save uploaded files to server and return count and paths"""
        saved_paths = []
        for uploaded_file in uploaded_files:
            file_path = Config.UPLOADS_DIR / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_paths.append(str(file_path))
        return len(saved_paths), saved_paths

    @staticmethod
    def get_image_paths() -> List[str]:
        """Get all image paths from uploads directory"""
        image_paths = []
        for ext in Config.IMAGE_EXTENSIONS:
            image_paths.extend(Config.UPLOADS_DIR.glob(f"*{ext}"))
        return sorted([str(p) for p in image_paths])

    @staticmethod
    def get_annotation_path(image_path: str) -> str:
        """Get corresponding annotation path for an image"""
        image_path = Path(image_path)
        annotation_path = Config.ANNOTATIONS_DIR / f"{image_path.stem}.txt"
        return str(annotation_path)

    @staticmethod
    def delete_image_and_annotation(image_path: str):
        """Delete image and its annotation"""
        image_path = Path(image_path)
        if image_path.exists():
            image_path.unlink()
        
        annotation_path = Config.ANNOTATIONS_DIR / f"{image_path.stem}.txt"
        if annotation_path.exists():
            annotation_path.unlink()