from typing import List, Optional
from pathlib import Path
import subprocess
import requests
import shutil
import os
from config import Config

class ModelUtils:
    @staticmethod
    def download_yolo_model(model_name: str) -> Optional[str]:
        """Download a YOLOv8 model from Ultralytics if not already exists"""
        model_path = Config.MODELS_DIR / f"{model_name}.pt"
        
        if not model_path.exists():
            try:
                # Using ultralytics hub (this is a simplified version)
                url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}.pt"
                response = requests.get(url, stream=True)
                
                if response.status_code == 200:
                    with open(model_path, "wb") as f:
                        shutil.copyfileobj(response.raw, f)
                    return str(model_path)
                else:
                    return None
            except Exception as e:
                print(f"Error downloading model: {e}")
                return None
        return str(model_path)

    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available YOLO models in models directory"""
        models = []
        for file in Config.MODELS_DIR.glob("*.pt"):
            models.append(file.stem)
        return models

    @staticmethod
    def check_model_classes(model_path: str, target_classes: List[str]) -> bool:
        """Check if model classes match target classes (simplified)"""
        # In a real implementation, we would load the model and check its class names
        # This is a placeholder implementation
        return True