import streamlit as st
from typing import List, Optional
from pathlib import Path
from config import Config
from models.yolo_model import YOLOModel
from utils.file_utils import FileUtils
from utils.annotation_utils import AnnotationUtils
from utils.model_utils import ModelUtils

class AutoLabelComponent:
    def __init__(self):
        self.file_utils = FileUtils()
        self.annotation_utils = AnnotationUtils()
        self.model_utils = ModelUtils()
        self.yolo_model = None

    def render_model_selection(self, available_models: List[str]) -> Optional[str]:
        """Render model selection interface"""
        st.subheader("Model Selection")
        
        model_option = st.radio(
            "Choose model option",
            ["Use existing model", "Upload custom model"]
        )
        
        if model_option == "Use existing model":
            selected_model = st.selectbox(
                "Select YOLOv8 model",
                options=available_models
            )
            return selected_model
        else:
            uploaded_model = st.file_uploader(
                "Upload YOLOv8 model (.pt file)",
                type=["pt"]
            )
            if uploaded_model:
                model_path = Path(Config.MODELS_DIR) / uploaded_model.name
                with open(model_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                return uploaded_model.name
        return None

    def render_autolabel_controls(self, class_names: List[str], image_paths: List[str]):
        """Render auto-labeling controls"""
        st.subheader("Auto-Labeling Settings")
        
        available_models = self.model_utils.get_available_models()
        if not available_models:
            st.warning("No models available. Please upload a YOLOv8 model first.")
            return
            
        selected_model = self.render_model_selection(available_models)
        if not selected_model:
            return
            
        model_path = str(Path(Config.MODELS_DIR) / f"{selected_model}.pt")
        
        # Check if model classes match project classes
        if not self.model_utils.check_model_classes(model_path, class_names):
            st.error("Model classes don't match project classes. Please use a compatible model.")
            return
            
        # Confidence threshold
        conf_threshold = st.slider(
            "Confidence threshold",
            0.0, 1.0, 0.5, 0.01
        )
        
        # Load model button
        if st.button("Load Model"):
            with st.spinner("Loading model..."):
                self.yolo_model = YOLOModel(model_path)
            st.success("Model loaded successfully!")
            
        if self.yolo_model:
            st.subheader("Auto-Label Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Label Current Image"):
                    self._autolabel_single_image(
                        image_paths[st.session_state.get('current_image_idx', 0)],
                        class_names,
                        conf_threshold
                    )
                    
            with col2:
                if st.button("Label All Images"):
                    with st.spinner("Processing all images..."):
                        for img_path in image_paths:
                            self._autolabel_single_image(
                                img_path,
                                class_names,
                                conf_threshold
                            )
                    st.success("All images processed!")

    def _autolabel_single_image(self, image_path: str, class_names: List[str], conf_threshold: float):
        """Auto-label a single image"""
        if not self.yolo_model:
            st.warning("Model not loaded. Please load the model first.")
            return
            
        annotation_path = self.file_utils.get_annotation_path(image_path)
        
        # Run prediction
        predictions = self.yolo_model.predict(image_path, conf_threshold)
        
        # Filter predictions to only include classes that exist in our project
        filtered_predictions = []
        model_class_names = self.yolo_model.get_class_names()
        
        for pred in predictions:
            class_id = pred[0]
            if class_id < len(model_class_names):
                class_name = model_class_names[class_id]
                if class_name in class_names:
                    new_class_id = class_names.index(class_name)
                    filtered_predictions.append((new_class_id, *pred[1:]))
        
        # Save annotations
        self.annotation_utils.write_yolo_annotation(annotation_path, filtered_predictions)
        st.success(f"Auto-labeled {Path(image_path).name}")

    def render(self, class_names: List[str], image_paths: List[str]):
        """Main render method for auto-labeling interface"""
        if not class_names:
            st.warning("Please define classes first.")
            return
            
        if not image_paths:
            st.warning("No images available for auto-labeling. Please upload images first.")
            return
            
        self.render_autolabel_controls(class_names, image_paths)