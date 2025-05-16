import streamlit as st
from pathlib import Path
from typing import List

from config import Config
from models.yolo_model import YOLOModel
from utils.file_utils import FileUtils
from utils.annotation_utils import AnnotationUtils


class AutoLabelComponent:
    def __init__(self):
        self.file_utils = FileUtils()
        self.annotation_utils = AnnotationUtils()

    def render(self, class_names: List[str], image_paths: List[str]):
        if not class_names or not image_paths:
            st.warning("Define classes and upload images first.")
            return

        st.header("Auto-Labeling")

        uploaded_model = st.file_uploader("Upload YOLOv8 `.pt` model", type=["pt"])
        if uploaded_model:
            model_path = Path(Config.MODELS_DIR) / uploaded_model.name
            with open(model_path, "wb") as f:
                f.write(uploaded_model.getbuffer())
            st.success(f"Model uploaded: {uploaded_model.name}")

            try:
                with st.spinner("Loading model..."):
                    st.session_state["yolo_model"] = YOLOModel(str(model_path))
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                return

        if "yolo_model" not in st.session_state or st.session_state["yolo_model"] is None:
            st.info("Upload and load a YOLO model to enable auto-labeling.")
            return

        model = st.session_state["yolo_model"]
        conf = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Label Current Image"):
                self._autolabel_single_image(
                    image_paths[st.session_state.current_image_idx],
                    class_names, conf, model
                )
        with col2:
            if st.button("Label All Images"):
                with st.spinner("Processing..."):
                    for path in image_paths:
                        self._autolabel_single_image(path, class_names, conf, model)
                st.success("All images labeled.")

    def _autolabel_single_image(
        self,
        image_path: str,
        class_names: List[str],
        conf_threshold: float,
        model: YOLOModel
    ):
        ann_path = self.file_utils.get_annotation_path(image_path)
        preds = model.predict(image_path, conf_threshold)

        filtered = []
        for cls, x, y, w, h in preds:
            model_classes = model.get_class_names()
            if cls < len(model_classes):
                name = model_classes[cls]
                if name in class_names:
                    filtered.append((class_names.index(name), x, y, w, h))

        self.annotation_utils.write_yolo_annotation(ann_path, filtered)
        st.success(f"Labeled: {Path(image_path).name}")
