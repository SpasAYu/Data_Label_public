from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from config import Config
from utils.file_utils import FileUtils
from utils.annotation_utils import AnnotationUtils


class AnnotatorComponent:
    def __init__(self):
        self.file_utils = FileUtils()
        self.annotation_utils = AnnotationUtils()
        self.current_image_idx: int = 0
        self.annotations: List[Tuple[int, float, float, float, float]] = []
        self.class_names: List[str] = []
        self.image_paths: List[str] = []

    def render_class_input(self) -> List[str]:
        st.subheader("Define Classes")
        class_input = st.text_area(
            "Enter class names (one per line)",
            value="class1\nclass2\nclass3",
            height=150,
        )
        return [c.strip() for c in class_input.split("\n") if c.strip()]

    def render_image_navigation(self):
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
        with col1:
            if st.button("Previous"):
                if self.current_image_idx > 0:
                    self.current_image_idx -= 1
        with col2:
            if st.button("Next"):
                if self.current_image_idx < len(self.image_paths) - 1:
                    self.current_image_idx += 1
        with col3:
            if st.button("Delete Current"):
                self.file_utils.delete_image_and_annotation(
                    self.image_paths[self.current_image_idx]
                )
                self.image_paths.pop(self.current_image_idx)
                if self.current_image_idx >= len(self.image_paths):
                    self.current_image_idx = max(0, len(self.image_paths) - 1)
        with col4:
            st.write(f"Image {self.current_image_idx + 1} / {len(self.image_paths)}")

    def render_annotation_controls(self, image: np.ndarray):
        if image is None:
            return

        h, w = image.shape[:2]
        image_path = self.image_paths[self.current_image_idx]
        ann_path = self.file_utils.get_annotation_path(image_path)

        # Загружаем аннотации только один раз
        if f"annotations_{self.current_image_idx}" not in st.session_state:
            st.session_state[f"annotations_{self.current_image_idx}"] = \
                self.annotation_utils.read_yolo_annotation(ann_path)

        annotations = st.session_state[f"annotations_{self.current_image_idx}"]

        new_class = st.selectbox("Class for new boxes", options=self.class_names)
        mode = st.radio("Annotation mode", ["Draw new", "Edit existing"])
        drawing_mode = "rect" if mode == "Draw new" else "transform"

        shapes = []
        for idx, (cls, xc, yc, bw, bh) in enumerate(annotations):
            left = (xc - bw / 2) * w
            top = (yc - bh / 2) * h
            shapes.append({
                "type": "rect",
                "left": left,
                "top": top,
                "width": bw * w,
                "height": bh * h,
                "strokeWidth": 2,
                "stroke": Config.get_class_color(cls),
                "fill": "rgba(0,0,0,0)",
                "metadata": {"class_id": cls},
                "id": str(idx),
            })

        # Convert BGR→RGB for display
        bg_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=2,
            stroke_color=Config.get_class_color(self.class_names.index(new_class)),
            background_image=bg_pil,
            update_streamlit=True,
            height=h,
            width=w,
            drawing_mode=drawing_mode,
            initial_drawing={"objects": shapes},
            key=f"canvas_{self.current_image_idx}"
        )

        # Сохраняем аннотации по кнопке
        if st.button("Сохранить аннотации"):
            new_anns: List[Tuple[int, float, float, float, float]] = []

            if canvas_result and canvas_result.json_data and "objects" in canvas_result.json_data:
                for obj in canvas_result.json_data["objects"]:
                    left, top = obj["left"], obj["top"]
                    pw, ph = obj["width"], obj["height"]

                    xc_new = (left + pw / 2) / w
                    yc_new = (top + ph / 2) / h
                    bw_new = pw / w
                    bh_new = ph / h
                    cls_new = obj.get("metadata", {}).get(
                        "class_id", self.class_names.index(new_class)
                    )

                    new_anns.append((cls_new, xc_new, yc_new, bw_new, bh_new))

            self.annotation_utils.write_yolo_annotation(ann_path, new_anns)
            st.session_state[f"annotations_{self.current_image_idx}"] = new_anns
            st.success("Аннотации сохранены!")

        # Предпросмотр с подписями
        preview = self.annotation_utils.draw_bboxes(image, annotations, self.class_names)
        st.image(preview, use_column_width=True, channels="BGR", output_format="PNG")

    def render(self, class_names: List[str], image_paths: List[str]):
        self.class_names = class_names
        self.image_paths = image_paths

        if not image_paths:
            st.warning("Upload images first.")
            return

        if self.current_image_idx >= len(self.image_paths):
            self.current_image_idx = 0

        self.render_image_navigation()

        image_path = self.image_paths[self.current_image_idx]
        try:
            pil_image = Image.open(image_path)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            st.error(f"❌ Ошибка загрузки изображения: {image_path}\n{e}")
            return

        self.render_annotation_controls(image)
