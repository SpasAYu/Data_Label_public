import streamlit as st
#import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from utils.file_utils import FileUtils
from utils.annotation_utils import AnnotationUtils

class AnnotatorComponent:
    def __init__(self):
        self.file_utils = FileUtils()
        self.annotation_utils = AnnotationUtils()
        self.current_image_idx = 0
        self.annotations = []
        self.class_names = []
        self.image_paths = []

    def render_class_input(self) -> List[str]:
        """Render class name input and return list of class names"""
        st.subheader("Define Classes")
        class_input = st.text_area(
            "Enter class names (one per line)", 
            value="class1\nclass2\nclass3",
            height=150
        )
        return [name.strip() for name in class_input.split("\n") if name.strip()]

    def render_image_navigation(self):
        """Render image navigation controls"""
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
        
        with col1:
            if st.button("Previous") and self.current_image_idx > 0:
                self.current_image_idx -= 1
                st.experimental_rerun()
                
        with col2:
            if st.button("Next") and self.current_image_idx < len(self.image_paths) - 1:
                self.current_image_idx += 1
                st.experimental_rerun()
                
        with col3:
            if st.button("Delete Current"):
                self.file_utils.delete_image_and_annotation(self.image_paths[self.current_image_idx])
                self.image_paths.pop(self.current_image_idx)
                if self.current_image_idx >= len(self.image_paths):
                    self.current_image_idx = max(0, len(self.image_paths) - 1)
                st.experimental_rerun()
                
        with col4:
            st.write(f"Image {self.current_image_idx + 1} of {len(self.image_paths)}")

    def render_annotation_controls(self, image: np.ndarray):
        """Render annotation controls and handle user input"""
        if image is None:
            return
            
        height, width = image.shape[:2]
        annotation_path = self.file_utils.get_annotation_path(self.image_paths[self.current_image_idx])
        
        # Load existing annotations
        self.annotations = self.annotation_utils.read_yolo_annotation(annotation_path)
        
        # Create columns for image and controls
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Image display with annotations
            img_with_boxes = self.annotation_utils.draw_bboxes(image, self.annotations, self.class_names)
            st.image(img_with_boxes, use_column_width=True, channels="BGR")
            
        with col2:
            st.subheader("Annotation Tools")
            
            # Class selection
            selected_class = st.selectbox("Select class", options=self.class_names, key="class_select")
            
            # Bounding box creation
            if st.button("Add Box"):
                self.annotations.append((self.class_names.index(selected_class), 0.5, 0.5, 0.1, 0.1))
                self.annotation_utils.write_yolo_annotation(annotation_path, self.annotations)
                st.experimental_rerun()
                
            # Bounding box editing
            if self.annotations:
                selected_box_idx = st.selectbox(
                    "Select box to edit", 
                    options=list(range(len(self.annotations))),
                    format_func=lambda x: f"Box {x+1} ({self.class_names[self.annotations[x][0]]})"
                )
                
                box_col1, box_col2 = st.columns(2)
                with box_col1:
                    if st.button("Delete Box"):
                        self.annotations.pop(selected_box_idx)
                        self.annotation_utils.write_yolo_annotation(annotation_path, self.annotations)
                        st.experimental_rerun()
                        
                with box_col2:
                    if st.button("Update Class"):
                        new_class = st.selectbox(
                            "New class", 
                            options=self.class_names, 
                            key="update_class",
                            index=self.annotations[selected_box_idx][0]
                        )
                        ann = list(self.annotations[selected_box_idx])
                        ann[0] = self.class_names.index(new_class)
                        self.annotations[selected_box_idx] = tuple(ann)
                        self.annotation_utils.write_yolo_annotation(annotation_path, self.annotations)
                        st.experimental_rerun()
                
                # Box coordinates editing
                st.subheader("Box Coordinates")
                _, x_center, y_center, box_w, box_h = self.annotations[selected_box_idx]
                
                x_center = st.slider("X Center", 0.0, 1.0, x_center, 0.01)
                y_center = st.slider("Y Center", 0.0, 1.0, y_center, 0.01)
                box_w = st.slider("Width", 0.0, 1.0, box_w, 0.01)
                box_h = st.slider("Height", 0.0, 1.0, box_h, 0.01)
                
                if st.button("Update Coordinates"):
                    self.annotations[selected_box_idx] = (
                        self.annotations[selected_box_idx][0],
                        x_center, y_center, box_w, box_h
                    )
                    self.annotation_utils.write_yolo_annotation(annotation_path, self.annotations)
                    st.experimental_rerun()

    def render(self, class_names: List[str], image_paths: List[str]):
        """Main render method for annotation interface"""
        self.class_names = class_names
        self.image_paths = image_paths
        
        if not image_paths:
            st.warning("No images available for annotation. Please upload images first.")
            return
            
        self.render_image_navigation()
        
        # Load current image
        current_image_path = self.image_paths[self.current_image_idx]
        image = cv2.imread(current_image_path)
        
        if image is None:
            st.error(f"Failed to load image: {current_image_path}")
            return
            
        self.render_annotation_controls(image)