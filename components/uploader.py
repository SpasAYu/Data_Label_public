import streamlit as st
from typing import List, Tuple
from utils.file_utils import FileUtils

class UploaderComponent:
    def __init__(self):
        self.file_utils = FileUtils()

    def render(self) -> Tuple[bool, List[str]]:
        """Render file uploader and return (uploaded status, image paths)"""
        st.header("Upload Dataset")
        
        uploaded_files = st.file_uploader(
            "Upload images for annotation",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            with st.spinner("Saving uploaded files..."):
                count, saved_paths = self.file_utils.save_uploaded_files(uploaded_files)
            st.success(f"Saved {count} files to server")
            return True, saved_paths
        
        existing_images = self.file_utils.get_image_paths()
        if existing_images:
            st.info(f"Found {len(existing_images)} existing images in dataset")
            return False, existing_images
        
        return False, []