import streamlit as st
from components.uploader import UploaderComponent
from components.annotator import AnnotatorComponent
from components.autolabel import AutoLabelComponent
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Fix for ScriptRunContext warning
if not get_script_run_ctx():
    import os
    os.environ['STREAMLIT_BARE_MODE'] = 'true'

def main():
    st.set_page_config(
        page_title="Dataset Annotation Tool",
        page_icon=":pencil2:",
        layout="wide"
    )
    
    st.title("Dataset Annotation Tool")
    st.markdown("Annotate images for YOLOv8 training")
    
    # Initialize components
    uploader = UploaderComponent()
    annotator = AnnotatorComponent()
    autolabel = AutoLabelComponent()
    
    # Initialize session state with default values
    session_defaults = {
        'current_image_idx': 0,
        'class_names': ["class1", "class2", "class3"],
        'image_paths': []
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # File upload section
    with st.expander("Upload Images", expanded=True):
        has_new_uploads, new_image_paths = uploader.render()
        if has_new_uploads:
            st.session_state.image_paths = new_image_paths
            st.session_state.current_image_idx = 0
    
    # Class definition section
    with st.expander("Class Definition", expanded=True):
        new_class_names = annotator.render_class_input()
        if new_class_names:
            st.session_state.class_names = new_class_names
        else:
            st.warning("Please enter at least one class name")
    
    # Main content
    if not st.session_state.class_names:
        st.error("❌ No classes defined. Please define classes first.")
        return
        
    if not st.session_state.image_paths:
        st.error("❌ No images uploaded. Please upload images first.")
        return
    
    # Main tabs
    tab1, tab2 = st.tabs(["Manual Annotation", "Auto-Labeling"])
    
    with tab1:
        annotator.render(
            st.session_state.class_names,
            st.session_state.image_paths
        )
    
    with tab2:
        autolabel.render(
            st.session_state.class_names,
            st.session_state.image_paths
        )

if __name__ == "__main__":
    main()