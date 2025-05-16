# utils/file_utils.py
import os
import io
import base64
import tempfile
import streamlit as st
from datetime import datetime
from typing import Union, Dict, Any

def save_uploaded_file_to_session(uploaded_file) -> bool:
    """Save uploaded file data to session state."""
    if uploaded_file is not None:
        # Store file information
        st.session_state.pdf_file_name = uploaded_file.name
        st.session_state.pdf_file_data = uploaded_file.getvalue()
        st.session_state.pdf_file_size = uploaded_file.size / 1024  # KB
        st.session_state.upload_time = datetime.now().isoformat()
        
        # Reset processing flags
        st.session_state.validated = False
        st.session_state.processed = False
        st.session_state.extracted_to_df = False
        
        # Reset preview page
        st.session_state.preview_page = 0
        
        return True
    return False

def save_validation_results(validation_result: Dict[str, Any]) -> None:
    """Save PDF validation results to session state."""
    st.session_state.validation_result = validation_result
    st.session_state.is_pdf_valid = validation_result["valid"]
    st.session_state.validated = True
    
    # Reset subsequent processing flags
    st.session_state.processed = False
    st.session_state.extracted_to_df = False

def get_file_download_link(data: Union[str, bytes], filename: str, label: str = "Download", mime: str = "text/plain") -> str:
    """Generate a download link for file data."""
    if isinstance(data, str):
        data = data.encode()
    
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:{mime};base64,{b64}" download="{filename}">{label}</a>'
    return href

def load_custom_css() -> None:
    """Load custom CSS styles."""
    css = """
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .uploadedFile {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #f9f9f9;
    }
    .stButton > button {
        width: 100%;
    }
    .results-area {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .validation-success {
        background-color: #d1e7dd;
        color: #0a3622;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .validation-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .validation-error {
        background-color: #f8d7da;
        color: #842029;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .metadata-box {
        background-color: #e9ecef;
        padding: 15px;
        border-radius: 5px;
        margin-top: 15px;
    }
</style>
"""
    st.markdown(css, unsafe_allow_html=True)