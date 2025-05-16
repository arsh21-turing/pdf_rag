# components/upload_tab.py
import streamlit as st
import pandas as pd
from core.pdf_validator import validate_pdf
from utils.file_utils import save_uploaded_file_to_session, save_validation_results
from utils.visualization import create_metadata_table

def render_upload_tab():
    """Render the upload tab content."""
    st.header("📤 Upload your PDF document")

    # Show upload instructions
    with st.expander("How to upload a PDF", expanded=not st.session_state.get("validated")):
        st.markdown("""
        1. Use the file uploader in the sidebar to select your PDF file
        2. The system will validate your document automatically
        3. Once validated, you can proceed to the Extract tab
        """)
    
    # Show validation status
    if st.session_state.get("validated"):
        # Checking validation_result instead of looking for the 'valid' key
        if st.session_state.get("validation_result", {}).get("ok", False):
            st.success(f"✅ PDF validated successfully: {st.session_state.get('pdf_file_name')}")
            st.balloons()
            
            # Show a preview if available
            if st.session_state.get("pdf_file_data"):
                st.subheader("Document Preview")
                # Preview code here...
        else:
            msg = st.session_state.get("validation_result", {}).get("message", "Unknown validation error")
            st.error(f"❌ PDF validation failed: {msg}")
            st.warning("Please upload a different PDF document to continue")
    else:
        st.info("👈 Please upload a PDF document using the sidebar")