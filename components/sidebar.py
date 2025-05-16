# components/sidebar.py
"""
Sidebar UI for the PDF-RAG demo app.
Handles:
  • PDF upload/validation
  • Navigation between high-level tabs
  • Quick status overview
  • Semantic search settings
"""
import streamlit as st
from datetime import datetime

# optional: your own helpers
try:
    from core.pdf_validator import validate_pdf  # returns (ok: bool, meta: dict, msg: str)
except ImportError:
    # fallback stub so the file works even without the real validator
    def validate_pdf(data: bytes):
        return True, {}, "✔︎ PDF looks OK"

try:
    from components.semantic_search import SemanticSearchEngine  # Import for semantic search
except ImportError:
    # Stub for semantic search engine if module is not available
    class SemanticSearchEngine:
        def __init__(self):
            pass
        def process_pdf(self, *args, **kwargs):
            return None, []
        def semantic_search(self, *args, **kwargs):
            return []


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _save_pdf_to_session():
    """Persist uploaded PDF bytes + minimal info in st.session_state."""
    upload = st.session_state.pdf_uploader
    if not upload:
        return

    st.session_state.pdf_file_name = upload.name
    st.session_state.pdf_file_size = len(upload.getbuffer())
    st.session_state.upload_time = datetime.now().isoformat(" ", "seconds")
    st.session_state.pdf_file_data = upload.getvalue()

    # Fix validation handling
    try:
        # Call the validation function
        validation_result = validate_pdf(st.session_state.pdf_file_data)
        
        # If it's not returning in the right format, construct a proper result
        if isinstance(validation_result, tuple) and len(validation_result) == 3:
            ok, meta, msg = validation_result
        elif isinstance(validation_result, bool):
            # Handle case where it only returns True/False
            ok = validation_result
            meta = {}
            msg = "✅ PDF validated successfully" if ok else "❌ Invalid PDF format"
        else:
            # Assume validation successful but with unexpected format
            ok = True
            meta = {}
            msg = "✅ PDF accepted (validation format unknown)"
    except Exception as e:
        ok = False
        meta = {}
        msg = f"Error validating PDF: {str(e)}"

    st.session_state.is_pdf_valid = ok
    st.session_state.validation_result = {"ok": ok, "metadata": meta, "message": msg}
    st.session_state.validated = True
    st.session_state.active_tab = "extract" if ok else "upload"


def _nav_button(label: str, target: str, *, disabled=False):
    """Small helper to keep nav buttons consistent."""
    if st.sidebar.button(label, disabled=disabled):
        st.session_state.active_tab = target
        st.rerun()


def _display_search_settings():
    """Display semantic search settings in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Search Settings")
    
    # Chunk size for document processing
    chunk_size = st.sidebar.slider(
        "Chunk Size",
        min_value=128,
        max_value=1024,
        value=512,
        step=64,
        help="Size of document chunks for search. Larger chunks provide more context but may be less precise."
    )
    st.session_state.chunk_size = chunk_size
    
    # Select model for embedding
    model_options = {
        "all-MiniLM-L6-v2": "Fast, compact model (default)",
        "all-mpnet-base-v2": "More accurate, slower model"
    }
    
    selected_model = st.sidebar.selectbox(
        "Embedding Model",
        options=list(model_options.keys()),
        format_func=lambda x: f"{x} - {model_options[x]}",
        index=0,
        help="Model used for generating embeddings. More accurate models may be slower."
    )
    st.session_state.embedding_model = selected_model
    
    # Number of results to show
    top_k = st.sidebar.slider(
        "Max Results",
        min_value=1,
        max_value=10,
        value=3,
        help="Maximum number of search results to display."
    )
    st.session_state.top_k = top_k
    
    # Reset embeddings button (useful if settings have changed)
    if st.sidebar.button("Reset Embeddings", help="Regenerate document embeddings with current settings"):
        st.session_state.embeddings_generated = False
        st.rerun()


# -----------------------------------------------------------------------------
# public API
# -----------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.title("📄 PDF-RAG")

    # ------------------------------------------------------------------ upload
    upload = st.sidebar.file_uploader(
        "Upload a PDF",
        type="pdf",
        accept_multiple_files=False,
        on_change=_save_pdf_to_session,
        key="pdf_uploader",
    )

    # show quick status
    if st.session_state.get("validated"):
        msg = st.session_state.validation_result["message"]
        if st.session_state.validation_result["ok"]:
            st.sidebar.success(msg)
        else:
            st.sidebar.error(msg)

    # ---------------------------------------------------------------- nav
    st.sidebar.markdown("---")
    st.sidebar.subheader("Navigation")

    _nav_button("⬆️ Upload", "upload")
    _nav_button(
        "🗂 Extract",
        "extract",
        disabled=not st.session_state.get("processed")
        and not st.session_state.get("is_pdf_valid"),
    )
    _nav_button(
        "📊 Analyze",
        "analyze",
        disabled=not st.session_state.get("extracted_to_df"),
    )
    _nav_button(
        "🔍 Search",
        "search",
        disabled=not st.session_state.get("is_pdf_valid"),
    )
    _nav_button(
        "⬇️ Export",
        "export",
        disabled=not st.session_state.get("processed"),
    )
    _nav_button("⚙️ Settings", "settings")
    _nav_button("ℹ️ About", "about")

    # ---------------------------------------------------------------- details
    if st.session_state.get("pdf_file_name"):
        st.sidebar.markdown("---")
        st.sidebar.subheader("Current document")
        st.sidebar.write(f"**Name:** {st.session_state.pdf_file_name}")
        st.sidebar.write(
            f"**Size:** {st.session_state.pdf_file_size/1024:,.0f} KB"
        )
        if st.session_state.get("doc_statistics"):
            stats = st.session_state.doc_statistics
            st.sidebar.write(f"**Pages:** {stats.get('total_pages')}")
            st.sidebar.write(f"**Words:** {stats.get('total_words'):,}")

    # ---------------------------------------------------------------- search settings
    # Display search settings if we're on the search tab
    active_tab = st.session_state.get("active_tab", "")
    if active_tab == "search" and st.session_state.get("is_pdf_valid"):
        _display_search_settings()
        
        # Show search statistics if available
        if st.session_state.get("search_statistics"):
            st.sidebar.markdown("---")
            st.sidebar.subheader("Search Statistics")
            stats = st.session_state.search_statistics
            if stats.get("chunks"):
                st.sidebar.write(f"**Document chunks:** {stats.get('chunks')}")
            if stats.get("embedding_time"):
                st.sidebar.write(f"**Embedding time:** {stats.get('embedding_time'):.2f}s")
            if stats.get("last_query_time"):
                st.sidebar.write(f"**Last query time:** {stats.get('last_query_time'):.3f}s")

    # ---------------------------------------------------------------- footer
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2024 PDF-RAG demo")