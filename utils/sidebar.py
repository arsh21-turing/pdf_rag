# components/sidebar.py
import streamlit as st

def render_sidebar():
    """Render the application sidebar."""
    with st.sidebar:
        st.header("Navigation")
        
        # Navigation tabs
        tabs = {
            "upload": "📤 Upload & Validate",
            "extract": "📑 Extract Content",
            "analyze": "📊 Analyze Document",
            "export": "💾 Export Data",
            "settings": "⚙️ Settings"
        }
        
        # Only enable tabs that make sense based on the current state
        disabled_tabs = set()
        if not st.session_state.pdf_file_data:
            disabled_tabs.update(["extract", "analyze", "export"])
        elif not st.session_state.validated or not st.session_state.is_pdf_valid:
            disabled_tabs.update(["extract", "analyze", "export"])
        elif not st.session_state.processed:
            disabled_tabs.update(["analyze", "export"])
        
        # Create navigation buttons
        for tab_id, tab_name in tabs.items():
            button_disabled = tab_id in disabled_tabs
            if st.sidebar.button(
                tab_name, 
                key=f"nav_{tab_id}",
                disabled=button_disabled,
                use_container_width=True
            ):
                st.session_state.active_tab = tab_id
                st.rerun()
        
        # Show document information if available
        if st.session_state.pdf_file_data:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Current Document")
            st.sidebar.info(f"File: {st.session_state.pdf_file_name}")
            st.sidebar.info(f"Size: {st.session_state.pdf_file_size:.1f} KB")
            
            if st.session_state.validation_result:
                if st.session_state.is_pdf_valid:
                    st.sidebar.success(f"{st.session_state.validation_result['page_count']} Pages | Valid PDF")
                else:
                    st.sidebar.error("Invalid PDF")
            
            # Display statistics if available
            if st.session_state.doc_statistics:
                st.sidebar.markdown("---")
                st.sidebar.subheader("Document Statistics")
                
                stats = st.session_state.doc_statistics
                st.sidebar.info(f"Pages: {stats.get('total_pages', 0)}")
                st.sidebar.info(f"Words: {stats.get('total_words', 0):,}")
                st.sidebar.info(f"Paragraphs: {stats.get('total_paragraphs', 0)}")
                if 'total_tables' in stats:
                    st.sidebar.info(f"Tables: {stats.get('total_tables', 0)}")