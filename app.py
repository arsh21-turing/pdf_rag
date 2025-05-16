"""
app.py - Document Q&A Streamlit Application

This application uses PDF processing and embedding to provide document Q&A
capabilities, with optional LLM integration for enhanced answers.
"""

import streamlit as st
import os
import time
import tempfile
from typing import Dict, Any, List, Optional, Union
import logging
import numpy as np

# Import our modules
from crew_ai import CrewManager, StreamlitIntegration
from core.pdf_extractor import extract_text_from_pdf, get_pdf_page_image
from pdf_embedder import PDFEmbedder

# Import app components
from components.about import show_about

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")]
)
logger = logging.getLogger("app")

# Session state initialization
def init_session_state():
    """Initialize all session state variables"""
    if 'processed_document' not in st.session_state:
        st.session_state.processed_document = None
    
    if 'crew_manager' not in st.session_state:
        st.session_state.crew_manager = None
    
    if 'tab' not in st.session_state:
        st.session_state.tab = "upload"
    
    if 'document_stats' not in st.session_state:
        st.session_state.document_stats = {}
    
    if 'model_name' not in st.session_state:
        st.session_state.model_name = "llama3-8b-8192"
    
    if 'chunk_size' not in st.session_state:
        st.session_state.chunk_size = 500  # Smaller default chunk size
        
    if 'chunk_overlap' not in st.session_state:
        st.session_state.chunk_overlap = 50  # Smaller default overlap
    
    if 'current_groq_api_key' not in st.session_state:
        st.session_state.current_groq_api_key = ""
        
    if 'api_mode' not in st.session_state:
        st.session_state.api_mode = True
        
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = ""
    
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
        
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
        
    if 'pdf_embedder' not in st.session_state:
        try:
            # Always initialize the PDFEmbedder with embedding model
            st.session_state.pdf_embedder = PDFEmbedder(model_name="all-MiniLM-L6-v2", log_level=logging.INFO)
            logger.info("PDFEmbedder initialized with embedding model")
        except Exception as e:
            logger.error(f"Failed to initialize PDFEmbedder: {e}")
            st.session_state.pdf_embedder = None

# CrewManager creation (optional, only used if API key is provided)
def create_crew_manager(groq_api_key: str) -> CrewManager:
    """Create a CrewManager instance with the current settings"""
    # Get configuration from settings or use defaults
    temperature = st.session_state.get("temperature", 0.7)
    model_name = st.session_state.get("model_name", "llama3-8b-8192")
    
    # Create synthesizer config that includes the model settings
    synthesizer_config = {
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": 1500,
    }
    
    # Create the CrewManager with appropriate configuration
    manager = CrewManager(
        config={
            "groq_api_key": groq_api_key,  # Pass API key to the manager
            "synthesizer_config": synthesizer_config,
        },
        log_level=logging.INFO
    )
    
    logger.info(f"CrewManager created with model={model_name}")
    return manager

def check_and_update_crew_manager(api_key: str):
    """Update the CrewManager if needed (only when using API)"""
    if (not st.session_state.crew_manager or
        st.session_state.current_groq_api_key != api_key or
        st.session_state.current_model_name != st.session_state.model_name):
        logger.info("Creating or updating CrewManager with new API key or model")
        st.session_state.crew_manager = create_crew_manager(api_key)
        st.session_state.current_groq_api_key = api_key
        st.session_state.current_model_name = st.session_state.model_name

# Find relevant chunks for a query using embeddings
def semantic_search(query: str, top_k: int = 3) -> List[Dict]:
    """Perform semantic search on embeddings to find relevant chunks for a query"""
    if not st.session_state.pdf_embedder:
        logger.warning("PDFEmbedder not available for search")
        return []
    
    try:
        results = st.session_state.pdf_embedder.semantic_search(
            query=query,
            embeddings=st.session_state.embeddings,
            chunks=st.session_state.chunks,
            top_k=top_k
        )
        return results
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        return []

# Sidebar UI
def display_sidebar():
    """Display the application sidebar"""
    with st.sidebar:
        st.title("PDF-RAG")

        # Mode selector
        app_mode = st.radio(
            "Response Mode",
            ["Use Groq LLM API", "Basic Search Only"],
            index=0 if st.session_state.api_mode else 1
        )
        st.session_state.api_mode = app_mode == "Use Groq LLM API"

        groq_api_key = None
        if st.session_state.api_mode:
            groq_api_key = StreamlitIntegration.render_groq_api_key_input()
            if not groq_api_key:
                st.warning("⚠️ Please enter your Groq API key for LLM responses")

        st.markdown("---")
        st.subheader("Navigation")
        if st.button("📄 Upload Document", key="nav_upload"):
            st.session_state.tab = "upload"
        if st.button(
            "🔍 Ask Questions", 
            key="nav_qa",
            disabled=len(st.session_state.get("chunks", [])) == 0
        ):
            st.session_state.tab = "qa"
        if st.button("⚙️ Settings", key="nav_settings"):
            st.session_state.tab = "settings"
        if st.button("ℹ️ About", key="nav_about"):
            st.session_state.tab = "about"

        if st.session_state.document_stats:
            st.markdown("---")
            st.subheader("Document Statistics")
            stats = st.session_state.document_stats
            if "title" in stats:
                st.write(f"**Document:** {stats['title']}")
            if "pages" in stats:
                st.write(f"**Pages:** {stats['pages']}")
            if "chunks" in stats:
                st.write(f"**Chunks:** {stats['chunks']}")
            if "embedding_dim" in stats:
                st.write(f"**Embedding Dim:** {stats['embedding_dim']}")
            if "process_time" in stats:
                st.write(f"**Process Time:** {stats['process_time']:.2f}s")

        return groq_api_key

# Upload tab
def upload_tab():
    """Handle document uploading and processing"""
    st.title("📄 Upload Document")
    st.write("Upload a document to process and analyze.")

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])

    if uploaded_file:
        st.write(f"File uploaded: **{uploaded_file.name}**")
        
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input("Chunk Size", min_value=100, max_value=5000, value=500)
        with col2:
            chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=chunk_size//2, value=50)
        
        st.session_state.chunk_size = chunk_size
        st.session_state.chunk_overlap = chunk_overlap

        if st.button("Process Document"):
            start_time = time.time()
            with st.spinner("Processing document..."):
                try:
                    # Get file content
                    file_content = uploaded_file.getvalue()
                    file_name = uploaded_file.name
                    
                    # 1. Extract text using core.pdf_extractor
                    if file_name.lower().endswith('.pdf'):
                        # Use PDF extractor first
                        full_text, text_by_page = extract_text_from_pdf(file_content)
                        pages_count = len(text_by_page)
                        st.info(f"Successfully extracted text from {pages_count} pages")
                    elif file_name.lower().endswith('.txt'):
                        # Simple text file
                        full_text = file_content.decode('utf-8', errors='replace')
                        text_by_page = {"1": full_text}
                        pages_count = 1
                    else:
                        st.error("Unsupported file type")
                        return
                    
                    # Show text extraction preview
                    with st.expander("Extracted Text Preview", expanded=False):
                        st.text_area("First 2000 characters", full_text[:2000] + "...", height=200)
                        st.write(f"Total text length: {len(full_text)} characters")
                    
                    # 2. Process with PDFEmbedder to create chunks and embeddings
                    embedder = st.session_state.pdf_embedder
                    if embedder:
                        # Process the text directly
                        embedding_result = embedder.embed_text(
                            full_text, 
                            chunk_size=chunk_size, 
                            overlap=chunk_overlap
                        )
                        
                        if embedding_result.get("success", False):
                            chunks = embedding_result["chunks"]
                            embeddings = embedding_result["embeddings"]
                            metadata = embedding_result.get("metadata", {})
                            
                            # Store in session state
                            st.session_state.extracted_text = full_text
                            st.session_state.chunks = chunks
                            st.session_state.embeddings = embeddings
                            
                            # Show embedding details
                            with st.expander("Embedding Details", expanded=False):
                                st.write(f"Created {len(chunks)} chunks")
                                st.write(f"Generated {len(embeddings)} embeddings")
                                st.write(f"Embedding dimension: {embeddings.shape[1]}")
                                
                            # Store document stats
                            process_time = time.time() - start_time
                            st.session_state.document_stats = {
                                "title": file_name,
                                "pages": pages_count,
                                "chunks": len(chunks),
                                "embedding_dim": embeddings.shape[1],
                                "process_time": process_time
                            }
                            
                            # Show chunks preview
                            with st.expander("Document Chunks", expanded=False):
                                for i, chunk in enumerate(chunks[:5]):
                                    st.markdown(f"**Chunk {i+1}:**")
                                    st.text_area(f"chunk_{i}", chunk[:300] + "..." if len(chunk) > 300 else chunk, height=100)
                                
                                if len(chunks) > 5:
                                    st.info(f"Showing 5 of {len(chunks)} chunks")
                            
                            # Initialize CrewManager if in API mode
                            if st.session_state.api_mode:
                                groq_api_key = StreamlitIntegration.get_groq_api_key()
                                if groq_api_key:
                                    check_and_update_crew_manager(groq_api_key)
                            
                            st.success(f"Document processed in {process_time:.2f} seconds!")
                            
                            # Automatically redirect to question page
                            st.session_state.tab = "qa"
                            st.rerun()
                        else:
                            st.error(f"Embedding failed: {embedding_result.get('error', 'Unknown error')}")
                    else:
                        st.error("PDFEmbedder not available. Cannot create chunks and embeddings.")
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    logger.exception("Exception in document processing")
    else:
        # Show informational content when no file is uploaded
        st.info("👆 Upload a PDF or text file to get started")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### What happens after upload?
            
            When you upload a document, PDF-RAG will:
            1. Extract all text content
            2. Split the text into semantic chunks
            3. Create vector embeddings for search
            4. Enable semantic questioning
            """)
        
        with col2:
            st.markdown("""
            ### Supported file types
            
            - **PDF files** (.pdf) - Including scanned PDFs
            - **Text files** (.txt) - Plain text documents
            
            The system works best with text-based PDFs rather than scanned documents
            or images, though it can handle those as well.
            """)
        
        st.markdown("---")
        st.markdown("""
        For more information about this application, check out the 
        [About](#) section using the navigation menu in the sidebar.
        """)


# Q&A tab
def qa_tab():
    """Handle document Q&A functionality"""
    st.title("🔍 Ask Questions")
    
    if not st.session_state.chunks or len(st.session_state.chunks) == 0:
        st.warning("No document processed. Please upload and process a document first.")
        if st.button("Go to Upload"):
            st.session_state.tab = "upload"
            st.rerun()
        return
    
    # Question input
    question = st.text_input("Enter your question about the document")
    
    if question:
        col1, col2 = st.columns([1, 1])
        with col1:
            search_button = st.button("🔍 Search Document")
        with col2:
            if st.session_state.api_mode:
                answer_button = st.button("🤖 Get LLM Answer")
            else:
                answer_button = False
                st.info("Enter API key in sidebar to enable LLM answers")
        
        # Handle search
        if search_button:
            with st.spinner("Searching document..."):
                search_results = semantic_search(question, top_k=5)
                
                if search_results:
                    st.subheader("Relevant Document Sections")
                    for i, result in enumerate(search_results):
                        with st.expander(f"Section {i+1} (Score: {result['score']:.4f})", expanded=i==0):
                            st.markdown(result["chunk"])
                else:
                    st.warning("No relevant sections found")
                    
                    # Fall back to keyword search
                    st.info("Falling back to keyword search...")
                    keyword_results = []
                    for i, chunk in enumerate(st.session_state.chunks):
                        if question.lower() in chunk.lower():
                            keyword_results.append((i, chunk))
                    
                    if keyword_results:
                        st.subheader("Keyword Search Results")
                        for i, (idx, chunk) in enumerate(keyword_results[:5]):
                            with st.expander(f"Section {i+1}", expanded=i==0):
                                st.markdown(chunk)
                    else:
                        st.error("No matching content found")
        
        # Handle LLM answer
        if answer_button and st.session_state.api_mode:
            groq_api_key = StreamlitIntegration.get_groq_api_key()
            
            if not groq_api_key:
                st.error("API key not provided. Please enter your Groq API key in the sidebar.")
                return
                
            try:
                with st.spinner("Generating answer..."):
                    # Get relevant chunks for context
                    context_chunks = semantic_search(question, top_k=3)
                    context_text = "\n\n".join([chunk["chunk"] for chunk in context_chunks])
                    
                    # Create prompt for LLM
                    prompt = f"""Answer the following question based ONLY on the provided context. 
                    If you cannot answer the question from the context, say "I don't have enough information to answer that."
                    
                    CONTEXT:
                    {context_text}
                    
                    QUESTION: {question}
                    
                    ANSWER:"""
                    
                    # Check and update CrewManager if needed
                    check_and_update_crew_manager(groq_api_key)
                    
                    # Get answer from crew manager's synthesizer
                    # Initialize language model first to ensure api_type is set
                    if not hasattr(st.session_state.crew_manager.synthesizer, 'api_type'):
                        st.session_state.crew_manager.synthesizer._init_language_model()
                        
                    answer = st.session_state.crew_manager.synthesizer._generate_text(prompt)
                    
                    # Display answer
                    st.subheader("Answer")
                    st.write(answer)
                    
                    # Display sources
                    st.subheader("Sources")
                    for i, chunk in enumerate(context_chunks):
                        with st.expander(f"Source {i+1} (Score: {chunk['score']:.4f})", expanded=False):
                            st.markdown(chunk["chunk"])
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
                logger.exception("Exception in answer generation")

# Settings tab
def settings_tab():
    """Display settings options for the application"""
    st.title("⚙️ Settings")
    
    st.subheader("Document Processing")
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input(
            "Chunk Size", 
            min_value=100, 
            max_value=5000, 
            value=st.session_state.get("chunk_size", 1000),
            help="Size of text chunks in characters for document processing"
        )
    with col2:
        chunk_overlap = st.number_input(
            "Chunk Overlap", 
            min_value=0, 
            max_value=chunk_size//2, 
            value=st.session_state.get("chunk_overlap", 200),
            help="Overlap between chunks in characters to maintain context"
        )
    
    # Update session state
    st.session_state.chunk_size = chunk_size
    st.session_state.chunk_overlap = chunk_overlap
    
    # Model settings only shown in API mode
    if st.session_state.api_mode:
        st.subheader("LLM Settings")
        
        # Model selection
        model_options = {
            "llama3-8b-8192": "Llama 3 (8B)",
            "llama3-70b-8192": "Llama 3 (70B)",
            "gemma-7b-it": "Gemma (7B)",
            "mixtral-8x7b-32768": "Mixtral 8x7B",
        }
        
        selected_model = st.selectbox(
            "Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=list(model_options.keys()).index(st.session_state.model_name) if st.session_state.model_name in model_options else 0,
            help="Select the LLM to use for answering questions"
        )
        
        # Store selected model in session state
        st.session_state.model_name = selected_model
        
        # Temperature setting
        temperature = st.slider(
            "Temperature", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state.get("temperature", 0.7), 
            step=0.1,
            help="Controls randomness of the model (lower = more deterministic)"
        )
        st.session_state.temperature = temperature
    
    # Reset buttons
    with st.expander("Reset Options"):
        clear_doc = st.button("Clear Current Document")
        if clear_doc:
            st.session_state.extracted_text = ""
            st.session_state.chunks = []
            st.session_state.embeddings = None
            st.session_state.document_stats = {}
            st.success("Document cleared")
            
        reset_all = st.button("Reset All Settings")
        if reset_all:
            for key in st.session_state.keys():
                del st.session_state[key]
            st.success("All settings have been reset")
            st.rerun()

# About tab
def about_tab():
    """Show the About tab with information about the application"""
    # Use the imported about page component
    show_about()
    
    # Add any additional about content specific to this app
    with st.expander("PDF-RAG Document Q&A Capabilities"):
        st.markdown("""
        This application extends the PDF-RAG platform with specialized document question-answering capabilities:
        
        - **Semantic Search**: Find relevant document sections using semantic similarity
        - **Embedding Generation**: Create vector embeddings for advanced document understanding
        - **LLM Integration**: Optional integration with Groq LLMs for AI-powered answers
        - **Chunk Management**: Smart document chunking for better context retrieval
        
        Whether you need to quickly search through documents or get AI-generated answers to complex questions,
        this tool provides both options with a simple, intuitive interface.
        """)
        
        st.info("You can use this tool in Basic Search mode without any API keys, or enable the LLM features by providing a Groq API key.")

# Main function
def main():
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title="PDF-RAG Document Q&A",
        page_icon="📑",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Display sidebar and get API key
    groq_api_key = display_sidebar()
    
    # Check and update CrewManager if API key is available and in API mode
    if st.session_state.api_mode and groq_api_key:
        check_and_update_crew_manager(groq_api_key)
    
    # Display the current tab
    if st.session_state.tab == "upload":
        upload_tab()
    elif st.session_state.tab == "qa":
        qa_tab()
    elif st.session_state.tab == "settings":
        settings_tab()
    elif st.session_state.tab == "about":
        about_tab()
    
    # Show mode indicator at bottom of page
    st.markdown("---")
    current_mode = "LLM API Mode" if st.session_state.api_mode else "Basic Search Mode"
    st.caption(f"Running in {current_mode} | PDF-RAG © 2024")

if __name__ == "__main__":
    main()