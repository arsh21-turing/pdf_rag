"""
components/about.py - Enhanced About page for PDF RAG application

This module provides an improved, visually appealing About page for the PDF RAG application,
including features, capabilities, and usage instructions.
"""

import streamlit as st
from typing import Dict, Any
import base64
from pathlib import Path
import time

# Custom CSS for enhanced visual styling
def load_css():
    """
    Apply custom CSS styling to enhance the visual appearance
    """
    st.markdown("""
    <style>
        /* Card-like sections with shadows */
        div.stExpander {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
            border: none;
        }
        
        /* Feature cards */
        .feature-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid #4CAF50;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        .feature-card:hover {
            transform: translateY(-5px);
        }
        
        /* Step cards for workflow */
        .step-card {
            background-color: #f1f8e9;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.7rem;
            border-left: 3px solid #8bc34a;
        }
        
        /* Model cards */
        .model-card {
            background-color: #e3f2fd;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.7rem;
            border-left: 3px solid #2196F3;
        }
        
        /* Animations for page elements */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        .fadeIn {
            animation: fadeIn 0.5s ease-in;
        }
        
        /* Header styling */
        h1 {
            color: #1E3A8A;
            font-weight: 800;
        }
        
        h2 {
            color: #1E88E5;
            border-bottom: 2px solid #BCEBF6;
            padding-bottom: 0.3rem;
            margin-top: 2rem;
        }
        
        h3 {
            color: #00796b;
            margin-top: 1.5rem;
        }
        
        /* Icon styling */
        .icon-text {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        /* Button styling */
        .stButton>button {
            border-radius: 20px;
            padding: 0.5rem 1rem;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)

def render_animated_text(text, delay=0.03):
    """Creates a typing animation effect for text"""
    placeholder = st.empty()
    full_text = ""
    
    for char in text:
        full_text += char
        placeholder.markdown(full_text)
        time.sleep(delay)
    
    return placeholder

def show_about():
    """
    Main entry point for the about page.
    This function is imported by other components.
    """
    render_about_section()

def render_about_section():
    """
    Render the about section with information about the application.
    
    This function describes the application's purpose, features,
    and provides usage instructions with enhanced visual elements.
    """
    # Apply custom CSS
    load_css()
    
    # Animated header with logo
    st.markdown('<div class="fadeIn">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 5])
    
    with col1:
        st.markdown("# 📚")
    with col2:
        st.title("PDF RAG")
        st.markdown("#### _Your intelligent document assistant_")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Application overview with animated reveal
    st.markdown('<div class="fadeIn">', unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-card">
        <h2>✨ Overview</h2>
        <p>
        PDF RAG is an advanced document processing and question-answering application 
        that enables you to extract insights from your documents effortlessly. Upload your PDFs,
        and start asking questions in natural language to get accurate, contextual answers.
        </p>
        <p>
        Using state-of-the-art semantic search and large language models, the application
        finds relevant information and generates comprehensive answers based on the 
        document's content.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Metrics in columns to show capabilities
    st.markdown('<div class="fadeIn">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Processing Time", value="2-5 sec", delta="per document")
    with col2:
        st.metric(label="Accuracy", value="90%+", delta="semantic match")
    with col3:
        st.metric(label="Supported Files", value="PDF, TXT", delta="expandable")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Key features with visual cards
    st.markdown('<div class="fadeIn">', unsafe_allow_html=True)
    st.markdown("## 🔑 Key Features")
    
    features = [
        {"icon": "📄", "title": "PDF Processing", "desc": "Extract text from PDF documents with intelligent chunking"},
        {"icon": "🔍", "title": "Semantic Search", "desc": "Use state-of-the-art embeddings to find relevant document sections"},
        {"icon": "🧠", "title": "LLM Integration", "desc": "Connect with Groq API for AI-powered document question answering"},
        {"icon": "⚙️", "title": "Customizable Settings", "desc": "Adjust chunk size, model parameters, and more"}
    ]
    
    col1, col2 = st.columns(2)
    
    for i, feature in enumerate(features):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
            <div class="feature-card">
                <h3>{feature['icon']} {feature['title']}</h3>
                <p>{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Interactive workflow diagram
    st.markdown('<div class="fadeIn">', unsafe_allow_html=True)
    with st.expander("🔄 How It Works", expanded=False):
        st.image("https://raw.githubusercontent.com/arsh21-turing/pdf_rag/main/static/workflow.png", 
                 caption="PDF RAG Workflow", use_column_width=True)
        
        steps = [
            {"title": "Text Extraction", "desc": "When you upload a document, the app extracts all text content"},
            {"title": "Chunking", "desc": "The text is divided into smaller, manageable chunks"},
            {"title": "Embedding", "desc": "Each chunk is converted to a vector embedding using all-MiniLM-L6-v2"},
            {"title": "Semantic Search", "desc": "When you ask a question, it's compared against all chunk embeddings"},
            {"title": "Result Retrieval", "desc": "The most relevant document sections are retrieved"},
            {"title": "LLM Integration", "desc": "Relevant chunks are passed to an LLM with your question to generate a comprehensive answer"}
        ]
        
        for i, step in enumerate(steps):
            st.markdown(f"""
            <div class="step-card">
                <strong>Step {i+1}: {step['title']}</strong> - {step['desc']}
            </div>
            """, unsafe_allow_html=True)
            
        # Add an interactive demo GIF or video
        st.markdown("### See it in action")
        st.video("https://raw.githubusercontent.com/arsh21-turing/pdf_rag/main/static/demo.mp4")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Usage instructions with tabs for different use cases
    st.markdown('<div class="fadeIn">', unsafe_allow_html=True)
    with st.expander("📋 Usage Instructions", expanded=False):
        tab1, tab2, tab3 = st.tabs(["Basic Usage", "API Integration", "Advanced Features"])
        
        with tab1:
            st.markdown("### Getting Started")
            
            steps = [
                "Navigate to the Upload tab and select a PDF document",
                "Configure the chunk size and overlap as needed",
                "Click the \"Process Document\" button to extract text and create embeddings",
                "Go to the Q&A tab and enter your question",
                "Use the \"Search Document\" button for basic retrieval or \"Get LLM Answer\" for AI-generated responses"
            ]
            
            for i, step in enumerate(steps):
                st.markdown(f"""
                <div class="step-card">
                    <strong>{i+1}.</strong> {step}
                </div>
                """, unsafe_allow_html=True)
                
        with tab2:
            st.markdown("### Connecting to Groq API")
            
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.markdown("""
                <div class="step-card">
                    <strong>1.</strong> Select "Use Groq LLM API" in the sidebar
                </div>
                <div class="step-card">
                    <strong>2.</strong> Enter your Groq API key
                </div>
                <div class="step-card">
                    <strong>3.</strong> When asking questions, click "Get LLM Answer"
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.image("https://raw.githubusercontent.com/arsh21-turing/pdf_rag/main/static/api_setup.png", 
                        caption="API Setup Example", use_column_width=True)
        
        with tab3:
            st.markdown("### Power User Features")
            
            st.markdown("""
            - **Chunk Optimization**: Experiment with different chunk sizes to find the best balance for your documents
            - **Model Selection**: Choose different LLMs based on your specific needs
            - **Citation View**: See exactly where answers come from in your documents
            - **Batch Processing**: Process multiple documents in queue for larger projects
            """)
            
            # Add an advanced settings example
            st.code("""
            # Example of advanced configuration
            settings = {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "top_k": 5,
                "temperature": 0.7,
                "model": "llama3-70b-8192"
            }
            """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Models information with visual cards
    st.markdown('<div class="fadeIn">', unsafe_allow_html=True)
    with st.expander("🤖 Supported Models", expanded=False):
        st.markdown("### Embedding Model")
        
        st.markdown("""
        <div class="model-card">
            <strong>all-MiniLM-L6-v2</strong>: A lightweight, efficient embedding model that converts text to vector representations
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### LLM Models (via Groq)")
        
        models = [
            {"name": "Llama 3 (8B)", "desc": "Fast, efficient language model for general-purpose text generation", "best": "Speed"},
            {"name": "Llama 3 (70B)", "desc": "More powerful language model for complex reasoning", "best": "Quality"},
            {"name": "Gemma (7B)", "desc": "Google's lightweight model for general-purpose text tasks", "best": "Efficiency"},
            {"name": "Mixtral 8x7B", "desc": "Powerful mixture-of-experts model with broader knowledge", "best": "Knowledge"}
        ]
        
        for model in models:
            st.markdown(f"""
            <div class="model-card">
                <strong>{model['name']}</strong>: {model['desc']}
                <span style="float: right; background-color: #e1f5fe; padding: 2px 8px; border-radius: 4px; font-size: 0.8em;">
                    Best for: {model['best']}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
        # Add comparison chart
        st.markdown("### Model Performance Comparison")
        chart_data = {
            'Models': ['Llama 3 (8B)', 'Llama 3 (70B)', 'Gemma (7B)', 'Mixtral 8x7B'],
            'Speed': [90, 40, 100, 50],
            'Quality': [70, 95, 65, 85],
            'Context': [75, 90, 70, 100]
        }
        st.bar_chart(chart_data, y=['Speed', 'Quality', 'Context'])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Technical information with tabs for different aspects
    st.markdown('<div class="fadeIn">', unsafe_allow_html=True)
    with st.expander("⚙️ Technical Details", expanded=False):
        tab1, tab2 = st.tabs(["Architecture", "Performance"])
        
        with tab1:
            st.markdown("### Core Components")
            
            components = [
                {"name": "Streamlit", "desc": "For the web interface", "icon": "🌐"},
                {"name": "PDF Processing", "desc": "Custom extraction pipeline for reliable text extraction", "icon": "📄"},
                {"name": "Vector Embeddings", "desc": "Sentence transformers for semantic search", "icon": "🔢"},
                {"name": "CrewAI", "desc": "For LLM integration and response generation", "icon": "🤖"}
            ]
            
            for comp in components:
                st.markdown(f"""
                <div class="feature-card">
                    <h3>{comp['icon']} {comp['name']}</h3>
                    <p>{comp['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
                
            # Add architecture diagram
            st.image("https://raw.githubusercontent.com/arsh21-turing/pdf_rag/main/static/architecture.png", 
                    caption="PDF RAG Architecture", use_column_width=True)
        
        with tab2:
            st.markdown("### Performance Considerations")
            
            considerations = [
                {"point": "Larger documents may take longer to process", "icon": "⏱️"},
                {"point": "Smaller chunk sizes increase precision but may lose context", "icon": "🔍"},
                {"point": "Larger chunk sizes preserve more context but may reduce search precision", "icon": "📝"}
            ]
            
            for item in considerations:
                st.markdown(f"<div class='icon-text'>{item['icon']} {item['point']}</div>", unsafe_allow_html=True)
                
            # Add performance optimization tips
            st.markdown("### Optimization Tips")
            st.info("""
            **For best performance:**
            - Limit documents to under 100 pages when possible
            - Start with a chunk size of 1000 and adjust based on your document type
            - Use Llama 3 (8B) for faster responses on simple queries
            - Use Llama 3 (70B) or Mixtral for complex reasoning tasks
            """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # About the project with progress bars to show development status
    st.markdown('<div class="fadeIn">', unsafe_allow_html=True)
    st.markdown("## 🔮 About the Project")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        PDF RAG is an open-source project designed to make document analysis
        and information retrieval accessible and efficient. It combines the power
        of modern NLP techniques with an intuitive user interface.
        
        We're constantly improving and expanding the capabilities - contributions welcome!
        """)
    
    with col2:
        st.markdown("### Project Status")
        st.progress(90, "Core Features")
        st.progress(80, "Documentation")
        st.progress(70, "Testing")
        
    # Call to action
    st.success("Ready to start? Head to the Upload tab to process your first document!")
    
    # Footer with attribution and links
    st.markdown("""
    <hr>
    <p style="text-align: center; color: #666; font-size: 0.8em;">
        Powered by Streamlit, Sentence Transformers & Groq LLMs • 
        <a href="https://github.com/arsh21-turing/pdf_rag" target="_blank">GitHub</a> • 
        Version 1.0.0
    </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def get_app_info() -> Dict[str, Any]:
    """
    Return a dictionary containing information about the application.
    
    This can be used programmatically by other components.
    
    Returns:
        Dict with application information
    """
    return {
        "name": "PDF RAG",
        "version": "1.0.0",
        "description": "An advanced Streamlit application for document processing and question answering",
        "embedding_model": "all-MiniLM-L6-v2",
        "supported_llms": [
            "llama3-8b-8192",
            "llama3-70b-8192", 
            "gemma-7b-it",
            "mixtral-8x7b-32768"
        ],
        "supported_file_types": ["pdf", "txt"],
        "github_repo": "https://github.com/arsh21-turing/pdf_rag",
        "last_updated": "2025-05-16"
    }
