"""
crew_ai.py - FIXED VERSION
This module provides a flexible CrewManager framework that coordinates
specialized AI agents for document processing, embedding, retrieval,
and synthesis tasks.  The agents work together in a pipeline to process
documents, answer questions, and generate insights.
"""

import os
import time
import logging
import tempfile
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np

# Import streamlit (try-except to allow non-streamlit usage)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Initialize session state for Groq API key if not already set
if STREAMLIT_AVAILABLE:
    if 'groq_api_key' not in st.session_state:
        st.session_state.groq_api_key = ""
    if 'groq_key_is_set' not in st.session_state:
        st.session_state.groq_key_is_set = False

class StreamlitIntegration:
    """
    Streamlit integration utilities for Crew AI.
    
    This class provides helper methods for integrating Crew AI with Streamlit applications,
    handling UI components, session state management, and rendering.
    """
    
    @staticmethod
    def render_groq_api_key_input():
        """
        Render a sidebar field for the Groq API Key that securely stores it in session state.
        
        The API key is stored in st.session_state.groq_api_key and a boolean flag
        st.session_state.groq_key_is_set indicates whether a key has been provided.
        
        Returns:
            str: The current Groq API key
        """
        if not STREAMLIT_AVAILABLE:
            logger.warning("Streamlit is not available, cannot render Groq API key input")
            return None
        
        with st.sidebar:
            st.subheader("API Configuration")
            
            # Display different UI based on whether the key is set
            if not st.session_state.groq_key_is_set or st.session_state.groq_api_key == "":
                # Show password input field when key is not set
                groq_key = st.text_input(
                    "Groq API Key",
                    type="password",
                    help="Enter your Groq API key to use the Groq LLM API",
                    key="groq_api_key_input"
                )
                
                # Store the API key when provided
                if groq_key:
                    st.session_state.groq_api_key = groq_key
                    st.session_state.groq_key_is_set = True
                    # Force a rerun to update the UI
                    st.rerun()
            else:
                # Display a message and option to reset when key is already set
                st.success("✅ Groq API Key is set")
                if st.button("Reset Groq API Key", key="reset_groq_key"):
                    st.session_state.groq_api_key = ""
                    st.session_state.groq_key_is_set = False
                    st.rerun()
            
            # Return the current key
            return st.session_state.groq_api_key
    
    @staticmethod
    def get_groq_api_key():
        """
        Get the Groq API key from session state.
        
        Returns:
            str: The current Groq API key or None if not set
        """
        if not STREAMLIT_AVAILABLE:
            logger.warning("Streamlit is not available, cannot access Groq API key")
            return None
        
        # Check if the key is set in session state
        if hasattr(st.session_state, 'groq_api_key') and st.session_state.groq_api_key:
            return st.session_state.groq_api_key
        
        return None
    
    @staticmethod
    def get_or_request_groq_api_key():
        """
        Get the Groq API key from session state or request it from the user.
        
        This is a convenience method that will check if the key is already set,
        and if not, it will render the input field and only return once the key is provided.
        
        Returns:
            str: The Groq API key
        """
        if not STREAMLIT_AVAILABLE:
            logger.warning("Streamlit is not available, cannot access Groq API key")
            return None
            
        # First check if the key is already set
        key = StreamlitIntegration.get_groq_api_key()
        if key:
            return key
            
        # If not set, render the input field
        return StreamlitIntegration.render_groq_api_key_input()

# --------------------------------------------------------------------------- #
#  Logging setup                                                              #
# --------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)
if not logger.handlers:
    ch = logging.StreamHandler()
    fh = logging.FileHandler("crew_ai.log")
    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
#  Role identifiers                                                           #
# --------------------------------------------------------------------------- #

class Role:
    PROCESSOR   = "processor"
    EMBEDDER    = "embedder"
    RETRIEVER   = "retriever"
    SYNTHESIZER = "synthesizer"

# --------------------------------------------------------------------------- #
#  Base Agent                                                                 #
# --------------------------------------------------------------------------- #

class Agent(ABC):
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 log_level: int = logging.INFO):
        self.config = config or {}
        self.log_level = log_level
        self._timers: Dict[str, float] = {}
        self._stats: Dict[str, Dict[str, Any]] = {}
        self._init_logger()

    # -------------- instrumentation --------------
    def _init_logger(self):
        self.logger = logging.getLogger(f"agent.{self.__class__.__name__}")
        self.logger.setLevel(self.log_level)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self.logger.addHandler(h)

    def start(self, name: str):
        self._timers[name] = time.time()

    def stop(self, name: str) -> float:
        if name not in self._timers:
            return 0.0
        dur = time.time() - self._timers[name]
        s = self._stats.setdefault(name, dict(count=0, total=0.0,
                                              min=float("inf"),
                                              max=0.0))
        s["count"] += 1
        s["total"] += dur
        s["min"] = min(s["min"], dur)
        s["max"] = max(s["max"], dur)
        return dur

    # -------------- interface --------------
    @abstractmethod
    def process_document(self, document: Any) -> Dict[str, Any]:
        """
        Process a document and extract information.
        
        Args:
            document (Any): Document to process (can be a file path, bytes, dict, or text)
            
        Returns:
            Dict[str, Any]: Processing results with 'success' key indicating outcome
        """
        pass

    @abstractmethod
    def answer_question(self, document: Any, question: str) -> Dict[str, Any]:
        """
        Answer a question based on document content.
        
        Args:
            document (Any): Processed document or document reference
            question (str): Question to answer based on the document
            
        Returns:
            Dict[str, Any]: Answer with relevant metadata
        """
        pass

# --------------------------------------------------------------------------- #
#  Processor Agent                                                            #
# --------------------------------------------------------------------------- #

class ProcessorAgent(Agent):
    """Extracts text / chunks from files or raw text."""
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 log_level: int = logging.INFO):
        super().__init__(config, log_level)
        self.config.setdefault("chunk_size", 1000)
        self.config.setdefault("chunk_overlap", 200)

        # Initialize PDF embedder
        self.pdf_embedder = None
        try:
            # Try direct import first
            try:
                from pdf_embedder import PDFEmbedder
                self.pdf_embedder = PDFEmbedder(log_level=self.log_level)
                self.logger.info("PDFEmbedder initialized directly")
            except ImportError:
                # Try through helper
                try:
                    from pdf_helper import get_pdf_embedder
                    self.pdf_embedder = get_pdf_embedder(
                        chunk_size=self.config["chunk_size"],
                        chunk_overlap=self.config["chunk_overlap"],
                        log_level=self.log_level
                    )
                    self.logger.info("PDFEmbedder initialized via helper")
                except Exception as e:
                    self.logger.warning(f"PDFEmbedder not available via helper: {e}")
                    self.logger.warning("PDF processing will be limited")
        except Exception as e:
            self.logger.warning(f"Failed to initialize PDFEmbedder: {e}")

    # ---- helpers ----
    def _chunk_text(self, txt: str) -> List[str]:
        sz   = self.config["chunk_size"]
        over = self.config["chunk_overlap"]
        if not txt or len(txt) <= sz:
            return [txt] if txt else []
        chunks = []
        start = 0
        while start < len(txt):
            end = min(start + sz, len(txt))
            chunk = txt[start:end]
            chunks.append(chunk.strip())
            start = end - over
        return chunks

    def _process_pdf(self, path: str) -> Tuple[str, List[str]]:
        if self.pdf_embedder:
            # If PDFEmbedder is available, use it directly
            try:
                if hasattr(self.pdf_embedder, 'process_pdf'):
                    success = self.pdf_embedder.process_pdf(path)
                    if not success:
                        raise RuntimeError("PDF processing failed")
                    return self.pdf_embedder.document_text, self.pdf_embedder.document_chunks
                else:
                    self.logger.error("PDFEmbedder instance doesn't have process_pdf method")
            except Exception as e:
                self.logger.error(f"Error using PDFEmbedder: {e}")
        
        # Fallback method using PyPDF2 if PDFEmbedder is not available
        try:
            import PyPDF2
            self.logger.info("Using PyPDF2 fallback for PDF processing")
            
            with open(path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n\n"
            
            chunks = self._chunk_text(text)
            return text, chunks
        except Exception as e:
            self.logger.error(f"PyPDF2 fallback also failed: {e}")
            raise RuntimeError(f"PDF processing failed: {e}")

    def _get_document_id(self, document: Any) -> str:
        """
        Generate a unique identifier for the document for caching purposes.
        
        Args:
            document (Any): The document (can be bytes, string, dictionary, or Streamlit UploadedFile)
            
        Returns:
            str: A unique identifier
        """
        import hashlib
        
        # Handle Streamlit UploadedFile
        if STREAMLIT_AVAILABLE and hasattr(document, "name") and hasattr(document, "read") and hasattr(document, "getvalue"):
            # This appears to be a Streamlit UploadedFile
            try:
                file_bytes = document.getvalue()
                file_name = getattr(document, "name", "uploaded_file")
                # Create a hash of the content plus the filename
                content_hash = hashlib.md5(file_bytes).hexdigest()
                return f"uploaded_{os.path.basename(file_name)}_{content_hash[:8]}"
            except Exception as e:
                self.logger.warning(f"Error generating ID for UploadedFile: {e}")
                # Fallback to object ID
                return f"uploaded_file_{time.time()}_{id(document)}"
                
        # Handle bytes
        elif isinstance(document, bytes):
            return f"bytes_{hashlib.md5(document).hexdigest()}"
        
        # Handle string (file path or text)
        elif isinstance(document, str):
            # If it's a file path, use the filename and modification time
            if os.path.exists(document):
                stat = os.stat(document)
                return f"file_{os.path.basename(document)}_{stat.st_mtime}"
            else:
                # For raw text, use a hash of the content
                return f"text_{hashlib.md5(document.encode()).hexdigest()[:16]}"
        
        # Handle dictionary with type
        elif isinstance(document, dict) and "type" in document:
            if document["type"] == "pdf_file" and "path" in document:
                stat = os.stat(document["path"])
                return f"file_{os.path.basename(document['path'])}_{stat.st_mtime}"
            
            elif document["type"] == "pdf_bytes" and "content" in document:
                return f"bytes_{hashlib.md5(document['content']).hexdigest()}"
            
            elif document["type"] == "text" and "content" in document:
                return f"text_{hashlib.md5(document['content'].encode()).hexdigest()[:16]}"
            
        # Fallback to timestamp and object id for unrecognized types
        return f"doc_{time.time()}_{id(document)}"

    # ---- API ----
    def process_document(self, document: Any) -> Dict[str, Any]:
        self.start("process_document")
        try:
            # Check for Streamlit's UploadedFile type
            if STREAMLIT_AVAILABLE and hasattr(document, "name") and hasattr(document, "read") and hasattr(document, "getvalue"):
                # This is likely a Streamlit UploadedFile
                # Extract file name and bytes content
                file_name = getattr(document, "name", "uploaded_file")
                self.logger.info(f"Processing Streamlit UploadedFile: {file_name}")
                
                try:
                    # Get file content as bytes
                    file_bytes = document.getvalue()
                    file_size = len(file_bytes)/1024
                    self.logger.info(f"UploadedFile size: {file_size:.2f} KB")
                    
                    if not file_bytes:
                        raise ValueError("Empty file content")
                        
                    # Extract file extension
                    _, ext = os.path.splitext(file_name.lower())
                    
                    # Process based on file type
                    if ext == '.pdf':
                        # Validate PDF format
                        if len(file_bytes) < 5 or not file_bytes.startswith(b'%PDF-'):
                            self.logger.error(f"Invalid PDF header in uploaded file")
                            raise ValueError("The uploaded file does not appear to be a valid PDF. Missing PDF signature.")
                            
                        # Write to temporary file for processing
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
                            tf.write(file_bytes)
                            tmp_path = tf.name
                        
                        try:
                            # Process the PDF
                            text, chunks = self._process_pdf(tmp_path)
                            doc_type = "pdf"
                        finally:
                            # Clean up temp file
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                                
                    elif ext == '.txt':
                        # Process as text
                        text = file_bytes.decode('utf-8', errors='replace')
                        chunks = self._chunk_text(text)
                        doc_type = "text"
                        
                    else:
                        raise ValueError(f"Unsupported file type: {ext}. Supported formats: .pdf, .txt")
                
                except Exception as e:
                    self.logger.error(f"Error processing UploadedFile: {e}")
                    if "PDF" in str(e):
                        return {"success": False, "error": f"Error processing uploaded PDF: {e}"}
                    else:
                        return {"success": False, "error": f"Error processing uploaded file '{file_name}': {e}"}
                
            elif isinstance(document, str):
                if os.path.isfile(document):
                    _, ext = os.path.splitext(document)
                    if ext.lower() == ".pdf":
                        try:
                            text, chunks = self._process_pdf(document)
                            doc_type = "pdf"
                        except Exception as e:
                            self.logger.error(f"PDF processing failed: {e}")
                            return {"success": False, "error": f"PDF processing failed: {e}"}
                    else:  # treat as plain text file
                        try:
                            with open(document, "r", encoding="utf-8") as f:
                                text = f.read()
                            chunks = self._chunk_text(text)
                            doc_type = "text"
                        except UnicodeDecodeError:
                            # Try with a different encoding if utf-8 fails
                            try:
                                with open(document, "r", encoding="latin-1") as f:
                                    text = f.read()
                                chunks = self._chunk_text(text)
                                doc_type = "text"
                            except Exception as e:
                                self.logger.error(f"Failed to read text file: {e}")
                                return {"success": False, "error": f"Failed to read text file: {e}"}
                else:
                    # raw text
                    text = document
                    chunks = self._chunk_text(text)
                    doc_type = "text"
                    
            elif isinstance(document, bytes):
                # Check if it's a PDF by magic bytes
                if document[:4] == b'%PDF':
                    # store to tmp and recurse
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
                        tf.write(document)
                        tmp_path = tf.name
                    try:
                        result = self.process_document(tmp_path)
                        return result
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                else:
                    # Try to decode as text
                    try:
                        text = document.decode('utf-8')
                        chunks = self._chunk_text(text)
                        doc_type = "text_bytes"
                    except UnicodeDecodeError:
                        return {"success": False, "error": "Unable to process binary data that is not a PDF"}
                        
            elif isinstance(document, dict) and "type" in document and "content" in document:
                # Handle dictionary with type and content
                if document["type"] == "text":
                    text = document["content"]
                    chunks = self._chunk_text(text)
                    doc_type = "text"
                elif document["type"] == "pdf" and isinstance(document["content"], bytes):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
                        tf.write(document["content"])
                        tmp_path = tf.name
                    try:
                        text, chunks = self._process_pdf(tmp_path)
                        doc_type = "pdf"
                    except Exception as e:
                        self.logger.error(f"PDF processing failed: {e}")
                        return {"success": False, "error": f"PDF processing failed: {e}"}
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                else:
                    return {"success": False, "error": f"Unsupported document type: {document['type']}"}
            else:
                self.logger.error(f"Unsupported document format: {type(document)}")
                return {"success": False, "error": f"Unsupported document format: {type(document)}. Expected string, bytes, dictionary with type and content, or Streamlit UploadedFile."}

            return {"success": True,
                    "text": text,
                    "chunks": chunks,
                    "document_type": doc_type,
                    "metadata": {"chunk_count": len(chunks),
                                 "size": len(text)}}
        except Exception as e:
            self.logger.error(f"Processor error: {e}")
            return {"success": False, "error": str(e)}
        finally:
            self.stop("process_document")

    def answer_question(self, document: Any, question: str) -> Dict[str, Any]:
        return {"success": False,
                "error": "ProcessorAgent does not handle Q&A directly"}

# --------------------------------------------------------------------------- #
#  Embedder Agent                                                             #
# --------------------------------------------------------------------------- #

class EmbedderAgent(Agent):
    """Generates dense embeddings for chunks."""
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 log_level: int = logging.INFO):
        super().__init__(config, log_level)
        self.config.setdefault("model_name", "all-MiniLM-L6-v2")
        self.doc_chunks:  List[str]         = []
        self.doc_embeds:  Optional[np.ndarray] = None

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.config["model_name"])
        except Exception as e:
            self._model = None
            self.logger.warning(f"sentence-transformers not available: {e}")

    def process_document(self, document: Any) -> Dict[str, Any]:
        self.start("embed")
        try:
            if not isinstance(document, dict):
                return {"success": False, "error": "Expected dictionary from processor agent"}
                
            if not document.get("success", False):
                # Pass through the error
                return document
                
            if "chunks" not in document:
                return {"success": False, "error": "No chunks found in processor payload"}
                
            chunks = document["chunks"]
            if not chunks:
                return {"success": False, "error": "Empty chunks list in processor payload"}
                
            if self._model is None:
                return {"success": False, "error": "Embedding model unavailable"}

            embeds = self._model.encode(chunks, show_progress_bar=False)
            self.doc_chunks = chunks
            self.doc_embeds = embeds

            return {"success": True,
                    "chunks": chunks,
                    "embeddings": embeds,
                    "metadata": {"embedding_dim": embeds.shape[1] if len(embeds) > 0 else 0}}
        except Exception as e:
            self.logger.error(f"Embedder error: {e}")
            return {"success": False, "error": str(e)}
        finally:
            self.stop("embed")

    def answer_question(self, document: Any, question: str) -> Dict[str, Any]:
        return {"success": False,
                "error": "EmbedderAgent does not handle Q&A directly"}

# --------------------------------------------------------------------------- #
#  Retriever Agent                                                            #
# --------------------------------------------------------------------------- #

class RetrieverAgent(Agent):
    """Performs semantic search over stored embeddings."""
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 log_level: int = logging.INFO):
        super().__init__(config, log_level)
        self.config.setdefault("top_k", 5)
        self.config.setdefault("similarity_threshold", 0.3)

        self.chunks:  List[str]          = []
        self.embeds:  Optional[np.ndarray] = None

        try:
            from sentence_transformers import SentenceTransformer
            self._query_model = SentenceTransformer(
                config.get("model_name", "all-MiniLM-L6-v2") if config else "all-MiniLM-L6-v2")
        except Exception as e:
            self._query_model = None
            self.logger.warning(f"sentence-transformers not available: {e}")

    def process_document(self, document: Any) -> Dict[str, Any]:
        if not isinstance(document, dict):
            return {"success": False, "error": "Expected dictionary from embedder agent"}
            
        if not document.get("success", False):
            # Pass through the error
            return document
            
        if "chunks" not in document or "embeddings" not in document:
            return {"success": False, "error": "Missing chunks or embeddings in embedder payload"}
            
        self.chunks = document["chunks"]
        self.embeds = np.asarray(document["embeddings"])
        return {"success": True}

    def answer_question(self, document: Any, question: str) -> Dict[str, Any]:
        self.start("search")
        try:
            # If document is provided, override the stored embeds/chunks
            if document is not None and isinstance(document, dict) and document.get("success"):
                if "chunks" in document and "embeddings" in document:
                    self.chunks = document["chunks"]
                    self.embeds = np.asarray(document["embeddings"])
            
            if self.embeds is None or not self.chunks:
                return {"success": False, "error": "No indexed document"}

            if self._query_model is None:
                return {"success": False, "error": "Query model unavailable"}

            q_vec = self._query_model.encode(question)
            
            # Compute cosine similarity
            sims = np.dot(self.embeds, q_vec) / (
                np.linalg.norm(self.embeds, axis=1) *
                np.linalg.norm(q_vec) + 1e-10
            )
            
            # Apply threshold and get top-k results
            mask = sims >= self.config["similarity_threshold"]
            if not np.any(mask):  # No results meet the threshold
                return {"success": True, "relevant_chunks": []}
                
            idx_sorted = np.argsort(-sims[mask])[:self.config["top_k"]]
            indices = np.where(mask)[0][idx_sorted]

            results = [
                {"rank": i + 1,
                 "chunk": self.chunks[idx],
                 "score": float(sims[idx]),
                 "index": int(idx)}
                for i, idx in enumerate(indices)
            ]
            return {"success": True, "relevant_chunks": results, "question": question}
        except Exception as e:
            self.logger.error(f"Retriever error: {e}")
            return {"success": False, "error": str(e)}
        finally:
            self.stop("search")

# --------------------------------------------------------------------------- #
#  Synthesizer Agent                                                          #
# --------------------------------------------------------------------------- #

class SynthesizerAgent(Agent):
    """
    Agent specialized in synthesizing information and generating responses.
    
    Responsible for:
    - Combining information from multiple sources
    - Generating coherent answers to questions
    - Summarizing retrieved information
    - Ensuring generated content is accurate and relevant
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, log_level: int = logging.INFO):
        """Initialize a SynthesizerAgent."""
        config = config or {}
        config.update({
            "role": Role.SYNTHESIZER,
            "model_name": config.get("model_name", "llama3-8b-8192"),  # Default to Llama 3 8B model
            "max_tokens": config.get("max_tokens", 1000),
            "temperature": config.get("temperature", 0.7)
        })
        super().__init__(config, log_level)
        
        self.logger.info(f"Initializing SynthesizerAgent with model: {config.get('model_name', 'llama3-8b-8192')}")
        self.api_type = None  # Initialize api_type attribute
        
        # Initialize the language model right away
        self._init_language_model()
    
    def _init_language_model(self):
        """Initialize the language model based on configuration."""
        # Default to using Groq API with llama3-8b-8192 model
        model_name = self.config.get("model_name", "llama3-8b-8192")
        
        try:
            # Try to use Groq API first
            try:
                import groq
                self.api_type = "groq"
                # Groq client will be initialized for each request (to handle API key changes)
                self.logger.info(f"Using Groq API with model: {model_name}")
                return
            except ImportError:
                self.logger.warning("Groq package not available. Install with: pip install groq")
                
            # Try OpenAI API as second option
            try:
                import openai
                self.api_type = "openai"
                # OpenAI client will be initialized for each request (to handle API key changes)
                self.logger.info(f"Using OpenAI API with model: {model_name}")
                return
            except ImportError:
                self.logger.warning("OpenAI package not available")
                
            # Try Hugging Face transformers as fallback
            try:
                from transformers import pipeline
                self.generator = pipeline(
                    "text-generation", 
                    model=model_name if "gpt" not in model_name else "gpt2"
                )
                self.api_type = "huggingface"
                self.logger.info(f"Using Hugging Face with model: {model_name}")
                return
            except ImportError:
                self.logger.warning("Transformers package not available")
                
            # No language model available
            self.logger.error("No language model available. Install groq, openai or transformers package.")
            self.api_type = None
            
        except Exception as e:
            self.logger.error(f"Error initializing language model: {str(e)}")
            self.logger.exception("Exception details")
            self.api_type = None
            
    def _format_response(self, raw_response: Any) -> Dict[str, Any]:
        """
        Format the raw response from the agent into a standardized structure.
        
        Args:
            raw_response (Any): Raw response from the underlying implementation
            
        Returns:
            Dict[str, Any]: Standardized response dictionary
        """
        # Handle different types of raw response
        if isinstance(raw_response, dict):
            # Response is already a dictionary
            if "success" in raw_response:
                # Response is already in the expected format
                return raw_response
            else:
                # Add standardized fields
                return {
                    "success": True,
                    "content": raw_response.get("content", raw_response.get("text", str(raw_response))),
                    "metadata": raw_response.get("metadata", {})
                }
                
        elif isinstance(raw_response, str):
            # Response is a simple string, wrap it in our standard format
            return {
                "success": True,
                "content": raw_response,
                "metadata": {
                    "model": self.config.get("model_name", "unknown"),
                    "api_type": self.api_type,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        elif raw_response is None:
            # No response
            return {
                "success": False,
                "error": "No response generated",
                "metadata": {
                    "model": self.config.get("model_name", "unknown"),
                    "api_type": self.api_type,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        else:
            # Convert any other type to string
            return {
                "success": True,
                "content": str(raw_response),
                "metadata": {
                    "model": self.config.get("model_name", "unknown"),
                    "api_type": self.api_type,
                    "original_type": type(raw_response).__name__,
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    def process_document(self, document: Any) -> Dict[str, Any]:
        """
        Generate a summary or overview of the document.
        
        Args:
            document: Document with chunks or content
            
        Returns:
            Dict with generated summary and metadata
        """
        self.start("process_document")  # Changed from start_timing to start
        
        try:
            # Log document type for debugging
            self.logger.info(f"Processing document of type: {type(document).__name__}")
            
            # Extract document content
            chunks = []
            
            if isinstance(document, dict):
                if document.get("success", False):
                    self.logger.info("Processing successful document result")
                    
                if "chunks" in document:
                    # Extract text from each chunk properly
                    self.logger.info(f"Found {len(document['chunks'])} chunks in document")
                    for chunk in document["chunks"]:
                        if isinstance(chunk, dict):
                            if "text" in chunk:
                                chunks.append(chunk["text"])
                            elif "content" in chunk:
                                chunks.append(chunk["content"])
                            elif "chunk" in chunk:
                                chunks.append(chunk["chunk"])
                            else:
                                chunks.append(str(chunk))
                        else:
                            chunks.append(str(chunk))
                            
                elif "text" in document:
                    self.logger.info("Found text field in document")
                    chunks = [document["text"]]
                    
                elif "content" in document:
                    self.logger.info("Found content field in document")
                    chunks = [document["content"]]
                    
                else:
                    # Try to extract any text-like fields from the dictionary
                    self.logger.warning("No standard content fields found, searching for any text fields")
                    for key, value in document.items():
                        if isinstance(value, str) and len(value) > 100:  # Assume longer strings are content
                            chunks.append(value)
                            self.logger.info(f"Using text from field '{key}'")
                    
                    if not chunks:
                        self.logger.error("Document missing required content fields")
                        return {
                            "success": False,
                            "error": "Document missing content"
                        }
                        
            elif isinstance(document, list):
                # Process list items, extracting text from dictionaries if needed
                self.logger.info(f"Processing list document with {len(document)} items")
                
                for item in document:
                    if isinstance(item, dict):
                        # Try to extract text from dictionary item
                        if "text" in item:
                            chunks.append(item["text"])
                        elif "content" in item:
                            chunks.append(item["content"])
                        elif "chunk" in item:
                            chunks.append(item["chunk"])
                        else:
                            # Use string representation as fallback
                            chunks.append(str(item))
                    elif isinstance(item, str):
                        chunks.append(item)
                    else:
                        # Convert non-string/dict items to string
                        chunks.append(str(item))
                        
            elif isinstance(document, str):
                # Single text string
                self.logger.info("Processing document as a single string")
                chunks = [document]
                
            else:
                self.logger.error(f"Invalid document format for synthesis: {type(document).__name__}")
                return {
                    "success": False,
                    "error": f"Invalid document format: {type(document).__name__}"
                }
            
            # Log the number of extracted chunks
            self.logger.info(f"Extracted {len(chunks)} chunks for summarization")
            
            # Use more chunks than before, up to a reasonable limit
            max_chunks = min(len(chunks), 8)  # Increase from 5 to 8 for better coverage
            combined_text = "\n\n".join(chunks[:max_chunks])
            truncated = len(chunks) > max_chunks
            
            # Better structured prompt for summarization
            prompt = (
                "# DOCUMENT CONTENT\n"
                f"{combined_text}\n\n"
                f"{'Note: This is a partial extract of the full document.' if truncated else 'This is the complete document.'}\n\n"
                "# TASK\n"
                "Create a comprehensive summary of the document above that:\n"
                "1. Captures the main topic and purpose of the document\n"
                "2. Highlights the key points, findings, or arguments\n"
                "3. Preserves important details like names, dates, and figures\n"
                "4. Presents information in a well-structured format\n"
                "5. Is detailed while still being concise\n\n"
                "# SUMMARY\n"
            )
            
            # Generate summary
            self.start("generation")  # Changed from start_timing to start
            self.logger.info("Generating document summary using Groq API")
            summary = self._generate_text(prompt)
            generation_time = self.stop("generation")  # Changed from end_timing to stop
            
            result = {
                "success": True,
                "summary": summary,
                "metadata": {
                    "generation_time": generation_time,
                    "model": self.config["model_name"],
                    "chunk_count": len(chunks),
                    "chunks_used": max_chunks,
                    "truncated": truncated,
                    "generated_at": datetime.now().isoformat()
                }
            }
            
            self.logger.info(f"Document summary generated ({len(summary)} chars)")
            return self._format_response(result)
            
        except Exception as e:
            self.logger.error(f"Error generating document summary: {str(e)}")
            self.logger.exception("Exception details")
            return {
                "success": False,
                "error": str(e)
            }
            
        finally:
            self.stop("process_document")  # Changed from end_timing to stop
    
    def answer_question(self, document: Any, question: str) -> Dict[str, Any]:
        """
        Generate an answer to a question based on retrieved document chunks.
        
        Args:
            document: Retrieved chunks or document content
            question: Question to answer
            
        Returns:
            Dict with generated answer and metadata
        """
        self.start("answer_question")  # Changed from start_timing to start
        
        try:
            # Check if we have a language model
            if self.api_type is None:
                self.logger.error("No language model available for answer generation")
                return {
                    "success": False,
                    "error": "Language model not available. Please install groq, openai or transformers package."
                }
            
            # Extract relevant chunks from document
            relevant_chunks = []
            
            if isinstance(document, dict):
                if document.get("success") and "relevant_chunks" in document:
                    # This is output from the RetrieverAgent
                    chunk_objects = document["relevant_chunks"]
                    for chunk_obj in chunk_objects:
                        if isinstance(chunk_obj, dict) and "chunk" in chunk_obj:
                            relevant_chunks.append(chunk_obj["chunk"])
                        else:
                            relevant_chunks.append(str(chunk_obj))
                elif "chunks" in document:
                    # Raw chunks from processor or embedder
                    relevant_chunks = document["chunks"][:5]  # Take first few to avoid token limits
                elif "text" in document:
                    # Single text content
                    relevant_chunks = [document["text"]]
                else:
                    # Try to extract text from any field that might contain document content
                    for key, value in document.items():
                        if isinstance(value, str) and len(value) > 100:  # Likely to be text content
                            relevant_chunks.append(value)
                            self.logger.info(f"Extracted content from '{key}' field")
            elif isinstance(document, list):
                # Process list of text chunks or chunk objects
                chunk_count = 0
                for item in document:
                    if chunk_count >= 5:  # Limit chunks to avoid token limits
                        break
                    
                    if isinstance(item, dict):
                        # Try to extract text from dictionary items
                        if "chunk" in item:
                            relevant_chunks.append(item["chunk"])
                            chunk_count += 1
                        elif "text" in item:
                            relevant_chunks.append(item["text"])
                            chunk_count += 1
                        elif "content" in item:
                            relevant_chunks.append(item["content"])
                            chunk_count += 1
                    else:
                        # Convert to string if it's not already
                        relevant_chunks.append(str(item))
                        chunk_count += 1
            elif isinstance(document, str):
                # Single text string
                relevant_chunks = [document]
                
            # Handle the case when no chunks are found but document exists
            if not relevant_chunks and document is not None:
                self.logger.warning("Could not extract text chunks from document, trying to convert entire document to string")
                try:
                    # Last resort - convert entire document to string
                    doc_str = str(document)
                    if len(doc_str) > 100:  # Only use if it looks like actual content
                        relevant_chunks = [doc_str[:10000]]  # Limit size to avoid token limits
                        self.logger.info("Using stringified document as fallback")
                except:
                    pass
                    
            if not relevant_chunks:
                self.logger.error("No relevant content found to answer the question")
                return {
                    "success": False,
                    "error": "No relevant content available in the document to answer this question"
                }
            
            # Log chunk information
            self.logger.info(f"Using {len(relevant_chunks)} chunks with total length of {sum(len(c) for c in relevant_chunks)} characters")
            
            # Prepare context by joining chunks with clear separators
            # Use more informative separator if chunks have indexes
            context_parts = []
            for i, chunk in enumerate(relevant_chunks):
                # Trim chunks if they're extremely long
                if len(chunk) > 8000:
                    self.logger.warning(f"Truncating very long chunk {i+1} from {len(chunk)} to 8000 chars")
                    chunk = chunk[:8000] + "... [content truncated]"
                
                # Add separator for each chunk to make context clearer
                context_parts.append(f"[Document Chunk {i+1}]\n{chunk}")
            
            context = "\n\n" + "\n\n".join(context_parts)
            
            # Create a prompt that provides clear instructions for using the context
            prompt = (
                "Answer the following question based ONLY on the information provided in the document chunks below. "
                "If the information needed is not in the document chunks, respond with 'The document doesn't provide "
                "information to answer this question.' Be specific and cite relevant parts of the document in your answer.\n\n"
                f"DOCUMENT CHUNKS:\n{context}\n\n"
                f"QUESTION: {question}\n\n"
                "ANSWER:"
            )
            
            # Generate answer
            self.start("generation")  # Changed from start_timing to start
            answer = self._generate_text(prompt)
            generation_time = self.stop("generation")  # Changed from end_timing to stop
            
            result = {
                "success": True,
                "question": question,
                "answer": answer,
                "metadata": {
                    "generation_time": generation_time,
                    "model": self.config["model_name"],
                    "context_chunks": len(relevant_chunks),
                    "context_length": sum(len(c) for c in relevant_chunks),
                    "api_type": self.api_type,
                    "generated_at": datetime.now().isoformat()
                }
            }
            
            self.logger.info(f"Question '{question[:50]}{'...' if len(question) > 50 else ''}' answered ({len(answer)} chars)")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            self.logger.exception("Exception details")
            return {
                "success": False,
                "error": f"Error generating answer: {str(e)}",
                "question": question
            }
            
        finally:
            self.stop("answer_question")  # Changed from end_timing to stop
    
    def _generate_text(self, prompt: str) -> str:
        """
        Generate text using the configured language model.
        
        Args:
            prompt: Text prompt for generation
            
        Returns:
            Generated text
        """
        max_tokens = self.config.get("max_tokens", 1000)
        temperature = self.config.get("temperature", 0.7)
        
        if self.api_type == "groq":
            # Get Groq API key from environment or Streamlit session 
            api_key = os.environ.get("GROQ_API_KEY")
            
            # Check if we're in a Streamlit environment and try to get the key from session state
            if not api_key and STREAMLIT_AVAILABLE and 'groq_api_key' in st.session_state:
                api_key = st.session_state.groq_api_key
                
            if not api_key:
                self.logger.error("Groq API key not found in environment or session state")
                return "Error: Groq API key not found. Please set the GROQ_API_KEY environment variable or provide it through the Streamlit interface."
                
            # Use Groq API
            import groq
            
            # Create Groq client
            client = groq.Client(api_key=api_key)
            model_name = self.config.get("model_name", "llama3-8b-8192")
            
            try:
                self.logger.info(f"Making Groq API request with model {model_name}")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant that answers questions based on the document content provided. Always be factual and stick to the information in the provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                self.logger.info(f"Groq API response received, content length: {len(response.choices[0].message.content)}")
                return response.choices[0].message.content
                
            except Exception as e:
                self.logger.error(f"Error using Groq API: {str(e)}")
                self.logger.exception("Exception details")
                return f"Error generating response with Groq: {str(e)}"
                
        elif self.api_type == "openai":
            # Use OpenAI API
            import openai
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model=self.config["model_name"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        elif self.api_type == "huggingface":
            # Use Hugging Face Transformers
            response = self.generator(
                prompt, 
                max_length=len(prompt.split()) + max_tokens,
                temperature=temperature,
                num_return_sequences=1
            )
            
            # Extract generated text and remove the prompt
            generated_text = response[0]["generated_text"]
            if generated_text.startswith(prompt):
                return generated_text[len(prompt):].strip()
            return generated_text.strip()
            
        else:
            # Fallback to simple template-based generation for demo
            self.logger.warning("Using fallback text generation (no language model available)")
            return (
                "I cannot generate a proper response because no language model is available. "
                "Please install the Groq package, OpenAI package, or Hugging Face Transformers to enable full functionality."
            )
# --------------------------------------------------------------------------- #
#  Crew Manager                                                               #
# --------------------------------------------------------------------------- #

class CrewManager:
    """Coordinates Processor → Embedder → Retriever → Synthesizer."""
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 log_level: int = logging.INFO):
        self.config = config or {}
        self.log_level = log_level
        self.logger = logging.getLogger("crew.CrewManager")
        self.logger.setLevel(log_level)

        # Agents
        self.processor = ProcessorAgent(config=self.config.get("processor_config"), log_level=log_level)
        self.embedder = EmbedderAgent(config=self.config.get("embedder_config"), log_level=log_level)
        self.retriever = RetrieverAgent(config=self.config.get("retriever_config"), log_level=log_level)
        self.synthesizer = SynthesizerAgent(config=self.config.get("synthesizer_config"), log_level=log_level)

        # State
        self._processed: Optional[Dict[str, Any]] = None
        self._embedded: Optional[Dict[str, Any]] = None
        self._document_id = None
        
        # Add a documents dictionary to store processed documents
        self.documents = {}

    # -------------- document management --------------
    def _generate_document_id(self) -> str:
        """Generate a unique document ID"""
        return str(uuid.uuid4())
        
    # -------------- workflow --------------
    def process_document(self, document: Any) -> Dict[str, Any]:
        """
        Process a document through the agent pipeline.
        
        Args:
            document (Any): Can be a file path string, bytes content, or a dictionary
                        with 'type' and 'content' fields
        
        Returns:
            Dict[str, Any]: Processing results with summary if successful
        """
        self.logger.info("CrewManager: processing document")
        start_time = time.time()
        
        # Generate a document ID
        self._document_id = self._generate_document_id()
        
        # Step 1: Process the document with the ProcessorAgent
        self.logger.debug("Step 1: Processing document with ProcessorAgent")
        proc_result = self.processor.process_document(document)
        self._processed = proc_result
        
        if not proc_result.get("success", False):
            self.logger.error(f"Document processing failed: {proc_result.get('error', 'Unknown error')}")
            return proc_result
        
        # Step 2: Generate embeddings with EmbedderAgent
        self.logger.debug("Step 2: Generating embeddings with EmbedderAgent")
        emb_result = self.embedder.process_document(proc_result)
        self._embedded = emb_result
        
        if not emb_result.get("success", False):
            self.logger.error(f"Embedding generation failed: {emb_result.get('error', 'Unknown error')}")
            return emb_result
        
        # Step 3: Index the embedded document with RetrieverAgent
        self.logger.debug("Step 3: Indexing embeddings with RetrieverAgent")
        ret_result = self.retriever.process_document(emb_result)
        
        if not ret_result.get("success", False):
            self.logger.error(f"Indexing failed: {ret_result.get('error', 'Unknown error')}")
            return ret_result
        
        # Step 4: Generate a summary with SynthesizerAgent
        self.logger.debug("Step 4: Generating summary with SynthesizerAgent")
        summ_result = self.synthesizer.process_document(proc_result)
        
        # Add document metadata and timing info
        result = {
            "success": True,
            "document_id": self._document_id,
            "summary": summ_result.get("summary", ""),
            "metadata": {
                "document_type": proc_result.get("document_type", "unknown"),
                "chunks": len(proc_result.get("chunks", [])),
                "text_size": len(proc_result.get("text", "")),
                "embedding_dim": emb_result.get("metadata", {}).get("embedding_dim", 0),
                "processing_time": time.time() - start_time
            }
        }
        
        # Store the processed document in the documents dictionary
        self.documents[self._document_id] = {
            "id": self._document_id,
            "name": getattr(document, "name", str(self._document_id)) if hasattr(document, "name") else str(self._document_id),
            "processed_data": {
                "processor_result": proc_result,
                "embedder_result": emb_result,
                "retriever_result": ret_result
            },
            "processed_at": datetime.now().isoformat()
        }
        
        self.logger.info(f"Document processed in {result['metadata']['processing_time']:.2f} seconds")
        return result
    
        # First change - update query_document method
    def query_document(self, document_id: str, question: str) -> Dict[str, Any]:
        """
        Query a processed document with a question.
        
        Args:
            document_id (str): ID of the document to query
            question (str): Question to ask about the document
            
        Returns:
            Dict[str, Any]: Answer to the question with supporting context
        """
        self.logger.info(f"Querying document {document_id} with question: {question}")
        start_time = time.time()
        
        if document_id not in self.documents:
            raise ValueError(f"Document with ID {document_id} not found")
        
        document = self.documents[document_id]
        
        try:
            # IMPORTANT: Use the embedding agent directly, avoid calling answer_question
            # to prevent infinite recursion
            self.logger.debug(f"Delegating question answering to EmbeddingAgent")
            answer = self.embedding_agent.answer_question(
                document["processed_data"], 
                question
            )
            
            # Format the response
            response = {
                "question": question,
                "answer": answer.get("answer", "No answer found"),
                "document_id": document_id,
                "document_name": document["name"],
                "context_chunks": answer.get("context_chunks", []),
                "query_time_seconds": time.time() - start_time
            }
            
            self.logger.info(f"Question answered in {response['query_time_seconds']:.2f} seconds")
            return response
            
        except Exception as e:
            self.logger.error(f"Error querying document: {str(e)}")
            self.logger.exception("Exception details:")
            raise

    # Second change - update answer_question method
    def answer_question(self, document_id_or_question, question=None):
        """
        Compatibility wrapper for different calling patterns.
        
        This method serves as a compatibility layer for different calling conventions:
        1. If called with two arguments (document_id, question), it processes directly
        2. If called with (question, document_id=None), it handles the newer calling convention
        
        Args:
            document_id_or_question: Either the document_id or the question
            question: If provided, this is the question and document_id_or_question is the document ID
            
        Returns:
            Dict[str, Any]: Answer to the question with supporting context
        """
        # Handle compatibility - is this called as (document_id, question) ?
        if question is not None:
            # Traditional call with (document_id, question) from the app.py code
            document_id = document_id_or_question
            
            # Ensure document_id is a string, not a dictionary
            if isinstance(document_id, dict):
                # Extract document_id from the dictionary or generate a new one
                document_id = document_id.get("document_id", str(uuid.uuid4()))
                self.logger.warning(f"Converted dictionary to document_id: {document_id}")
            
            if document_id not in self.documents:
                self.logger.error(f"Document ID {document_id} not found in documents dictionary")
                return {"success": False, "error": f"Document ID {document_id} not found"}
            
            document = self.documents[document_id]
            start_time = time.time()
            
            # Step 1: Retrieve relevant chunks with RetrieverAgent
            self.logger.debug("Step 1: Retrieving relevant chunks with RetrieverAgent")
            retrieval_result = self.retriever.answer_question(document["processed_data"]["embedder_result"], question)
            
            if not retrieval_result.get("success", False):
                self.logger.error(f"Retrieval failed: {retrieval_result.get('error', 'Unknown error')}")
                return retrieval_result
            
            # Step 2: Generate answer with SynthesizerAgent
            self.logger.debug("Step 2: Generating answer with SynthesizerAgent")
            answer_result = self.synthesizer.answer_question(retrieval_result, question)
            
            # Format the response
            response = {
                "success": True,
                "question": question,
                "answer": answer_result.get("answer", "No answer found"),
                "document_id": document_id,
                "document_name": document.get("name", "Unknown"),
                "context_chunks": retrieval_result.get("relevant_chunks", []),
                "query_time_seconds": time.time() - start_time
            }
            
            return response
        
        else:
            # Called with just a question, and optional document_id parameter
            # Not implemented in the original version
            question = document_id_or_question
            
            self.logger.error("answer_question called with only question argument - not supported in this version")
            raise NotImplementedError(
                "This version of CrewManager requires document_id to be provided explicitly. " 
                "Please use answer_question(document_id, question) instead."
            )
# --------------------------------------------------------------------------- #
#  Example usage                                                              #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    crew = CrewManager(config={"verbose": True})
    pdf_path = "example.pdf"

    if os.path.exists(pdf_path):
        doc_res = crew.process_document(pdf_path)
        print("Summary:\n", doc_res.get("summary", "")[:500], "...\n")

        qa = crew.answer_question("What is the main topic?")
        print("Q:", qa.get("question", "What is the main topic?"))
        print("A:", qa.get("answer", ""))
    else:
        print("Place a PDF named 'example.pdf' beside this script to test.")