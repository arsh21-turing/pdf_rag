"""
PDFEmbedder - A utility class for extracting, chunking, and embedding text from PDF files

This module provides a PDFEmbedder class that handles PDF document processing with the following steps:
1. Text extraction using PyPDF2 with fallback to pdfplumber
2. Text cleaning and normalization 
3. Smart chunking with respect to document structure
4. Embedding generation using sentence-transformers
5. Semantic search capabilities for question answering
"""

import os
import re
import io
import logging
import tempfile
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

class PDFEmbedder:
    """
    A class for extracting text from PDFs, chunking it, and generating embeddings.
    
    This class handles the full pipeline of PDF processing:
    - PDF text extraction with fallback mechanisms
    - Text cleaning and normalization
    - Intelligent text chunking that respects semantic boundaries
    - Embedding generation using sentence-transformers
    - Semantic search capabilities
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2", 
                 log_level: int = logging.INFO):
        """
        Initialize PDFEmbedder with the specified model and logging configuration.
        
        Args:
            model_name (str): The name of the sentence-transformers model to use
            log_level (int): Logging level (default: logging.INFO)
        """
        self.model_name = model_name
        
        # Set up logging
        self.logger = logging.getLogger("PDFEmbedder")
        self.logger.setLevel(log_level)
        
        # Initialize state variables
        self.document_text = ""
        self.document_chunks = []
        self.document_embeddings = []
        self.document_metadata = {}
        
        # Import required libraries
        try:
            # Primary PDF reader
            import PyPDF2
            self.PyPDF2 = PyPDF2
            self.logger.debug(f"Loaded PyPDF2 version: {PyPDF2.__version__}")
        except ImportError:
            self.PyPDF2 = None
            self.logger.warning("PyPDF2 not available. Will try pdfplumber as fallback.")
        
        try:
            # Fallback PDF reader
            import pdfplumber
            self.pdfplumber = pdfplumber
            self.logger.debug("Loaded pdfplumber")
        except ImportError:
            self.pdfplumber = None
            if self.PyPDF2 is None:
                self.logger.error("Neither PyPDF2 nor pdfplumber available. PDF processing will not work.")
            else:
                self.logger.warning("pdfplumber not available. Using PyPDF2 only.")
        
        try:
            # For embedding generation
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.logger.info(f"Loaded embedding model: {model_name}")
        except ImportError:
            self.model = None
            self.logger.error("sentence-transformers not available. Embeddings will not work.")
    
    def _extract_text_with_pypdf2(self, pdf_file: Union[str, bytes, io.BytesIO]) -> str:
        """
        Extract text from a PDF using PyPDF2.
        
        Args:
            pdf_file: Either a file path (str), bytes object, or BytesIO object
            
        Returns:
            str: Extracted text
        """
        if self.PyPDF2 is None:
            self.logger.error("PyPDF2 is not available")
            return ""
        
        try:
            # Handle different input types
            if isinstance(pdf_file, str):
                # Path to file
                pdf_reader = self.PyPDF2.PdfReader(pdf_file)
            elif isinstance(pdf_file, bytes):
                # Bytes content
                pdf_reader = self.PyPDF2.PdfReader(io.BytesIO(pdf_file))
            else:
                # BytesIO object
                pdf_reader = self.PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text() or ""
                text += page_text + "\n\n"
            
            return text
        except Exception as e:
            self.logger.error(f"PyPDF2 extraction error: {e}")
            return ""
    
    def _extract_text_with_pdfplumber(self, pdf_file: Union[str, bytes, io.BytesIO]) -> str:
        """
        Extract text from a PDF using pdfplumber.
        
        Args:
            pdf_file: Either a file path (str), bytes object, or BytesIO object
            
        Returns:
            str: Extracted text
        """
        if self.pdfplumber is None:
            self.logger.error("pdfplumber is not available")
            return ""
        
        try:
            # Handle different input types
            if isinstance(pdf_file, str):
                # Path to file
                pdf = self.pdfplumber.open(pdf_file)
            elif isinstance(pdf_file, bytes):
                # Bytes content
                pdf = self.pdfplumber.open(io.BytesIO(pdf_file))
            else:
                # BytesIO object
                pdf = self.pdfplumber.open(pdf_file)
            
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n\n"
            
            pdf.close()
            return text
        except Exception as e:
            self.logger.error(f"pdfplumber extraction error: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text (str): Raw text from PDF
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix broken sentences
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
        
        # Normalize line endings
        text = text.replace('\r', '\n')
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix common OCR errors
        text = text.replace('|', 'I')
        text = re.sub(r'(\w)\/(\w)', r'\1/\2', text)
        
        # Trim whitespace
        text = text.strip()
        
        return text
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into chunks with overlap, trying to respect sentence boundaries.
        
        Args:
            text (str): Text to chunk
            chunk_size (int): Target size of each chunk
            overlap (int): Overlap between consecutive chunks
            
        Returns:
            List[str]: List of text chunks
        """
        if not text:
            return []
        
        # If text is shorter than chunk_size, return it as a single chunk
        if len(text) <= chunk_size:
            return [text]
        
        # Split text into paragraphs
        paragraphs = re.split(r'\n{2,}', text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk_size,
            # store the current chunk and start a new one
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from the end of the previous chunk
                # Try to find a sentence boundary in the overlap area
                sentences = re.split(r'(?<=[.!?])\s+', current_chunk)
                overlap_text = ""
                
                # Add sentences from the end until we reach the desired overlap
                for sentence in reversed(sentences):
                    if len(overlap_text) + len(sentence) < overlap:
                        overlap_text = sentence + " " + overlap_text
                    else:
                        break
                
                current_chunk = overlap_text + paragraph
            else:
                # Add the paragraph to the current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_pdf(self, pdf_path: str, chunk_size: int = 1000, overlap: int = 200) -> bool:
        """
        Process a PDF file to extract text and create chunks.
        
        Args:
            pdf_path (str): Path to the PDF file
            chunk_size (int): Target size of each chunk
            overlap (int): Overlap between consecutive chunks
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        self.logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text from PDF
        text = self._extract_text_with_pypdf2(pdf_path)
        
        # If PyPDF2 failed, try pdfplumber
        if not text and self.pdfplumber is not None:
            self.logger.info("PyPDF2 extraction failed. Trying pdfplumber...")
            text = self._extract_text_with_pdfplumber(pdf_path)
        
        if not text:
            self.logger.error("Failed to extract text from PDF")
            return False
        
        # Clean the text
        cleaned_text = self._clean_text(text)
        self.document_text = cleaned_text
        
        # Create metadata
        file_size = os.path.getsize(pdf_path)
        file_name = os.path.basename(pdf_path)
        self.document_metadata = {
            "file_name": file_name,
            "file_size": file_size,
            "text_length": len(cleaned_text)
        }
        
        # Create chunks
        self.document_chunks = self._chunk_text(cleaned_text, chunk_size, overlap)
        self.document_metadata["chunk_count"] = len(self.document_chunks)
        
        self.logger.info(f"Successfully processed PDF with {len(self.document_chunks)} chunks")
        return True
    
    def embed_pdf_file(self, pdf_path: str, chunk_size: int = 1000, overlap: int = 200) -> Dict[str, Any]:
        """
        Process a PDF file and generate embeddings for the chunks.
        
        Args:
            pdf_path (str): Path to the PDF file
            chunk_size (int): Target size of each chunk
            overlap (int): Overlap between consecutive chunks
            
        Returns:
            Dict[str, Any]: Dictionary with text chunks and embeddings
        """
        # Process the PDF
        if not self.process_pdf(pdf_path, chunk_size, overlap):
            return {"success": False, "error": "Failed to process PDF"}
        
        # Check if we have a model
        if self.model is None:
            return {
                "success": False, 
                "error": "Embedding model not available",
                "chunks": self.document_chunks
            }
        
        # Generate embeddings
        self.logger.info(f"Generating embeddings for {len(self.document_chunks)} chunks")
        try:
            embeddings = self.model.encode(self.document_chunks, show_progress_bar=True)
            self.document_embeddings = embeddings
            
            result = {
                "success": True,
                "chunks": self.document_chunks,
                "embeddings": embeddings,
                "metadata": self.document_metadata
            }
            
            return result
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return {
                "success": False,
                "error": f"Embedding generation failed: {e}",
                "chunks": self.document_chunks
            }
    
    def embed_pdf_bytes(self, pdf_bytes: bytes, chunk_size: int = 1000, overlap: int = 200) -> Dict[str, Any]:
        """
        Process PDF bytes and generate embeddings for the chunks.
        
        Args:
            pdf_bytes (bytes): PDF content as bytes
            chunk_size (int): Target size of each chunk
            overlap (int): Overlap between consecutive chunks
            
        Returns:
            Dict[str, Any]: Dictionary with text chunks and embeddings
        """
        self.logger.info(f"Processing PDF bytes ({len(pdf_bytes)/1024:.2f} KB)")
        
        # Extract text from PDF bytes
        text = self._extract_text_with_pypdf2(pdf_bytes)
        
        # If PyPDF2 failed, try pdfplumber
        if not text and self.pdfplumber is not None:
            self.logger.info("PyPDF2 extraction failed. Trying pdfplumber...")
            text = self._extract_text_with_pdfplumber(pdf_bytes)
        
        if not text:
            self.logger.error("Failed to extract text from PDF bytes")
            return {"success": False, "error": "Text extraction failed"}
        
        # Clean the text
        cleaned_text = self._clean_text(text)
        self.document_text = cleaned_text
        
        # Create metadata
        self.document_metadata = {
            "file_size": len(pdf_bytes),
            "text_length": len(cleaned_text)
        }
        
        # Create chunks
        self.document_chunks = self._chunk_text(cleaned_text, chunk_size, overlap)
        self.document_metadata["chunk_count"] = len(self.document_chunks)
        
        # Check if we have a model
        if self.model is None:
            return {
                "success": False, 
                "error": "Embedding model not available",
                "chunks": self.document_chunks
            }
        
        # Generate embeddings
        self.logger.info(f"Generating embeddings for {len(self.document_chunks)} chunks")
        try:
            embeddings = self.model.encode(self.document_chunks, show_progress_bar=True)
            self.document_embeddings = embeddings
            
            result = {
                "success": True,
                "chunks": self.document_chunks,
                "embeddings": embeddings,
                "metadata": self.document_metadata
            }
            
            return result
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return {
                "success": False,
                "error": f"Embedding generation failed: {e}",
                "chunks": self.document_chunks
            }
    
    def embed_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> Dict[str, Any]:
        """
        Process raw text and generate embeddings for the chunks.
        
        Args:
            text (str): Raw text to process
            chunk_size (int): Target size of each chunk
            overlap (int): Overlap between consecutive chunks
            
        Returns:
            Dict[str, Any]: Dictionary with text chunks and embeddings
        """
        if not text:
            return {"success": False, "error": "Empty text provided"}
        
        self.logger.info(f"Processing text ({len(text)/1024:.2f} KB)")
        
        # Clean the text
        cleaned_text = self._clean_text(text)
        self.document_text = cleaned_text
        
        # Create metadata
        self.document_metadata = {
            "text_length": len(cleaned_text)
        }
        
        # Create chunks
        self.document_chunks = self._chunk_text(cleaned_text, chunk_size, overlap)
        self.document_metadata["chunk_count"] = len(self.document_chunks)
        
        # Check if we have a model
        if self.model is None:
            return {
                "success": False, 
                "error": "Embedding model not available",
                "chunks": self.document_chunks
            }
        
        # Generate embeddings
        self.logger.info(f"Generating embeddings for {len(self.document_chunks)} chunks")
        try:
            embeddings = self.model.encode(self.document_chunks, show_progress_bar=True)
            self.document_embeddings = embeddings
            
            result = {
                "success": True,
                "chunks": self.document_chunks,
                "embeddings": embeddings,
                "metadata": self.document_metadata
            }
            
            return result
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return {
                "success": False,
                "error": f"Embedding generation failed: {e}",
                "chunks": self.document_chunks
            }
    
    def semantic_search(self, 
                        query: str, 
                        embeddings: Optional[np.ndarray] = None, 
                        chunks: Optional[List[str]] = None,
                        top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search to find relevant chunks for a query.
        
        Args:
            query (str): The search query
            embeddings (np.ndarray, optional): Precomputed embeddings (uses document_embeddings if None)
            chunks (List[str], optional): Text chunks (uses document_chunks if None)
            top_k (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of search results with chunks and scores
        """
        if self.model is None:
            self.logger.error("Embedding model not available")
            return []
        
        # Use provided embeddings or document_embeddings
        if embeddings is None:
            if not self.document_embeddings:
                self.logger.error("No embeddings available for search")
                return []
            embeddings = self.document_embeddings
        
        # Use provided chunks or document_chunks
        if chunks is None:
            if not self.document_chunks:
                self.logger.error("No chunks available for search")
                return []
            chunks = self.document_chunks
        
        # Ensure embeddings and chunks have the same length
        if len(chunks) != len(embeddings):
            self.logger.error(f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})")
            return []
        
        # Encode query
        query_embedding = self.model.encode(query)
        
        # Compute cosine similarity
        embeddings_array = np.array(embeddings)
        similarities = np.dot(embeddings_array, query_embedding) / (
            np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding) + 1e-10
        )
        
        # Get top_k results
        top_indices = np.argsort(-similarities)[:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "chunk": chunks[idx],
                "score": float(similarities[idx]),
                "index": int(idx)
            })
        
        return results

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize PDFEmbedder
    embedder = PDFEmbedder(model_name="all-MiniLM-L6-v2")
    
    # Process a PDF file
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            sys.exit(1)
            
        print(f"Processing PDF: {pdf_path}")
        result = embedder.embed_pdf_file(pdf_path)
        
        if result.get("success", False):
            print(f"Successfully processed PDF with {len(result['chunks'])} chunks")
            
            # Example search
            query = "What is the main topic of this document?"
            print(f"\nSearching for: '{query}'")
            
            search_results = embedder.semantic_search(
                query=query,
                embeddings=result["embeddings"],
                chunks=result["chunks"],
                top_k=3
            )
            
            print("\nTop 3 results:")
            for i, res in enumerate(search_results):
                print(f"\n[{i+1}] Score: {res['score']:.4f}")
                print("-" * 40)
                print(res["chunk"][:200] + "...")
        else:
            print(f"Processing failed: {result.get('error', 'Unknown error')}")
    else:
        print("Usage: python pdf_embedder.py [path_to_pdf]")