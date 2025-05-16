# core/pdf_extractor.py
import os
import io
import tempfile
import logging
from typing import Dict, Any, Union, Tuple
from datetime import datetime
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_bytes: bytes) -> Tuple[str, Dict[int, str]]:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_bytes: The PDF file bytes
        
    Returns:
        Tuple containing:
        - Full text content
        - Dictionary mapping page numbers to text content
    """
    text = ""
    text_by_page = {}
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_path = tmp_file.name
    
    try:
        # Using PyMuPDF for text extraction
        doc = fitz.open(tmp_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            text += page_text + "\n\n"
            text_by_page[page_num+1] = page_text
        doc.close()
        
    except Exception as e:
        logger.error(f"Error extracting text with PyMuPDF: {e}")
        raise
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass
    
    return text, text_by_page

def get_pdf_page_image(pdf_bytes: bytes, page_num: int = 0) -> bytes:
    """
    Get a specific page of a PDF as an image.
    
    Args:
        pdf_bytes: PDF file bytes
        page_num: 0-based page number to extract
        
    Returns:
        Image bytes in PNG format
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_path = tmp_file.name
    
    try:
        # Open PDF document and render page
        doc = fitz.open(tmp_path)
        
        # Check page bounds
        if page_num < 0 or page_num >= len(doc):
            raise ValueError(f"Page {page_num+1} does not exist in PDF with {len(doc)} pages")
        
        # Render page
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Zoom factor of 2
        img_bytes = pix.tobytes("png")
        
        doc.close()
        return img_bytes
    
    except Exception as e:
        logger.error(f"Error rendering PDF page: {e}")
        raise
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass