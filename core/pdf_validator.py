# core/pdf_validator.py
import os
import io
import logging
import tempfile
from typing import Dict, Any, Union, Tuple
import fitz  # PyMuPDF
import pdfplumber
from PyPDF2 import PdfReader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_pdf(file_path_or_bytes: Union[str, bytes, io.BytesIO]) -> Tuple[bool, Dict[str, Any], str]:
    """
    Validates a PDF file checking for corruption, password protection, and format validity.
    
    Args:
        file_path_or_bytes: Either a file path string or PDF file bytes/BytesIO object
        
    Returns:
        Tuple containing:
        - Boolean indicating if the PDF is valid
        - Dictionary with metadata and additional information
        - String message (error message if invalid, success message if valid)
    """
    # Original validation logic remains the same
    result = {
        "valid": False,
        "errors": [],
        "is_encrypted": False,
        "page_count": 0,
        "metadata": {}
    }
    
    # Determine if input is a file path or bytes
    is_file_path = isinstance(file_path_or_bytes, str)
    
    # Validate the PDF using multiple libraries to be thorough
    valid_pymupdf = False
    valid_pdfplumber = False
    valid_pypdf2 = False
    
    # 1. Check with PyMuPDF (fitz)
    try:
        if is_file_path:
            pdf_document = fitz.open(file_path_or_bytes)
        else:
            # If it's bytes or BytesIO, convert to bytes if needed
            if isinstance(file_path_or_bytes, io.BytesIO):
                file_bytes = file_path_or_bytes.getvalue()
            else:
                file_bytes = file_path_or_bytes
                
            # Open from memory stream
            pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        
        # Check if encrypted
        if pdf_document.is_encrypted:
            result["is_encrypted"] = True
            result["errors"].append("PDF is password protected")
            
            # Try with empty password (some PDFs are technically encrypted but have no password)
            try:
                if pdf_document.authenticate(""):
                    # Can access with empty password
                    result["errors"].pop()  # Remove encryption error
                    logger.info("PDF is encrypted but accessible with empty password")
                else:
                    pdf_document.close()
                    valid_pymupdf = False
            except Exception:
                pdf_document.close()
                valid_pymupdf = False
        
        # Get page count and metadata if not encrypted or if empty password worked
        if not result["is_encrypted"] or (result["is_encrypted"] and not result["errors"]):
            result["page_count"] = len(pdf_document)
            
            # Extract metadata
            metadata = pdf_document.metadata
            if metadata:
                result["metadata"] = {
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "subject": metadata.get("subject", ""),
                    "creator": metadata.get("creator", ""),
                    "producer": metadata.get("producer", ""),
                    "creation_date": metadata.get("creationDate", ""),
                    "mod_date": metadata.get("modDate", "")
                }
            
            valid_pymupdf = True
        
        pdf_document.close()
        
    except Exception as e:
        logger.warning(f"PyMuPDF validation failed: {str(e)}")
        result["errors"].append(f"PyMuPDF validation error: {str(e)}")
    
    # 2. Verify with pdfplumber (another approach)
    try:
        if is_file_path:
            with pdfplumber.open(file_path_or_bytes) as pdf:
                # If we got here, the file opened successfully
                
                # Update page count (if PyMuPDF failed)
                if result["page_count"] == 0:
                    result["page_count"] = len(pdf.pages)
                
                # Additional check - try to read content from first page
                if len(pdf.pages) > 0:
                    # Attempt to extract text from the first page
                    _ = pdf.pages[0].extract_text()
                    valid_pdfplumber = True
        else:
            # For bytes or BytesIO
            if isinstance(file_path_or_bytes, io.BytesIO):
                file_bytes_io = file_path_or_bytes
                # Reset position to start of file
                file_bytes_io.seek(0)
            else:
                file_bytes_io = io.BytesIO(file_path_or_bytes)
                
            with pdfplumber.open(file_bytes_io) as pdf:
                # If we got here, the file opened successfully
                
                # Update page count (if PyMuPDF failed)
                if result["page_count"] == 0:
                    result["page_count"] = len(pdf.pages)
                
                # Additional check - try to read content from first page
                if len(pdf.pages) > 0:
                    # Attempt to extract text from the first page
                    _ = pdf.pages[0].extract_text()
                    valid_pdfplumber = True
                    
    except Exception as e:
        logger.warning(f"pdfplumber validation failed: {str(e)}")
        result["errors"].append(f"pdfplumber validation error: {str(e)}")
    
    # 3. Final check with PyPDF2 (especially good for encryption detection)
    try:
        if is_file_path:
            with open(file_path_or_bytes, 'rb') as file:
                reader = PdfReader(file)
                # Check encryption
                if reader.is_encrypted:
                    result["is_encrypted"] = True
                    
                    # Try to decrypt with empty password
                    try:
                        reader.decrypt('')
                        # If decrypt succeeded with empty password, file is technically accessible
                    except:
                        if not result["errors"] or "password protected" not in result["errors"][0]:
                            result["errors"].append("PDF is password protected (PyPDF2)")
                
                # If we got here without exception, the file is valid
                valid_pypdf2 = True
                
                # Update page count if we still don't have it
                if result["page_count"] == 0:
                    result["page_count"] = len(reader.pages)
        else:
            # For bytes or BytesIO
            if isinstance(file_path_or_bytes, io.BytesIO):
                file_bytes_io = file_path_or_bytes
                # Reset position to start of file
                file_bytes_io.seek(0)
            else:
                file_bytes_io = io.BytesIO(file_path_or_bytes)
                
            reader = PdfReader(file_bytes_io)
            # Check encryption
            if reader.is_encrypted:
                result["is_encrypted"] = True
                
                # Try to decrypt with empty password
                try:
                    reader.decrypt('')
                    # If decrypt succeeded with empty password, file is technically accessible
                except:
                    if not result["errors"] or "password protected" not in result["errors"][0]:
                        result["errors"].append("PDF is password protected (PyPDF2)")
            
            # If we got here without exception, the file is valid
            valid_pypdf2 = True
            
            # Update page count if we still don't have it
            if result["page_count"] == 0:
                result["page_count"] = len(reader.pages)
                
    except Exception as e:
        logger.warning(f"PyPDF2 validation failed: {str(e)}")
        result["errors"].append(f"PyPDF2 validation error: {str(e)}")
    
    # Overall validation result - PDF is valid if at least one library validates it
    result["valid"] = valid_pymupdf or valid_pdfplumber or valid_pypdf2
    
    # Final checks
    if result["valid"]:
        # If valid but no pages, that's suspicious
        if result["page_count"] == 0:
            result["valid"] = False
            result["errors"].append("PDF appears valid but contains no pages")
            
        # If valid but still has errors, it's partially valid with warnings
        if result["errors"] and not result["is_encrypted"]:
            logger.warning("PDF validated but with warnings")
    
    # Prepare metadata dictionary for return
    metadata_dict = result["metadata"].copy()
    metadata_dict["page_count"] = result["page_count"]
    metadata_dict["is_encrypted"] = result["is_encrypted"]
    
    # Create message based on validation results
    is_valid = result["valid"]
    if is_valid:
        if result["is_encrypted"] and not result["errors"]:
            message = f"Valid PDF with {result['page_count']} pages (encrypted but accessible)"
        elif not result["errors"]:
            message = f"Valid PDF with {result['page_count']} pages"
        else:
            message = f"PDF is usable but with warnings: {', '.join(result['errors'])}"
    else:
        if result["errors"]:
            message = f"Invalid PDF: {', '.join(result['errors'])}"
        else:
            message = "Invalid PDF format"
    
    # Return as tuple: (is_valid, metadata_dict, message)
    return is_valid, metadata_dict, message


def is_pdf_valid(file_path_or_bytes: Union[str, bytes, io.BytesIO]) -> Tuple[bool, str]:
    """
    Simplified function to check if a PDF is valid.
    
    Args:
        file_path_or_bytes: Either a file path string or PDF bytes/BytesIO
        
    Returns:
        Tuple containing:
        - Boolean indicating if the PDF is valid
        - String message (error message if invalid, success message if valid)
    """
    is_valid, _, message = validate_pdf(file_path_or_bytes)
    return is_valid, message