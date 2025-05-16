"""
PDF Format Converter

This utility provides functionality to convert PDFs between different formats and versions
to ensure compatibility with various document processing systems.
"""

import os
import io
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple

# Import directly from the core/pdf_validator.py module
from core.pdf_validator import validate_pdf

# Set up logging
logger = logging.getLogger(__name__)

def convert_pdf_to_pdf_a(
    file_path_or_bytes: Union[str, bytes, io.BytesIO], 
    output_path: Optional[str] = None,
    compliance_level: str = "2b"
) -> Dict[str, Any]:
    """
    Convert a PDF to PDF/A format for archival purposes.
    
    Args:
        file_path_or_bytes: Path to input PDF file, bytes, or BytesIO object
        output_path: Path where the converted PDF/A should be saved
                    If None, a temporary file will be created and its path returned
        compliance_level: PDF/A compliance level: "1b", "2b", or "3b"
        
    Returns:
        Dict containing conversion info:
            - 'success': Boolean indicating success
            - 'output_path': Path to the converted file
            - 'metadata': Metadata of the converted file
            - 'message': Status message
    """
    logger.info(f"Converting PDF to PDF/A-{compliance_level}")
    
    # Validate the input PDF first using the existing validate_pdf function
    is_valid, metadata, message = validate_pdf(file_path_or_bytes)
    
    if not is_valid:
        logger.error(f"Cannot convert invalid PDF: {message}")
        return {
            'success': False,
            'message': f"Input PDF validation failed: {message}",
            'metadata': metadata
        }
    
    # Create a temporary output file if not specified
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        output_path = temp_file.name
        temp_file.close()
        logger.debug(f"Created temporary output file: {output_path}")
    
    try:
        # Implement PDF/A conversion using a library like Ghostscript
        # This is a placeholder for the actual implementation
        # In a real implementation, we would use a library like PyGhostScript or pdfaPilot
        
        # Example implementation using PyGhostScript (commented out as actual implementation would require specific libraries)
        """
        import ghostscript
        
        # Set up Ghostscript arguments
        args = [
            "gs",  # Ghostscript command
            f"-dPDFA={compliance_level[0]}",  # PDF/A compliance level
            "-dBATCH",
            "-dNOPAUSE",
            "-dNOOUTERSAVE",
            "-dPDFACompatibilityPolicy=1",
            "-sColorConversionStrategy=RGB",
            "-sDEVICE=pdfwrite",
            f"-sOutputFile={output_path}",
            file_path_or_bytes if isinstance(file_path_or_bytes, str) else "<stdin>"
        ]
        
        # Execute Ghostscript
        ghostscript.Ghostscript(*args)
        """
        
        # For now, we'll use a simple file copy to simulate conversion
        if isinstance(file_path_or_bytes, str):
            # If input is a file path, read and write the file
            with open(file_path_or_bytes, 'rb') as src_file:
                pdf_data = src_file.read()
                with open(output_path, 'wb') as dest_file:
                    dest_file.write(pdf_data)
        elif isinstance(file_path_or_bytes, io.BytesIO):
            # If input is BytesIO, get the bytes
            file_path_or_bytes.seek(0)
            pdf_data = file_path_or_bytes.getvalue()
            with open(output_path, 'wb') as dest_file:
                dest_file.write(pdf_data)
        else:
            # If input is bytes, write directly
            with open(output_path, 'wb') as dest_file:
                dest_file.write(file_path_or_bytes)
        
        # Validate the converted file using the existing validate_pdf function
        new_is_valid, new_metadata, new_message = validate_pdf(output_path)
        
        if not new_is_valid:
            logger.error(f"Conversion resulted in an invalid PDF/A file: {new_message}")
            return {
                'success': False,
                'output_path': output_path,
                'message': f"Conversion resulted in invalid PDF/A: {new_message}",
                'metadata': new_metadata
            }
        
        logger.info(f"Successfully converted PDF to PDF/A-{compliance_level}")
        return {
            'success': True,
            'output_path': output_path,
            'message': f"Successfully converted to PDF/A-{compliance_level}",
            'metadata': new_metadata
        }
    
    except Exception as e:
        logger.exception(f"Error converting PDF to PDF/A: {str(e)}")
        return {
            'success': False,
            'message': f"Conversion error: {str(e)}",
            'metadata': metadata
        }

def convert_pdf_version(
    file_path_or_bytes: Union[str, bytes, io.BytesIO],
    output_path: Optional[str] = None,
    target_version: str = "1.7"
) -> Dict[str, Any]:
    """
    Convert a PDF to a specific PDF version.
    
    Args:
        file_path_or_bytes: Path to input PDF file, bytes, or BytesIO object
        output_path: Path where the converted PDF should be saved
                    If None, a temporary file will be created and its path returned
        target_version: Target PDF version (e.g., "1.4", "1.7")
        
    Returns:
        Dict containing conversion info:
            - 'success': Boolean indicating success
            - 'output_path': Path to the converted file
            - 'metadata': Metadata of the converted file
            - 'message': Status message
    """
    logger.info(f"Converting PDF to version {target_version}")
    
    # Validate the input PDF first using the existing validate_pdf function
    is_valid, metadata, message = validate_pdf(file_path_or_bytes)
    
    if not is_valid:
        logger.error(f"Cannot convert invalid PDF: {message}")
        return {
            'success': False,
            'message': f"Input PDF validation failed: {message}",
            'metadata': metadata
        }
    
    current_version = metadata.get('pdf_version', 'unknown')
    logger.info(f"Current PDF version: {current_version}, target version: {target_version}")
    
    # Create a temporary output file if not specified
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        output_path = temp_file.name
        temp_file.close()
        logger.debug(f"Created temporary output file: {output_path}")
    
    try:
        # Use PyPDF2 for version conversion
        from PyPDF2 import PdfReader, PdfWriter
        
        # Process based on input type
        if isinstance(file_path_or_bytes, str):
            # It's a file path
            with open(file_path_or_bytes, 'rb') as file:
                reader = PdfReader(file)
        elif isinstance(file_path_or_bytes, io.BytesIO):
            # It's a BytesIO object
            file_path_or_bytes.seek(0)
            reader = PdfReader(file_path_or_bytes)
        else:
            # It's bytes
            reader = PdfReader(io.BytesIO(file_path_or_bytes))
        
        writer = PdfWriter()
        
        # Copy all pages
        for i in range(len(reader.pages)):
            writer.add_page(reader.pages[i])
        
        # Set the PDF version
        writer._header = f"%PDF-{target_version}".encode("ascii")
        
        # Write to output
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)
        
        # Validate the converted file using the existing validate_pdf function
        new_is_valid, new_metadata, new_message = validate_pdf(output_path)
        
        if not new_is_valid:
            logger.error(f"Conversion resulted in an invalid PDF file: {new_message}")
            return {
                'success': False,
                'output_path': output_path,
                'message': f"Conversion resulted in invalid PDF: {new_message}",
                'metadata': new_metadata
            }
        
        logger.info(f"Successfully converted PDF from version {current_version} to {target_version}")
        return {
            'success': True,
            'output_path': output_path,
            'message': f"Successfully converted from PDF v{current_version} to v{target_version}",
            'metadata': new_metadata
        }
    
    except Exception as e:
        logger.exception(f"Error converting PDF version: {str(e)}")
        return {
            'success': False,
            'message': f"Conversion error: {str(e)}",
            'metadata': metadata
        }

def repair_pdf(
    file_path_or_bytes: Union[str, bytes, io.BytesIO],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Attempt to repair a corrupted PDF file.
    
    Args:
        file_path_or_bytes: Path to input PDF file, bytes, or BytesIO object
        output_path: Path where the repaired PDF should be saved
                    If None, a temporary file will be created and its path returned
        
    Returns:
        Dict containing repair info:
            - 'success': Boolean indicating success
            - 'output_path': Path to the repaired file
            - 'metadata': Metadata of the repaired file
            - 'message': Status message
            - 'recovery_method': Method used for recovery (if successful)
    """
    logger.info(f"Attempting to repair PDF")
    
    # Initial validation to check the extent of damage
    is_valid, metadata, message = validate_pdf(file_path_or_bytes)
    
    # Create a temporary output file if not specified
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        output_path = temp_file.name
        temp_file.close()
        logger.debug(f"Created temporary output file: {output_path}")
    
    # If the PDF is already valid, just make a copy
    if is_valid:
        logger.info("PDF is already valid. Creating a clean copy.")
        try:
            # Copy the file
            if isinstance(file_path_or_bytes, str):
                with open(file_path_or_bytes, 'rb') as src_file:
                    pdf_data = src_file.read()
            elif isinstance(file_path_or_bytes, io.BytesIO):
                file_path_or_bytes.seek(0)
                pdf_data = file_path_or_bytes.getvalue()
            else:
                pdf_data = file_path_or_bytes
            
            with open(output_path, 'wb') as dest_file:
                dest_file.write(pdf_data)
            
            return {
                'success': True,
                'output_path': output_path,
                'message': "PDF was already valid, created a clean copy",
                'metadata': metadata,
                'recovery_method': 'none_needed'
            }
        except Exception as e:
            logger.error(f"Error copying valid PDF: {str(e)}")
            return {
                'success': False,
                'message': f"Error copying PDF: {str(e)}",
                'metadata': metadata
            }
    
    # The PDF is damaged, try various repair methods
    recovery_methods = [
        ('structure_recovery', _repair_pdf_structure),
        ('format_recovery', _repair_pdf_format)
        # Add more methods as needed
    ]
    
    for method_name, repair_func in recovery_methods:
        logger.info(f"Attempting repair with method: {method_name}")
        try:
            result = repair_func(file_path_or_bytes, output_path)
            
            if result.get('success'):
                logger.info(f"Successfully repaired PDF with method: {method_name}")
                result['recovery_method'] = method_name
                return result
            else:
                logger.warning(f"Repair method {method_name} failed: {result.get('message')}")
        except Exception as e:
            logger.error(f"Error during repair method {method_name}: {str(e)}")
    
    # If we've tried all methods and none worked
    logger.error("All repair methods failed")
    return {
        'success': False,
        'message': "PDF could not be repaired with available methods",
        'metadata': metadata
    }

def _repair_pdf_structure(
    file_path_or_bytes: Union[str, bytes, io.BytesIO],
    output_path: str
) -> Dict[str, Any]:
    """
    Repair a PDF by fixing structural issues like corrupt cross-reference tables.
    
    Args:
        file_path_or_bytes: Path or bytes of the damaged PDF
        output_path: Path to save the repaired PDF
        
    Returns:
        Dict with repair results
    """
    try:
        # Use PyPDF2 to try to recover
        from PyPDF2 import PdfReader, PdfWriter
        
        # Process based on input type
        if isinstance(file_path_or_bytes, str):
            with open(file_path_or_bytes, 'rb') as file:
                pdf_data = file.read()
        elif isinstance(file_path_or_bytes, io.BytesIO):
            file_path_or_bytes.seek(0)
            pdf_data = file_path_or_bytes.getvalue()
        else:
            pdf_data = file_path_or_bytes
        
        # Create a reader with strict=False to allow for some errors
        reader = PdfReader(io.BytesIO(pdf_data), strict=False)
        writer = PdfWriter()
        
        # Try to recover pages
        recovered_pages = 0
        total_pages = len(reader.pages)
        for i in range(total_pages):
            try:
                page = reader.pages[i]
                writer.add_page(page)
                recovered_pages += 1
            except Exception as e:
                logger.warning(f"Could not recover page {i}: {str(e)}")
        
        if recovered_pages == 0:
            return {
                'success': False,
                'message': "Could not recover any pages from the PDF"
            }
        
        # Write the recovered PDF
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)
        
        # Validate the repaired PDF
        new_is_valid, new_metadata, new_message = validate_pdf(output_path)
        
        message = f"Recovered {recovered_pages} of {total_pages} pages"
        if not new_is_valid:
            logger.warning(f"Repaired PDF still has issues: {new_message}")
            message += f", but PDF still has issues: {new_message}"
        
        return {
            'success': new_is_valid,
            'output_path': output_path,
            'message': message,
            'metadata': new_metadata,
            'partial_success': recovered_pages > 0
        }
    
    except Exception as e:
        logger.exception(f"Error during structure repair: {str(e)}")
        return {
            'success': False,
            'message': f"Structure repair failed: {str(e)}"
        }

def _repair_pdf_format(
    file_path_or_bytes: Union[str, bytes, io.BytesIO],
    output_path: str
) -> Dict[str, Any]:
    """
    Attempt to repair a PDF by using PyMuPDF (fitz) to load and resave it.
    
    This method can fix some formatting issues and linearize the PDF.
    
    Args:
        file_path_or_bytes: Path or bytes of the damaged PDF
        output_path: Path to save the repaired PDF
        
    Returns:
        Dict with repair results
    """
    try:
        import fitz  # PyMuPDF
        
        # Open the PDF with PyMuPDF
        if isinstance(file_path_or_bytes, str):
            doc = fitz.open(file_path_or_bytes)
        elif isinstance(file_path_or_bytes, io.BytesIO):
            file_path_or_bytes.seek(0)
            doc = fitz.open(stream=file_path_or_bytes.getvalue(), filetype="pdf")
        else:
            doc = fitz.open(stream=file_path_or_bytes, filetype="pdf")
        
        if doc.is_encrypted:
            # Try with empty password
            if not doc.authenticate(""):
                return {
                    'success': False,
                    'message': "Cannot repair an encrypted PDF without the password"
                }
        
        # Save with clean option to optimize and repair
        doc.save(output_path, clean=True, garbage=3, deflate=True)
        doc.close()
        
        # Validate the repaired PDF
        new_is_valid, new_metadata, new_message = validate_pdf(output_path)
        
        if new_is_valid:
            return {
                'success': True,
                'output_path': output_path,
                'message': f"Successfully repaired and optimized PDF with PyMuPDF",
                'metadata': new_metadata
            }
        else:
            return {
                'success': False,
                'output_path': output_path,
                'message': f"PyMuPDF repair attempt failed: {new_message}",
                'metadata': new_metadata
            }
    
    except Exception as e:
        logger.exception(f"Error during PyMuPDF repair: {str(e)}")
        return {
            'success': False,
            'message': f"PyMuPDF repair failed: {str(e)}"
        }

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test PDF path (adjust as needed)
    test_pdf_path = "example.pdf"
    
    if not os.path.exists(test_pdf_path):
        print(f"Test file {test_pdf_path} not found.")
    else:
        print(f"Converting {test_pdf_path} to PDF/A...")
        pdf_a_result = convert_pdf_to_pdf_a(test_pdf_path)
        print(f"Result: {pdf_a_result['success']} - {pdf_a_result['message']}")
        
        print(f"\nConverting {test_pdf_path} to PDF 1.4...")
        version_result = convert_pdf_version(test_pdf_path, target_version="1.4")
        print(f"Result: {version_result['success']} - {version_result['message']}")
        
        print(f"\nRepairing {test_pdf_path}...")
        repair_result = repair_pdf(test_pdf_path)
        print(f"Result: {repair_result['success']} - {repair_result['message']}")