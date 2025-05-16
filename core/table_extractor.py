# core/table_extractor.py
import os
import io
import tempfile
import logging
from typing import Dict, List, Any, Union
import pandas as pd
import pdfplumber

logger = logging.getLogger(__name__)

def extract_tables_from_pdf(pdf_bytes: bytes) -> Dict[int, List[pd.DataFrame]]:
    """
    Extract tables from a PDF file.
    
    Args:
        pdf_bytes: PDF file bytes
        
    Returns:
        Dictionary mapping page numbers to lists of pandas DataFrames
    """
    tables_by_page = {}
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_path = tmp_file.name
    
    try:
        # Extract tables using pdfplumber
        with pdfplumber.open(tmp_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                raw_tables = page.extract_tables()
                
                if raw_tables:
                    processed_tables = []
                    
                    for table in raw_tables:
                        # Skip empty tables
                        if not table or not any(row for row in table if any(cell for cell in row)):
                            continue
                        
                        # Process the table into a DataFrame
                        try:
                            # Assume first row is header
                            # Replace None with empty string in headers
                            headers = [str(h) if h is not None else "" for h in table[0]]
                            
                            # Create DataFrame (skip header row)
                            if len(table) > 1:
                                df = pd.DataFrame(table[1:], columns=headers)
                                
                                # Replace None with empty string in data
                                df = df.fillna("")
                                
                                processed_tables.append(df)
                        except Exception as e:
                            logger.error(f"Error converting table to DataFrame: {e}")
                            # Add an empty DataFrame as a placeholder
                            processed_tables.append(pd.DataFrame())
                    
                    if processed_tables:
                        tables_by_page[page_num+1] = processed_tables
    
    except Exception as e:
        logger.error(f"Error extracting tables: {e}")
        raise
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass
    
    return tables_by_page