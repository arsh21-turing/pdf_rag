# core/dataframe_converter.py
"""
Core routines for turning a PDF into a structured DataFrame *and*
(optionally) persisting the results in Streamlit’s session state.
"""

import os
import re
import io
import tempfile
import logging
from datetime import datetime
from typing import Dict, Tuple, Any

import fitz               # PyMuPDF
import pdfplumber
import pandas as pd
import streamlit as st

from utils.text_analysis import (
    get_df_text_statistics,
    extract_pdf_sections,
    get_word_frequencies,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────────
# Low-level extractor – pure data, no Streamlit side effects
# ────────────────────────────────────────────────────────────────────────────────
def extract_pdf_to_dataframe(
    pdf_bytes: bytes,
    *,
    extract_paragraphs: bool = True,
    min_paragraph_length: int = 15,
    extract_metadata: bool = True,
    detect_sections: bool = True,
    table_extraction: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parse a PDF and return (dataframe, metadata).

    The dataframe contains rows for pages, paragraphs and tables; the
    metadata dict holds document info and extraction statistics.
    """
    # Write received bytes to a temp file so PyMuPDF / pdfplumber can open it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    data_rows = []
    meta: Dict[str, Any] = {
        "document_info": {},
        "extraction_stats": {
            "total_pages": 0,
            "total_paragraphs": 0,
            "total_sections": 0,
            "total_tables": 0,
            "extraction_time": datetime.now().isoformat(),
        },
        "table_data": {},
    }

    section_patterns = [
        r"^[A-Z][A-Z0-9\s\.\-,]{3,}(?:\n|\r\n)",                       # ALL-CAP header
        r"^\s*(?:[1-9][0-9]*\.)+\s+[A-Z]",                              # 1. / 1.1 / 1.1.1
        r"^\s*[IVXLCDM]+\.\s+[A-Z]",                                    # I.  II.
        r"^\s*[A-Z]\.\s+[A-Z]",                                         # A.  B.
        r"^\s*(?:INTRODUCTION|SUMMARY|CONCLUSION|BACKGROUND|"
        r"METHODOLOGY|RESULTS|DISCUSSION|APPENDIX)",
    ]

    try:
        doc = fitz.open(tmp_path)
        meta["extraction_stats"]["total_pages"] = len(doc)

        if extract_metadata:
            doc_info = doc.metadata or {}
            meta["document_info"] = {
                "title": doc_info.get("title", ""),
                "author": doc_info.get("author", ""),
                "subject": doc_info.get("subject", ""),
                "creator": doc_info.get("creator", ""),
                "producer": doc_info.get("producer", ""),
                "creation_date": doc_info.get("creationDate", ""),
                "modification_date": doc_info.get("modDate", ""),
                "page_count": len(doc),
            }

        current_section = "DOCUMENT START"
        section_count = 0
        para_count = 0

        for page_idx, page in enumerate(doc):
            page_text = page.get_text()

            # raw page record
            data_rows.append(
                {
                    "page_num": page_idx + 1,
                    "content_type": "page",
                    "section": current_section,
                    "paragraph_num": None,
                    "content": page_text,
                    "char_count": len(page_text),
                    "word_count": len(re.findall(r"\b\w+\b", page_text)),
                    "position": {"page": page_idx + 1, "index": 0},
                    "is_section_header": False,
                }
            )

            if not extract_paragraphs:
                continue

            paragraphs = re.split(r"\n\s*\n", page_text)
            page_para_idx = 0

            for para in paragraphs:
                para = para.strip()
                if len(para) < min_paragraph_length:
                    continue

                page_para_idx += 1
                para_count += 1
                is_header = False

                if detect_sections:
                    for pat in section_patterns:
                        if re.search(pat, para):
                            current_section = para.split("\n", 1)[0]
                            is_header = True
                            section_count += 1
                            break

                data_rows.append(
                    {
                        "page_num": page_idx + 1,
                        "content_type": "paragraph",
                        "section": current_section,
                        "paragraph_num": page_para_idx,
                        "content": para,
                        "char_count": len(para),
                        "word_count": len(re.findall(r"\b\w+\b", para)),
                        "position": {"page": page_idx + 1, "index": para_count},
                        "is_section_header": is_header,
                    }
                )

        meta["extraction_stats"]["total_paragraphs"] = para_count
        meta["extraction_stats"]["total_sections"] = section_count

        # ── Table extraction ─────────────────────────────────────────────
        if table_extraction:
            table_counter = 0
            with pdfplumber.open(tmp_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    for tbl_idx, table in enumerate(page.extract_tables() or []):
                        table_counter += 1
                        try:
                            df_table = pd.DataFrame(table[1:], columns=table[0]).fillna("")
                            table_id = f"page_{page_idx + 1}_table_{tbl_idx + 1}"
                            meta["table_data"][table_id] = df_table.to_dict("records")

                            # store placeholder row so users can see where table lives
                            tbl_text = "\n".join(
                                ", ".join(str(c) for c in row if c) for row in table if any(row)
                            )
                            data_rows.append(
                                {
                                    "page_num": page_idx + 1,
                                    "content_type": "table",
                                    "section": current_section,
                                    "paragraph_num": None,
                                    "content": f"[TABLE {tbl_idx + 1}]\n{tbl_text}",
                                    "char_count": len(tbl_text),
                                    "word_count": len(re.findall(r"\b\w+\b", tbl_text)),
                                    "position": {"page": page_idx + 1, "table": tbl_idx + 1},
                                    "is_section_header": False,
                                }
                            )
                        except Exception as exc:
                            logger.warning("Table %s extraction error: %s", table_id, exc)

            meta["extraction_stats"]["total_tables"] = table_counter

        doc.close()
    except Exception as exc:
        meta["extraction_error"] = str(exc)
        logger.exception("PDF extraction failed")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return pd.DataFrame(data_rows), meta


# ────────────────────────────────────────────────────────────────────────────────
# Helper that *also* writes to Streamlit session_state
# ────────────────────────────────────────────────────────────────────────────────
def extract_to_dataframe_and_save(pdf_bytes: bytes) -> bool:
    """
    Run `extract_pdf_to_dataframe()` and stash the results into
    `st.session_state`.  Returns True on success.
    """
    try:
        df, meta = extract_pdf_to_dataframe(
            pdf_bytes,
            extract_paragraphs=st.session_state.extract_paragraphs,
            min_paragraph_length=st.session_state.min_paragraph_length,
            extract_metadata=True,
            detect_sections=st.session_state.detect_sections,
            table_extraction=st.session_state.table_extraction,
        )

        st.session_state.text_df = df
        st.session_state.text_metadata = meta
        st.session_state.doc_statistics = get_df_text_statistics(df)
        st.session_state.detected_sections = extract_pdf_sections(df)
        st.session_state.word_frequencies = get_word_frequencies(df)
        st.session_state.extracted_to_df = True
        return True

    except Exception as exc:
        logger.error("extract_to_dataframe_and_save: %s", exc, exc_info=True)
        st.session_state.error_message = f"Error extracting to DataFrame: {exc}"
        return False
