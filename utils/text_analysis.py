# utils/text_analysis.py
import re
import pandas as pd
from typing import Dict, List, Any

def get_df_text_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate statistics from the text DataFrame.
    
    Args:
        df: DataFrame containing the extracted PDF text
        
    Returns:
        Dict with statistics about the text
    """
    stats = {
        "total_pages": df["page_num"].nunique(),
        "total_paragraphs": len(df[df["content_type"] == "paragraph"]),
        "total_sections": len(df[df["is_section_header"] == True]),
        "total_tables": len(df[df["content_type"] == "table"]),
        "total_characters": df["char_count"].sum(),
        "total_words": df["word_count"].sum(),
        "avg_chars_per_page": df[df["content_type"] == "page"]["char_count"].mean(),
        "avg_words_per_page": df[df["content_type"] == "page"]["word_count"].mean(),
        "avg_chars_per_paragraph": df[df["content_type"] == "paragraph"]["char_count"].mean(),
        "avg_words_per_paragraph": df[df["content_type"] == "paragraph"]["word_count"].mean(),
    }
    
    # Add some text complexity metrics
    if "content" in df.columns:
        # Get all paragraph texts joined
        all_text = " ".join(df[df["content_type"] == "paragraph"]["content"].tolist())
        
        # Calculate average sentence length
        sentences = re.split(r'[.!?]+', all_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            stats["avg_words_per_sentence"] = avg_sentence_length
            
            # Calculate long word ratio (words with >6 characters)
            all_words = re.findall(r'\b\w+\b', all_text.lower())
            long_words = [w for w in all_words if len(w) > 6]
            stats["long_word_ratio"] = len(long_words) / len(all_words) if all_words else 0
    
    return stats

def extract_pdf_sections(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Extract structured sections from the PDF DataFrame.
    
    Args:
        df: DataFrame containing the extracted PDF text
        
    Returns:
        List of dictionaries, each representing a section with its content
    """
    sections = []
    current_section = None
    
    # Sort by page and position
    sorted_df = df.sort_values(by=["page_num", "position.index"])
    
    # Filter to only paragraphs
    para_df = sorted_df[sorted_df["content_type"] == "paragraph"].copy()
    
    for _, row in para_df.iterrows():
        if row["is_section_header"] or current_section is None:
            # Start a new section
            if current_section is not None:
                sections.append(current_section)
            
            current_section = {
                "title": row["section"],
                "content": row["content"],
                "paragraphs": [row["content"]],
                "start_page": row["page_num"],
                "end_page": row["page_num"],
                "char_count": row["char_count"],
                "word_count": row["word_count"]
            }
        else:
            # Add to existing section
            current_section["content"] += f"\n\n{row['content']}"
            current_section["paragraphs"].append(row["content"])
            current_section["end_page"] = row["page_num"]
            current_section["char_count"] += row["char_count"]
            current_section["word_count"] += row["word_count"]
    
    # Add the last section
    if current_section is not None:
        sections.append(current_section)
    
    return sections

def get_word_frequencies(df: pd.DataFrame, min_length=3, max_words=100, stop_words=None) -> Dict[str, int]:
    """
    Calculate word frequencies from the text DataFrame.
    
    Args:
        df: DataFrame containing the extracted PDF text
        min_length: Minimum word length to include
        max_words: Maximum number of words to return
        stop_words: List of stop words to exclude
        
    Returns:
        Dict mapping words to their frequencies
    """
    if stop_words is None:
        # Common English stop words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                     'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for', 'with', 
                     'by', 'about', 'against', 'between', 'into', 'through', 'during', 
                     'before', 'after', 'above', 'below', 'from', 'up', 'down', 'of', 'off', 
                     'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 
                     'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 
                     'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 
                     'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 
                     'just', 'don', 'should', 'now'}
    
    # Get all paragraph text
    if 'paragraph' in df['content_type'].values:
        text = ' '.join(df[df['content_type'] == 'paragraph']['content'].astype(str).tolist())
    else:
        text = ' '.join(df['content'].astype(str).tolist())
    
    # Find all words, convert to lowercase
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    # Filter by length and stop words
    filtered_words = [word for word in words if len(word) >= min_length and word not in stop_words]
    
    # Count frequencies
    word_counts = {}
    for word in filtered_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency and limit to max_words
    sorted_counts = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:max_words])
    
    return sorted_counts

def search_pdf_dataframe(df: pd.DataFrame, search_term: str, case_sensitive: bool = False) -> pd.DataFrame:
    """
    Search for text in the PDF DataFrame.
    
    Args:
        df: DataFrame containing the extracted PDF text
        search_term: Term to search for
        case_sensitive: Whether search should be case sensitive
        
    Returns:
        DataFrame containing only the rows that match the search term
    """
    if not case_sensitive:
        # Case insensitive search
        mask = df["content"].str.contains(search_term, case=False, na=False)
    else:
        # Case sensitive search
        mask = df["content"].str.contains(search_term, na=False)
    
    return df[mask].copy()