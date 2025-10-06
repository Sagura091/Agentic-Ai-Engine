"""
Text normalization and hashing utilities for deduplication and cross-script matching.

This module provides production-grade normalization that enables:
- Deduplication across Unicode variations
- Cross-script matching (Latin/Cyrillic/Greek)
- Currency and number normalization
- Consistent hashing for content-addressable storage
"""

import hashlib
import unicodedata
import re
from typing import Dict, Optional


# Homoglyph mapping for common look-alike characters
# Maps visually similar characters to canonical forms
HOMOGLYPH_MAP = {
    # Cyrillic to Latin
    'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H', 'О': 'O', 'Р': 'P', 'С': 'C', 'Т': 'T', 'Х': 'X',
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x',
    
    # Greek to Latin
    'Α': 'A', 'Β': 'B', 'Ε': 'E', 'Ζ': 'Z', 'Η': 'H', 'Ι': 'I', 'Κ': 'K', 'Μ': 'M', 'Ν': 'N', 'Ο': 'O',
    'Ρ': 'P', 'Τ': 'T', 'Υ': 'Y', 'Χ': 'X',
    'α': 'a', 'β': 'b', 'ε': 'e', 'ι': 'i', 'κ': 'k', 'ο': 'o', 'ρ': 'p', 'τ': 't', 'υ': 'y', 'χ': 'x',
    
    # Common confusables
    '０': '0', '１': '1', '２': '2', '３': '3', '４': '4', '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',  # Fullwidth
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',  # Superscript
    '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4', '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',  # Subscript
    
    # Quotes and dashes
    ''': "'", ''': "'", '"': '"', '"': '"', '‚': ',', '„': '"',
    '–': '-', '—': '-', '―': '-', '−': '-',
    
    # Special spaces
    '\u00A0': ' ',  # Non-breaking space
    '\u2003': ' ',  # Em space
    '\u2009': ' ',  # Thin space
    '\u200B': '',   # Zero-width space
    '\u200C': '',   # Zero-width non-joiner
    '\u200D': '',   # Zero-width joiner
    '\uFEFF': '',   # Zero-width no-break space
}


# Currency symbol to ISO 4217 code mapping
CURRENCY_MAP = {
    '$': 'USD',
    '€': 'EUR',
    '£': 'GBP',
    '¥': 'JPY',
    '₹': 'INR',
    '₽': 'RUB',
    '₩': 'KRW',
    '₪': 'ILS',
    '₱': 'PHP',
    '₦': 'NGN',
    '₡': 'CRC',
    '₨': 'PKR',
    '฿': 'THB',
    '₫': 'VND',
    '₴': 'UAH',
    '₵': 'GHS',
    '＄': 'USD',  # Fullwidth dollar
    '￡': 'GBP',  # Fullwidth pound
    '￥': 'JPY',  # Fullwidth yen
}


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode text to NFKC form.
    
    NFKC (Normalization Form KC) performs compatibility decomposition
    followed by canonical composition. This handles:
    - Combining characters (é vs e + ́)
    - Compatibility characters (ﬁ vs fi)
    - Width variants (fullwidth vs halfwidth)
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if not text:
        return text
    
    # Apply NFKC normalization
    normalized = unicodedata.normalize('NFKC', text)
    
    return normalized


def apply_homoglyph_mapping(text: str) -> str:
    """
    Map visually similar characters to canonical forms.
    
    This enables matching across different scripts (Latin/Cyrillic/Greek)
    and prevents homoglyph-based spoofing.
    
    Args:
        text: Input text
        
    Returns:
        Text with homoglyphs mapped to canonical forms
    """
    if not text:
        return text
    
    # Build translation table
    translation_table = str.maketrans(HOMOGLYPH_MAP)
    
    # Apply mapping
    mapped = text.translate(translation_table)
    
    return mapped


def normalize_currency(text: str) -> str:
    """
    Normalize currency symbols to ISO 4217 codes.
    
    Converts currency symbols ($, €, £, etc.) to standard codes
    (USD, EUR, GBP) for consistent matching.
    
    Args:
        text: Input text
        
    Returns:
        Text with currency symbols replaced by codes
    """
    if not text:
        return text
    
    result = text
    
    # Replace currency symbols with codes
    # Process in order of specificity (longer symbols first)
    for symbol, code in sorted(CURRENCY_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        # Match currency symbol followed by digits or space
        # e.g., "$100" -> "USD 100", "$ 100" -> "USD 100"
        pattern = re.escape(symbol) + r'(?=\s*\d)'
        result = re.sub(pattern, code + ' ', result)
        
        # Also match digits followed by currency symbol
        # e.g., "100$" -> "100 USD"
        pattern = r'(\d)\s*' + re.escape(symbol) + r'(?!\d)'
        result = re.sub(pattern, r'\1 ' + code, result)
    
    return result


def normalize_numbers(text: str) -> str:
    """
    Normalize number formatting.
    
    Removes thousands separators (commas) and normalizes decimal points
    for consistent matching.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized numbers
    """
    if not text:
        return text
    
    # Remove thousands separators (commas between digits)
    # e.g., "1,000,000" -> "1000000"
    result = re.sub(r'(\d),(\d{3})', r'\1\2', text)
    
    # Handle multiple groups (e.g., "1,000,000")
    # Repeat until no more matches
    while re.search(r'(\d),(\d{3})', result):
        result = re.sub(r'(\d),(\d{3})', r'\1\2', result)
    
    return result


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace.
    
    - Converts all whitespace to single spaces
    - Removes leading/trailing whitespace
    - Collapses multiple spaces to single space
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    if not text:
        return text
    
    # Replace all whitespace with single space
    result = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    result = result.strip()
    
    return result


def normalize_text(text: str, 
                   apply_unicode: bool = True,
                   apply_homoglyphs: bool = True,
                   apply_currency: bool = True,
                   apply_numbers: bool = True,
                   apply_whitespace: bool = True,
                   lowercase: bool = True) -> str:
    """
    Apply full text normalization pipeline.
    
    This is the main normalization function that applies all transformations
    in the correct order for maximum consistency.
    
    Args:
        text: Input text
        apply_unicode: Apply Unicode NFKC normalization
        apply_homoglyphs: Apply homoglyph mapping
        apply_currency: Normalize currency symbols
        apply_numbers: Normalize number formatting
        apply_whitespace: Normalize whitespace
        lowercase: Convert to lowercase
        
    Returns:
        Fully normalized text
    """
    if not text:
        return text
    
    result = text
    
    # Apply transformations in order
    if apply_unicode:
        result = normalize_unicode(result)
    
    if apply_homoglyphs:
        result = apply_homoglyph_mapping(result)
    
    if apply_currency:
        result = normalize_currency(result)
    
    if apply_numbers:
        result = normalize_numbers(result)
    
    if lowercase:
        result = result.lower()
    
    if apply_whitespace:
        result = normalize_whitespace(result)
    
    return result


def compute_sha256(text: str) -> str:
    """
    Compute SHA-256 hash of text.
    
    Args:
        text: Input text
        
    Returns:
        Hexadecimal SHA-256 hash
    """
    if not text:
        return hashlib.sha256(b'').hexdigest()
    
    # Encode to UTF-8 and hash
    hash_obj = hashlib.sha256(text.encode('utf-8'))
    return hash_obj.hexdigest()


def compute_content_sha(content: str) -> str:
    """
    Compute content hash for exact deduplication.
    
    This hash is computed on the raw content without normalization,
    allowing exact duplicate detection.
    
    Args:
        content: Raw content
        
    Returns:
        SHA-256 hash of content
    """
    return compute_sha256(content)


def compute_norm_text_sha(content: str) -> str:
    """
    Compute normalized text hash for fuzzy deduplication.
    
    This hash is computed on normalized content, allowing detection
    of duplicates across Unicode variations, currency formats, etc.
    
    Args:
        content: Raw content
        
    Returns:
        SHA-256 hash of normalized content
    """
    normalized = normalize_text(content)
    return compute_sha256(normalized)


def compute_hashes(content: str) -> Dict[str, str]:
    """
    Compute all hashes for content.
    
    Convenience function that computes both raw and normalized hashes.
    
    Args:
        content: Content to hash
        
    Returns:
        Dictionary with 'content_sha' and 'norm_text_sha'
    """
    return {
        'content_sha': compute_content_sha(content),
        'norm_text_sha': compute_norm_text_sha(content)
    }


def are_duplicates(content1: str, content2: str, use_normalized: bool = True) -> bool:
    """
    Check if two pieces of content are duplicates.
    
    Args:
        content1: First content
        content2: Second content
        use_normalized: Use normalized hash (fuzzy) vs raw hash (exact)
        
    Returns:
        True if contents are duplicates
    """
    if use_normalized:
        return compute_norm_text_sha(content1) == compute_norm_text_sha(content2)
    else:
        return compute_content_sha(content1) == compute_content_sha(content2)

