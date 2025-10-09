"""
Text preprocessing utilities for toxicity detection.
Handles cleaning, normalization, and optional lemmatization.
"""
import re
import html
import emoji
from typing import Optional

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)
    LEMMATIZER = WordNetLemmatizer()
    NLTK_AVAILABLE = True
except ImportError:
    LEMMATIZER = None
    NLTK_AVAILABLE = False

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG = re.compile(r"#[A-Za-z0-9_]+")
PUNCT = re.compile(r"[^\w\s]")

def clean_text(t: str) -> str:
    """
    Basic text cleaning (backward compatible).
    Removes URLs, mentions, hashtags, emojis, punctuation and lowercases.
    """
    t = html.unescape(str(t or ""))
    t = URL.sub(" ", t)
    t = MENTION.sub(" ", t)
    t = HASHTAG.sub(" ", t)
    t = emoji.replace_emoji(t, replace=" ")
    t = PUNCT.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

def clean_text_advanced(t: str, lemmatize: bool = False) -> str:
    """
    Advanced text cleaning with optional lemmatization.
    
    Args:
        t: Input text
        lemmatize: If True, apply lemmatization (requires nltk)
    
    Returns:
        Cleaned text
    """
    # Apply basic cleaning
    t = clean_text(t)
    
    # Apply lemmatization if requested
    if lemmatize and NLTK_AVAILABLE and LEMMATIZER:
        words = t.split()
        words = [LEMMATIZER.lemmatize(word, pos='v') for word in words]  # Lemmatize as verbs
        words = [LEMMATIZER.lemmatize(word, pos='n') for word in words]  # Then as nouns
        t = ' '.join(words)
    elif lemmatize and not NLTK_AVAILABLE:
        print("Warning: NLTK not available, skipping lemmatization")
    
    return t

