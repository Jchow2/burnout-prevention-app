"""
src/preprocessing/text_cleaner.py

Shared text cleaning pipeline used across all data sources.
Normalizes text before feature extraction and model training.
"""

import re
import logging
import unicodedata

import pandas as pd # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core cleaning functions
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters (smart quotes, em-dashes, etc.)."""
    text = unicodedata.normalize("NFKD", text)
    # Replace common Unicode characters with ASCII equivalents
    replacements = {
        "\u2018": "'", "\u2019": "'",   # Smart single quotes
        "\u201c": '"', "\u201d": '"',   # Smart double quotes
        "\u2013": "-", "\u2014": "-",   # En/em dashes
        "\u2026": "...",                 # Ellipsis
        "\u00a0": " ",                  # Non-breaking space
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def remove_urls(text: str) -> str:
    return re.sub(r"https?://[^\s]+", "", text)


def remove_emails(text: str) -> str:
    return re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "", text)


def remove_reddit_formatting(text: str) -> str:
    """Strip Reddit markdown artifacts."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)   # Bold
    text = re.sub(r"\*(.+?)\*", r"\1", text)        # Italic
    text = re.sub(r"~~(.+?)~~", r"\1", text)        # Strikethrough
    text = re.sub(r"&gt;", ">", text)                # Quoted text
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text) # Markdown links
    text = re.sub(r"#{1,6}\s*", "", text)            # Headers
    return text


def remove_glassdoor_artifacts(text: str) -> str:
    """
    Remove formatting artifacts common in Glassdoor CSV exports.

    Glassdoor exports from scraping tools often include:
      - "Pros:" / "Cons:" / "Advice to Management:" labels (added by merger,
        then re-encountered if re-processing an already-merged file)
      - HTML entities left over from web scraping (&amp; &nbsp; etc.)
      - Rating strings like "5.0 out of 5 stars" or "3/5"
      - Boilerplate like "Former Employee" / "Current Employee" fragments
        that sometimes bleed into the text field
      - Repeated punctuation artifacts (-----, =====) from copy-paste exports
    """
    # Remove Pros/Cons/Advice labels (safe to strip — merger already captured them)
    text = re.sub(r"^(Pros|Cons|Advice to management)\s*:\s*", "", text, flags=re.IGNORECASE)

    # HTML entities
    text = text.replace("&amp;", "&")
    text = text.replace("&nbsp;", " ")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")

    # Star rating strings
    text = re.sub(r"\d+(\.\d+)?\s*(out of|/)\s*\d+\s*stars?", "", text, flags=re.IGNORECASE)

    # Repeated punctuation dividers
    text = re.sub(r"[-=_]{4,}", " ", text)

    # Employment status fragments that leak into text
    text = re.sub(
        r"\b(Former|Current)\s+Employee\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into single spaces."""
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text: str, source: str = "unknown") -> str:
    """
    Full cleaning pipeline for a single text string.

    Args:
        text: Raw review/post text.
        source: Data source ("glassdoor", "reddit") for source-specific cleaning.

    Returns:
        Cleaned text string.
    """
    if pd.isna(text) or not text:
        return ""

    text = str(text)
    text = normalize_unicode(text)
    text = remove_urls(text)
    text = remove_emails(text)

    if source == "reddit":
        text = remove_reddit_formatting(text)
    elif source == "glassdoor":
        text = remove_glassdoor_artifacts(text)

    text = normalize_whitespace(text)

    # Remove very short leftover strings
    if len(text.split()) < 3:
        return ""

    return text


# ---------------------------------------------------------------------------
# DataFrame-level cleaning
# ---------------------------------------------------------------------------

def clean_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    source_column: str = "source",
    min_words: int = 10,
) -> pd.DataFrame:
    """
    Apply full cleaning pipeline to a DataFrame.

    Args:
        df: Input DataFrame with a text column.
        text_column: Name of the text column.
        source_column: Name of the source column (for source-specific rules).
        min_words: Drop rows with fewer than this many words after cleaning.

    Returns:
        Cleaned DataFrame (rows may be dropped).
    """
    initial_count = len(df)
    df = df.copy()

    # Apply cleaning
    df[text_column] = df.apply(
        lambda row: clean_text(
            row.get(text_column, ""),
            source=row.get(source_column, "unknown"),
        ),
        axis=1,
    )

    # Drop empty / too-short texts
    df["_word_count"] = df[text_column].str.split().str.len()
    df = df[df["_word_count"] >= min_words].drop(columns=["_word_count"])

    # Deduplicate on text (keep first occurrence)
    df = df.drop_duplicates(subset=[text_column], keep="first")

    dropped = initial_count - len(df)
    logger.info(
        f"Text cleaning: {initial_count} → {len(df)} rows "
        f"({dropped} dropped: short/empty/duplicate)"
    )
    return df.reset_index(drop=True)
