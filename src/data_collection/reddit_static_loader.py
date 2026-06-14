"""
src/data_collection/reddit_static_loader.py

Loads pre-downloaded Reddit data from Kaggle static CSV files.
Replaces the live PRAW collector (reddit_collector.py) for MVP use
when Reddit API credentials are not yet available.

Handles three dataset formats confirmed from dir output:

  data/raw/antiwork/
    - the-antiwork-subreddit-dataset-posts.csv   ← PRIMARY (153MB)
    - the-antiwork-subreddit-dataset-comments.csv ← SKIPPED (2.8GB, too large)

  data/raw/reddit_sentiment/
    - user_posts_preprocessed.csv                ← PRIMARY (already cleaned)
    - comments_preprocessed.csv                  ← SECONDARY (already cleaned)
    - posts_df_preprocessed.csv                  ← TERTIARY

  data/raw/glassdoor_kaggle/
    - glassdoor_reviews.csv                      ← handled by glassdoor_loader.py

This module mirrors the interface of glassdoor_loader.py so it plugs
directly into the SOURCE_RUNNERS registry in run_pipeline.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config.settings
from config.settings import DATA_RAW  # noqa: F401

import logging
import hashlib
import re  # noqa: F401
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Target subreddits for NeuroTrace — used to filter the reddit_sentiment
# dataset which covers many unrelated subreddits
# ---------------------------------------------------------------------------

TARGET_SUBREDDITS = {
    "warehouseworkers", "warehousing", "amazonfc", "fulfillmentcenter",
    "walmart", "target", "upsdrivers", "fedex", "usps",
    "manufacturing", "antiwork", "workreform", "workplace", "jobs",
    "talesfromretail", "talesfromyourserver",
}

# Burnout / workplace keyword filter — used when subreddit column is missing
WORKPLACE_KEYWORDS = [
    "warehouse", "manager", "boss", "shift", "quota", "overtime",
    "pay", "wage", "hours", "fired", "quit", "burnout", "exhausted",
    "injury", "forklift", "amazon", "ups", "fedex", "usps", "walmart",
    "worker", "employee", "job", "work", "coworker", "schedule",
]


# ---------------------------------------------------------------------------
# Antiwork dataset loader
# ---------------------------------------------------------------------------

def load_antiwork(
    data_dir: Path = config.settings.DATA_RAW / "antiwork",
    min_length: int = 30,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load the r/Antiwork Kaggle dataset (posts file only).

    The comments file (2.8GB) is intentionally skipped for MVP —
    posts are longer, more personal, and closer to review-length text.

    Args:
        data_dir:   Folder containing the antiwork CSV files.
        min_length: Minimum text character length to keep.
        max_rows:   Cap rows for testing (None = load all).

    Returns:
        Normalized DataFrame ready for merger.py.
    """
    posts_file = data_dir / "the-antiwork-subreddit-dataset-posts.csv"

    if not posts_file.exists():
        logger.warning(f"Antiwork posts file not found: {posts_file}")
        return pd.DataFrame()

    logger.info(f"Loading antiwork posts from {posts_file.name}...")

    try:
        df = pd.read_csv(
            posts_file,
            nrows=max_rows,
            low_memory=False,
            on_bad_lines="skip",
        )
    except Exception as e:
        logger.error(f"Failed to read antiwork posts: {e}")
        return pd.DataFrame()

    logger.info(f"  Loaded {len(df):,} rows, columns: {list(df.columns)}")

    # Detect text column — antiwork posts use 'selftext' or 'body' or 'title'
    text_col = _detect_text_column(df, ["selftext", "body", "text", "title"])
    if text_col is None:
        logger.error("Could not find text column in antiwork posts file.")
        return pd.DataFrame()

    # Combine title + body for richer text
    title_col = _detect_text_column(df, ["title"])
    if title_col and title_col != text_col:
        df["_combined_text"] = (
            df[title_col].fillna("").astype(str)
            + ". "
            + df[text_col].fillna("").astype(str)
        )
    else:
        df["_combined_text"] = df[text_col].fillna("").astype(str)

    # Filter out deleted/removed posts
    df = df[~df["_combined_text"].str.lower().isin(
        ["[deleted]", "[removed]", ".", ""]
    )].copy()

    # Minimum length filter
    df = df[df["_combined_text"].str.len() >= min_length].copy()

    return _normalize_reddit(df, text_col="_combined_text", subreddit="antiwork")


# ---------------------------------------------------------------------------
# Reddit sentiment dataset loader
# ---------------------------------------------------------------------------

def load_reddit_sentiment(
    data_dir: Path = config.settings.DATA_RAW / "reddit_sentiment",
    min_length: int = 30,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load the vijayj0shi Reddit sentiment Kaggle dataset.

    Loads user_posts_preprocessed.csv and comments_preprocessed.csv,
    filters to workplace-relevant subreddits or keyword matches,
    and merges into one DataFrame.

    Args:
        data_dir:   Folder containing the reddit_sentiment CSV files.
        min_length: Minimum text character length to keep.
        max_rows:   Cap rows per file for testing.

    Returns:
        Normalized DataFrame ready for merger.py.
    """
    files_to_load = [
        data_dir / "user_posts_preprocessed.csv",
        data_dir / "comments_preprocessed.csv",
        data_dir / "posts_df_preprocessed.csv",
    ]

    dfs = []
    for f in files_to_load:
        if not f.exists():
            logger.warning(f"  File not found, skipping: {f.name}")
            continue
        try:
            df = pd.read_csv(
                f,
                nrows=max_rows,
                low_memory=False,
                on_bad_lines="skip",
            )
            logger.info(f"  Loaded {f.name}: {len(df):,} rows")
            dfs.append((f.name, df))
        except Exception as e:
            logger.error(f"  Failed to read {f.name}: {e}")

    if not dfs:
        logger.warning("No reddit_sentiment files loaded.")
        return pd.DataFrame()

    normalized = []
    for fname, df in dfs:
        text_col = _detect_text_column(
            df, ["body", "selftext", "text", "content", "post_text", "comment"]
        )
        sub_col = _detect_subreddit_column(df)

        if text_col is None:
            logger.warning(f"  No text column found in {fname}, skipping.")
            continue

        # Filter to workplace-relevant content
        if sub_col:
            mask = df[sub_col].str.lower().isin(TARGET_SUBREDDITS)
            df_filtered = df[mask].copy()
            logger.info(
                f"  {fname}: {len(df_filtered):,} / {len(df):,} rows "
                f"match target subreddits"
            )
        else:
            # No subreddit column — fall back to keyword matching
            keyword_pattern = "|".join(WORKPLACE_KEYWORDS)
            mask = df[text_col].str.lower().str.contains(
                keyword_pattern, na=False, regex=True
            )
            df_filtered = df[mask].copy()
            logger.info(
                f"  {fname}: {len(df_filtered):,} / {len(df):,} rows "
                f"match workplace keywords (no subreddit column)"
            )

        if df_filtered.empty:
            continue

        # Minimum length filter
        df_filtered = df_filtered[
            df_filtered[text_col].str.len() >= min_length
        ].copy()

        subreddit_val = (
            df_filtered[sub_col].str.lower()
            if sub_col
            else "reddit_sentiment"
        )

        norm = _normalize_reddit(
            df_filtered,
            text_col=text_col,
            subreddit=subreddit_val,
        )
        normalized.append(norm)

    if not normalized:
        return pd.DataFrame()

    combined = pd.concat(normalized, ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"], keep="first")
    logger.info(
        f"Reddit sentiment combined: {len(combined):,} unique workplace rows"
    )
    return combined


# ---------------------------------------------------------------------------
# Combined loader — called by run_pipeline.py SOURCE_RUNNERS
# ---------------------------------------------------------------------------

def load_all_reddit_static(
    max_rows_per_file: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load and merge all static Reddit sources.
    This is the function SOURCE_RUNNERS calls in run_pipeline.py.

    Args:
        max_rows_per_file: Set a small number (e.g. 1000) for fast testing.

    Returns:
        Merged, deduplicated DataFrame of all Reddit sources.
    """
    logger.info("Loading static Reddit datasets...")

    antiwork_df = load_antiwork(max_rows=max_rows_per_file)
    sentiment_df = load_reddit_sentiment(max_rows=max_rows_per_file)

    dfs = [df for df in [antiwork_df, sentiment_df] if not df.empty]

    if not dfs:
        logger.warning("All Reddit static sources returned empty DataFrames.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"], keep="first")

    logger.info(f"Total Reddit static records: {len(combined):,}")

    # Log subreddit breakdown
    if "subreddit" in combined.columns:
        top = combined["subreddit"].value_counts().head(10)
        for sub, count in top.items():
            logger.info(f"  r/{sub}: {count:,}")

    return combined


# ---------------------------------------------------------------------------
# Shared normalizer — maps any reddit DataFrame to merger.py base schema
# ---------------------------------------------------------------------------

def _normalize_reddit(
    df: pd.DataFrame,
    text_col: str,
    subreddit,  # str or Series
) -> pd.DataFrame:
    """Map a raw Reddit DataFrame to the 9-column base schema."""
    out = pd.DataFrame()

    # Generate deterministic review IDs from text content
    out["review_id"] = df[text_col].apply(
        lambda t: f"rd_{hashlib.md5(str(t)[:120].encode()).hexdigest()[:12]}"
    )
    out["text"]               = df[text_col].astype(str).str.strip()
    out["source"]             = "reddit"
    out["subreddit"]          = (
        subreddit if isinstance(subreddit, str)
        else subreddit.reset_index(drop=True)
    )
    out["employee_job_title"] = "Unknown"
    out["employee_status"]    = "Unknown"
    out["company_name"]       = _infer_company(df, text_col)
    out["rating_date"]        = _detect_date(df)
    out["rating_overall"]     = pd.NA

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _detect_text_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first candidate column name that exists in df."""
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def _detect_subreddit_column(df: pd.DataFrame) -> Optional[str]:
    """Return the subreddit column name if present."""
    for c in df.columns:
        if c.lower() in {"subreddit", "sub", "subreddit_name"}:
            return c
    return None


def _detect_date(df: pd.DataFrame) -> pd.Series:
    """Try to find and parse a date column."""
    for c in df.columns:
        if c.lower() in {"created_utc", "created", "date", "timestamp", "date_created"}:
            try:
                if "utc" in c.lower():
                    return pd.to_datetime(df[c], unit="s", errors="coerce")
                return pd.to_datetime(df[c], errors="coerce")
            except Exception:
                pass
    return pd.Series([pd.NaT] * len(df))


def _infer_company(df: pd.DataFrame, text_col: str) -> pd.Series:
    """
    Best-effort company inference from post text.
    Returns 'Various' when no company can be detected.
    """
    company_patterns = {
        "Amazon":  r"\bamazon\b|\bafc\b|\bfulfillment center\b",
        "Walmart": r"\bwalmart\b|\bsam.s club\b",
        "UPS":     r"\bups\b|\bunited parcel\b",
        "FedEx":   r"\bfedex\b|\bfed ex\b",
        "USPS":    r"\busps\b|\bpost office\b|\bmail carrier\b",
        "Target":  r"\btarget\b",
    }
    text_lower = df[text_col].str.lower().fillna("")
    result = pd.Series(["Various"] * len(df), index=df.index)
    for company, pattern in company_patterns.items():
        mask = text_lower.str.contains(pattern, regex=True, na=False)
        result[mask] = company
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI — quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Quick test with 500 rows per file
    df = load_all_reddit_static(max_rows_per_file=500)

    if not df.empty:
        print(f"\nLoaded {len(df):,} total rows")
        print(f"Columns: {list(df.columns)}")
        print("\nSample:")
        print(df[["source", "subreddit", "company_name", "text"]].head(5).to_string())
    else:
        print("No data loaded — check file paths and column names above.")
