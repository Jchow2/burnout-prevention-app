"""
src/preprocessing/merger.py

Merges Glassdoor and Reddit data into a single unified DataFrame matching the
18-column schema expected by the ML model and persona pipeline.

Design: open normalizer registry — add new sources by registering a
normalizer function, no changes to core merge logic required.

Guardrail: source column must be in {"reddit", "glassdoor"} — any other value
raises ValueError to prevent accidental reintroduction of removed sources.
"""

import logging
from pathlib import Path
from typing import Callable

import pandas as pd # type: ignore

logger = logging.getLogger(__name__)

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import DATA_PROCESSED  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Unified schema — all sources must produce at least these columns
# ---------------------------------------------------------------------------

BASE_COLUMNS = [
    "review_id",
    "text",
    "source",
    "subreddit",
    "employee_job_title",
    "employee_status",
    "company_name",
    "rating_date",
    "rating_overall",
]


# ---------------------------------------------------------------------------
# Source-specific normalizers
# ---------------------------------------------------------------------------

def _normalize_glassdoor(df: pd.DataFrame) -> pd.DataFrame:         
    """Map Glassdoor columns to the unified schema."""
    out = pd.DataFrame()
    out["review_id"]          = df.get("review_id",          pd.Series(dtype=str))
    out["text"]               = df.get("text",               "")
    out["source"]             = "glassdoor"
    out["subreddit"]          = None
    out["employee_job_title"] = df.get("employee_job_title", "Unknown")
    out["employee_status"]    = df.get("employee_status",    "Unknown")
    out["company_name"]       = df.get("company_name",       "Amazon")
    out["rating_date"]        = df.get("rating_date",        pd.NaT)
    out["rating_overall"]     = df.get("rating_overall",     pd.NA)
    return out


def _normalize_reddit(df: pd.DataFrame) -> pd.DataFrame:
    """Map Reddit post columns to the unified schema."""
    out = pd.DataFrame()
    out["review_id"]          = df.get("review_id",     pd.Series(dtype=str))
    out["text"]               = df.get("text",          "")
    out["source"]             = "reddit"
    out["subreddit"]          = df.get("subreddit",     None)
    out["employee_job_title"] = "Unknown"
    out["employee_status"]    = "Unknown"
    out["company_name"]       = df.get("company_name",  "Various")
    out["rating_date"]        = df.get("created_date",  pd.NaT)
    out["rating_overall"]     = pd.NA
    return out


# ---------------------------------------------------------------------------
# Normalizer registry — register new sources here, nothing else changes
# ---------------------------------------------------------------------------

ALLOWED_SOURCES = {"reddit", "glassdoor"}

# Maps source_name → normalizer function
# To add a new source: define _normalize_<source>() above and add it here.
NORMALIZERS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "glassdoor": _normalize_glassdoor,
    "reddit":    _normalize_reddit,
}


def assert_no_youtube(df: pd.DataFrame, stage: str = "merge") -> None:
    """Raise ValueError if any row has source outside ALLOWED_SOURCES."""
    if "source" not in df.columns:
        return
    bad = df.loc[~df["source"].isin(ALLOWED_SOURCES), "source"].unique()
    if len(bad):
        raise ValueError(
            f"[{stage}] Disallowed source(s) detected: {list(bad)}. "
            f"Only {ALLOWED_SOURCES} are permitted. "
            "YouTube ingestion has been removed — check your data inputs."
        )


def register_source(name: str, normalizer: Callable[[pd.DataFrame], pd.DataFrame]):
    """
    Register a new data source normalizer at runtime.

    Args:
        name: Source name string (e.g. "indeed", "twitter")
        normalizer: Function that takes a raw DataFrame and returns one
                    conforming to BASE_COLUMNS.

    Example:
        from src.preprocessing.merger import register_source

        def _normalize_indeed(df):
            out = pd.DataFrame()
            out["review_id"] = df["id"]
            out["text"]      = df["review_body"]
            out["source"]    = "indeed"
            # ... fill remaining BASE_COLUMNS
            return out

        register_source("indeed", _normalize_indeed)
    """
    NORMALIZERS[name] = normalizer
    logger.info(f"Registered new source normalizer: '{name}'")


# ---------------------------------------------------------------------------
# Core merge function
# ---------------------------------------------------------------------------

def merge_sources(
    dataframes: dict[str, pd.DataFrame],
    deduplicate: bool = True,
) -> pd.DataFrame:
    """
    Merge multiple source DataFrames into a single unified DataFrame.

    Args:
        dataframes: Dict mapping source name → DataFrame.
            Example: {"glassdoor": gd_df, "reddit": rd_df}
        deduplicate: Drop duplicate texts across sources.

    Returns:
        Unified DataFrame with base columns (before labeling).
        Labeling columns (polarity, keywords, etc.) are added by
        running the labeler on the merged output.
    """
    normalized = []

    for source_name, df in dataframes.items():
        if df is None or df.empty:
            logger.info(f"Skipping {source_name}: empty DataFrame")
            continue

        normalizer = NORMALIZERS.get(source_name)
        if normalizer is None:
            logger.warning(
                f"No normalizer registered for source '{source_name}'. "
                f"Attempting passthrough — check BASE_COLUMNS alignment."
            )
            norm_df = df.copy()
        else:
            try:
                norm_df = normalizer(df)
            except Exception as e:
                logger.error(
                    f"Normalizer for '{source_name}' raised an error: {e}. "
                    f"Skipping this source."
                )
                continue

        # Ensure all base columns exist (fill missing with None)
        for col in BASE_COLUMNS:
            if col not in norm_df.columns:
                norm_df[col] = None

        logger.info(f"Normalized {source_name}: {len(norm_df)} rows")
        normalized.append(norm_df)

    if not normalized:
        logger.warning("No data to merge — all sources empty or failed.")
        return pd.DataFrame(columns=BASE_COLUMNS)

    merged = pd.concat(normalized, ignore_index=True)

    assert_no_youtube(merged, stage="merge_sources")

    if deduplicate:
        before = len(merged)
        merged = merged.drop_duplicates(subset=["text"], keep="first")
        removed = before - len(merged)
        if removed:
            logger.info(f"Deduplication removed {removed} duplicate texts.")

    logger.info(
        f"Merged dataset: {len(merged)} reviews from "
        f"{merged['source'].nunique()} sources"
    )

    # Source distribution log
    for src, count in merged["source"].value_counts().items():
        pct = count / len(merged) * 100
        logger.info(f"  {src}: {count} ({pct:.1f}%)")

    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_merged(df: pd.DataFrame, filename: str = "merged_reviews.csv") -> Path:
    """Save merged DataFrame to the processed data directory."""
    out_path = DATA_PROCESSED / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved merged dataset → {out_path}")
    return out_path
