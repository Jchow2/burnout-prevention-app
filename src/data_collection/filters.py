"""
Quality filters: subreddit allowlist, text hygiene, deduplication.

All functions are pure DataFrame -> DataFrame transformations so they can be
composed in any order.  The recommended order is:
  1. filter_by_subreddit   (primary gate — drop irrelevant rows early)
  2. filter_text_quality   (drop deleted/empty/URL-only/too-short)
  3. deduplicate           (by id, then by text hash)
  4. apply_date_filter     (optional)
  5. sample_per_subreddit  (optional, for balanced datasets)
"""
from __future__ import annotations

import logging
import re
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

_DELETED_VALUES = frozenset({"[deleted]", "[removed]", "[deleted by user]", "nan", ""})
_URL_ONLY_RE = re.compile(r"^\s*(https?://\S+\s*)+$")
_URL_RE = re.compile(r"https?://\S+")


def _clean_sub_name(name: str) -> str:
    return re.sub(r"^[rR]/", "", str(name)).strip().lower()


def filter_by_subreddit(df: pd.DataFrame, allowlist: Iterable[str]) -> pd.DataFrame:
    """
    Keep only rows whose 'subreddit' column matches the allowlist.

    This is the PRIMARY gate that makes the pipeline subreddit-first.
    Comparison is case-insensitive and strips leading 'r/' if present.
    """
    allowed = {_clean_sub_name(s) for s in allowlist}
    if "subreddit" not in df.columns:
        logger.warning("No 'subreddit' column found — subreddit filter skipped")
        return df

    mask = df["subreddit"].apply(_clean_sub_name).isin(allowed)
    out = df[mask].copy()
    logger.info("Subreddit filter: %d -> %d rows kept (allowlist: %s)", len(df), len(out), sorted(allowed))
    return out


def filter_text_quality(
    df: pd.DataFrame,
    min_length: int = 30,
    remove_url_only: bool = True,
) -> pd.DataFrame:
    """
    Drop rows where text is:
      - a deleted/removed placeholder
      - empty or whitespace-only
      - composed only of URLs
      - shorter than min_length after stripping URLs
    """
    if "text" not in df.columns:
        return df

    mask = pd.Series(True, index=df.index)

    # Deleted / removed / empty
    mask &= ~df["text"].str.strip().str.lower().isin(_DELETED_VALUES)

    # URL-only rows
    if remove_url_only:
        mask &= ~df["text"].str.match(_URL_ONLY_RE, na=False)

    # Too short after stripping inline URLs
    stripped_len = df["text"].str.replace(_URL_RE, "", regex=True).str.strip().str.len()
    mask &= stripped_len >= min_length

    out = df[mask].copy()
    logger.info("Text quality filter: %d -> %d rows kept (min_len=%d)", len(df), len(out), min_length)
    return out


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate by 'id' column when available; fall back to exact 'text' match.
    Within each duplicate group the first occurrence is kept.
    """
    if df.empty:
        return df
    before = len(df)
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="first")
        logger.info("Dedup by id: %d -> %d rows", before, len(df))
    else:
        df = df.drop_duplicates(subset=["text"], keep="first")
        logger.info("Dedup by text: %d -> %d rows", before, len(df))
    return df


def apply_date_filter(
    df: pd.DataFrame,
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    """Filter rows to the [date_from, date_to] inclusive range using created_utc."""
    if "created_utc" not in df.columns or (date_from is None and date_to is None):
        return df
    if date_from:
        df = df[df["created_utc"] >= pd.Timestamp(date_from, tz="UTC")]
    if date_to:
        df = df[df["created_utc"] <= pd.Timestamp(date_to, tz="UTC")]
    logger.info("Date filter [%s -> %s]: %d rows remain", date_from, date_to, len(df))
    return df


def filter_min_score(df: pd.DataFrame, min_score: int) -> pd.DataFrame:
    """Drop rows where score is below min_score.  Useful for pruning low-quality comments."""
    if "score" not in df.columns:
        return df
    before = len(df)
    df = df[pd.to_numeric(df["score"], errors="coerce").fillna(0) >= min_score].copy()
    logger.info("Min-score filter (>=%d): %d -> %d rows", min_score, before, len(df))
    return df


def sample_per_subreddit(df: pd.DataFrame, max_per_subreddit: int) -> pd.DataFrame:
    """Cap each subreddit at max_per_subreddit rows, sampling randomly (seed=42)."""
    if df.empty or "subreddit" not in df.columns:
        return df
    return (
        df.groupby("subreddit", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), max_per_subreddit), random_state=42))
        .reset_index(drop=True)
    )
