"""
Ingestion layer: reads raw Kaggle CSV / JSONL / Parquet files into DataFrames.

Each loader returns a raw, unmodified DataFrame.  Schema normalisation happens
in normalize.py; filtering happens in filters.py.

Adding a new Kaggle dataset
---------------------------
1. Drop your file(s) under  data/raw/<dataset_name>/
2. Call load_file(path) to get a raw DataFrame, or use load_kaggle_dataset()
   with an explicit field_map if the column names are non-standard.
3. Pass the result to the appropriate normalize_* function, or use
   normalize.normalize() with a custom field_map.
4. Register the path in config.yaml under inputs.extra_datasets.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Column name candidates (used by normalize.py too) ─────────────────────────
TEXT_CANDIDATES      = ["selftext", "body", "text", "content", "post_text"]
SUBREDDIT_CANDIDATES = ["subreddit", "subreddit_name_prefixed", "subreddit_name", "sub"]
DATE_CANDIDATES      = ["created_utc", "created", "timestamp", "date", "created_at", "date_created"]
ID_CANDIDATES        = ["id", "post_id", "comment_id", "link_id"]
TITLE_CANDIDATES     = ["title", "post_title", "name"]
SCORE_CANDIDATES     = ["score", "upvotes", "ups"]
AUTHOR_CANDIDATES    = ["author", "username", "user"]


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column present in df (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


# ── Generic file loader ────────────────────────────────────────────────────────

def load_file(path: str | Path, chunk_size: int = 50_000) -> pd.DataFrame:
    """Auto-detect format from extension and load into a DataFrame."""
    path = Path(path)
    if not path.exists():
        logger.warning("File not found: %s", path)
        return pd.DataFrame()

    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix in {".jsonl", ".ndjson"}:
            return pd.read_json(path, lines=True)
        if suffix == ".json":
            return pd.read_json(path)
        if suffix == ".csv":
            chunks = [
                chunk
                for chunk in pd.read_csv(
                    path, chunksize=chunk_size, low_memory=False, on_bad_lines="skip"
                )
            ]
            return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    except Exception as exc:
        logger.error("Failed to load %s: %s", path, exc)
        return pd.DataFrame()

    logger.warning("Unrecognised extension '%s' for %s", suffix, path)
    return pd.DataFrame()


# ── Named loaders ─────────────────────────────────────────────────────────────

def load_antiwork_posts(path: str | Path, chunk_size: int = 50_000) -> pd.DataFrame:
    """Load r/antiwork posts CSV (~153 MB)."""
    df = load_file(path, chunk_size)
    if not df.empty:
        logger.info("Antiwork posts raw: %d rows, cols: %s", len(df), list(df.columns))
    return df


def load_antiwork_comments(path: str | Path, chunk_size: int = 50_000) -> pd.DataFrame:
    """Load r/antiwork comments CSV (~2.8 GB).  Skipped if file absent."""
    path = Path(path)
    if not path.exists():
        logger.warning("Antiwork comments not found at %s — skipping", path)
        return pd.DataFrame()
    df = load_file(path, chunk_size)
    if not df.empty:
        logger.info("Antiwork comments raw: %d rows", len(df))
    return df


def load_reddit_sentiment_posts(path: str | Path, chunk_size: int = 50_000) -> pd.DataFrame:
    """Load vijayj0shi reddit_sentiment posts_df.csv (post-centric)."""
    df = load_file(path, chunk_size)
    if not df.empty:
        logger.info("Reddit-sentiment posts raw: %d rows, cols: %s", len(df), list(df.columns))
    return df


def load_reddit_sentiment_comments(path: str | Path, chunk_size: int = 50_000) -> pd.DataFrame:
    """Load vijayj0shi reddit_sentiment comments.csv."""
    df = load_file(path, chunk_size)
    if not df.empty:
        logger.info("Reddit-sentiment comments raw: %d rows", len(df))
    return df


def load_reddit_sentiment_user_posts(path: str | Path, chunk_size: int = 50_000) -> pd.DataFrame:
    """
    Load vijayj0shi reddit_sentiment user_posts.csv (user-centric).

    WARNING: This file tracks users across ALL their subreddits, which is
    why the old pipeline picked up r/relationships, gaming subs, etc.
    Always apply filter_by_subreddit() immediately after loading this.
    """
    df = load_file(path, chunk_size)
    if not df.empty:
        logger.info("Reddit-sentiment user_posts raw: %d rows", len(df))
    return df


# ── Kaggle enrichment helper ───────────────────────────────────────────────────

def load_kaggle_dataset(
    folder: str | Path,
    field_map: dict[str, str],
    record_type: str = "post",
    chunk_size: int = 50_000,
) -> pd.DataFrame:
    """
    Load any Kaggle Reddit dataset placed in <folder> and apply a field map.

    The result is a partially-normalised DataFrame ready for normalize.normalize().

    Parameters
    ----------
    folder      Path to a dataset folder or directly to a file.
                If a folder, the first .csv/.jsonl/.parquet found is used.
    field_map   Dict mapping source column names → normalised field names.
                Required target key: 'text'.
                Optional targets: 'id', 'subreddit', 'title', 'author',
                                  'created_utc', 'score', 'url'.
    record_type 'post' or 'comment'

    Example — antiwork posts
    ------------------------
    load_kaggle_dataset(
        "data/raw/antiwork",
        field_map={
            "id":          "id",
            "title":       "title",
            "selftext":    "text",
            "author":      "author",
            "subreddit":   "subreddit",
            "score":       "score",
            "created_utc": "created_utc",
            "url":         "url",
        },
        record_type="post",
    )

    Example — reddit_sentiment posts_df
    ------------------------------------
    load_kaggle_dataset(
        "data/raw/reddit_sentiment/posts_df.csv",
        field_map={
            "id":          "id",
            "title":       "title",
            "selftext":    "text",
            "subreddit":   "subreddit",
            "score":       "score",
            "created_utc": "created_utc",
        },
        record_type="post",
    )
    """
    folder = Path(folder)
    if folder.is_file():
        raw = load_file(folder, chunk_size)
    else:
        candidates = (
            list(folder.glob("*.csv"))
            + list(folder.glob("*.jsonl"))
            + list(folder.glob("*.parquet"))
        )
        if not candidates:
            logger.warning("No data files found in %s", folder)
            return pd.DataFrame()
        raw = load_file(candidates[0], chunk_size)

    if raw.empty:
        return raw

    out = pd.DataFrame()
    for src_col, tgt_col in field_map.items():
        if src_col in raw.columns:
            out[tgt_col] = raw[src_col]
        else:
            logger.warning("load_kaggle_dataset: column '%s' not in dataset %s", src_col, folder)

    out["type"] = record_type
    out["source_file"] = str(folder)
    logger.info("Kaggle dataset '%s': %d rows mapped", folder.name, len(out))
    return out
