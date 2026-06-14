"""
Schema normalisation: maps raw DataFrames from any source into the common schema.

Common schema
─────────────
  id           str            original Reddit post/comment ID
  type         str            'post' or 'comment'
  subreddit    str            lowercase, no 'r/' prefix
  title        str            post title; empty string for comments
  text         str            selftext (posts) or body (comments)
  author       str            username
  created_utc  datetime[UTC]  nullable UTC timestamp
  score        Int64          upvotes (nullable)
  url          str            post URL or empty string
  source_file  str            path of originating raw file
"""
from __future__ import annotations

import hashlib
import logging
import re

import pandas as pd

from .ingest import (
    find_col,
    TEXT_CANDIDATES,
    SUBREDDIT_CANDIDATES,
    DATE_CANDIDATES,
    ID_CANDIDATES,
    TITLE_CANDIDATES,
    SCORE_CANDIDATES,
    AUTHOR_CANDIDATES,
)

logger = logging.getLogger(__name__)

SCHEMA_COLUMNS = [
    "id", "type", "subreddit", "title", "text",
    "author", "created_utc", "score", "url", "source_file",
]


def _make_id(series_vals: list) -> str:
    key = "|".join(str(v) for v in series_vals)
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _strip_r_prefix(val) -> str:
    if isinstance(val, str):
        return re.sub(r"^[rR]/", "", val).strip().lower()
    return ""


def _parse_utc(series: pd.Series) -> pd.Series:
    """Handle both Unix epoch integers and ISO date strings."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() > len(series) * 0.5:
        return pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def normalize(
    df: pd.DataFrame,
    record_type: str,
    source_file: str = "",
    field_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Normalise any raw Reddit DataFrame into the common 10-column schema.

    If field_map is provided it takes priority over auto-detection.
    field_map keys = source column names, values = target schema names.
    """
    if df.empty:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    if field_map:
        rename = {k: v for k, v in field_map.items() if k in df.columns}
        df = df.rename(columns=rename)

    out = pd.DataFrame(index=df.index)

    # id
    id_col = find_col(df, ID_CANDIDATES)
    if id_col:
        out["id"] = df[id_col].astype(str)
    else:
        out["id"] = [
            _make_id([df.iloc[i, j] for j in range(min(3, len(df.columns)))])
            for i in range(len(df))
        ]

    # type
    out["type"] = df["type"].astype(str) if "type" in df.columns else record_type

    # subreddit
    sub_col = find_col(df, SUBREDDIT_CANDIDATES)
    out["subreddit"] = df[sub_col].apply(_strip_r_prefix) if sub_col else ""

    # title
    title_col = find_col(df, TITLE_CANDIDATES)
    out["title"] = df[title_col].fillna("").astype(str) if title_col else ""

    # text — prefer explicit 'text' column (already renamed by field_map), then candidates
    if "text" in df.columns:
        out["text"] = df["text"].fillna("").astype(str)
    else:
        text_col = find_col(df, TEXT_CANDIDATES)
        out["text"] = df[text_col].fillna("").astype(str) if text_col else ""

    # author
    author_col = find_col(df, AUTHOR_CANDIDATES)
    out["author"] = df[author_col].fillna("").astype(str) if author_col else ""

    # created_utc
    date_col = find_col(df, DATE_CANDIDATES)
    out["created_utc"] = _parse_utc(df[date_col]) if date_col else pd.NaT

    # score
    score_col = find_col(df, SCORE_CANDIDATES)
    out["score"] = (
        pd.to_numeric(df[score_col], errors="coerce").astype("Int64")
        if score_col
        else pd.array([pd.NA] * len(df), dtype="Int64")
    )

    # url
    out["url"] = df["url"].fillna("").astype(str) if "url" in df.columns else ""

    # source_file
    out["source_file"] = (
        df["source_file"].astype(str) if "source_file" in df.columns else source_file
    )

    return out[SCHEMA_COLUMNS].reset_index(drop=True)


# ── Pre-built field maps for known datasets ────────────────────────────────────
#
# Typical Reddit Kaggle export column names.  These are used as the default
# mappings; override by passing field_map= to normalize() directly.

ANTIWORK_POSTS_MAP = {
    "id":              "id",
    "title":           "title",
    "selftext":        "text",
    "author":          "author",
    "subreddit":       "subreddit",
    "subreddit.name":  "subreddit",   # Kaggle antiwork export uses dot-notation
    "score":           "score",
    "created_utc":     "created_utc",
    "url":             "url",
}

ANTIWORK_COMMENTS_MAP = {
    "id":              "id",
    "body":            "text",
    "author":          "author",
    "subreddit":       "subreddit",
    "subreddit.name":  "subreddit",   # same dot-notation issue
    "score":           "score",
    "created_utc":     "created_utc",
    "link_id":         "url",
}

REDDIT_SENTIMENT_POSTS_MAP = {
    "id":          "id",
    "title":       "title",
    "selftext":    "text",
    "author":      "author",
    "subreddit":   "subreddit",
    "score":       "score",
    "created_utc": "created_utc",
    "url":         "url",
}

REDDIT_SENTIMENT_COMMENTS_MAP = {
    "id":          "id",
    "body":        "text",
    "author":      "author",
    "subreddit":   "subreddit",
    "score":       "score",
    "created_utc": "created_utc",
}

REDDIT_SENTIMENT_USER_POSTS_MAP = {
    "id":          "id",
    "title":       "title",
    "selftext":    "text",
    "author":      "author",
    "subreddit":   "subreddit",
    "score":       "score",
    "created_utc": "created_utc",
    "url":         "url",
}


def normalize_antiwork_posts(df: pd.DataFrame, source_file: str = "") -> pd.DataFrame:
    return normalize(df, "post", source_file, ANTIWORK_POSTS_MAP)

def normalize_antiwork_comments(df: pd.DataFrame, source_file: str = "") -> pd.DataFrame:
    return normalize(df, "comment", source_file, ANTIWORK_COMMENTS_MAP)

def normalize_reddit_sentiment_posts(df: pd.DataFrame, source_file: str = "") -> pd.DataFrame:
    return normalize(df, "post", source_file, REDDIT_SENTIMENT_POSTS_MAP)

def normalize_reddit_sentiment_comments(df: pd.DataFrame, source_file: str = "") -> pd.DataFrame:
    return normalize(df, "comment", source_file, REDDIT_SENTIMENT_COMMENTS_MAP)

def normalize_reddit_sentiment_user_posts(df: pd.DataFrame, source_file: str = "") -> pd.DataFrame:
    return normalize(df, "post", source_file, REDDIT_SENTIMENT_USER_POSTS_MAP)
