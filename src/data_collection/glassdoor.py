"""
Glassdoor Kaggle dataset processor for the workforce sentiment pipeline.

Handles:
  - Chunked reading of the 292 MB / ~850k-row glassdoor_reviews.csv
  - PII redaction in free-text fields (emails, phone-like numbers, URLs)
  - Privacy anonymisation: firm hashing, location removal (config-driven)
  - Normalisation into the pipeline common schema + Glassdoor rating columns
  - Stable SHA-256 row IDs

Raw CSV columns (davidgauthier Kaggle dataset):
  firm, date_review, job_title, current, location,
  overall_rating, work_life_balance, culture_values, diversity_inclusion,
  career_opp, comp_benefits, senior_mgmt, recommend, ceo_approv, outlook,
  headline, pros, cons

Output schema (workforce_glassdoor_clean.parquet)
─────────────────────────────────────────────────
Common pipeline columns:
  id, type, subreddit, title, text, author, created_utc, score, url,
  source_file, source

Glassdoor-specific columns:
  firm_id, firm (if glassdoor_anonymize_firm=False),
  location (if glassdoor_keep_location=True),
  job_title, current,
  overall_rating, work_life_balance, culture_values, diversity_inclusion,
  career_opp, comp_benefits, senior_mgmt, recommend, ceo_approv, outlook
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── PII redaction patterns ─────────────────────────────────────────────────────

_EMAIL_RE    = re.compile(r"\b\S+@\S+\b")
_PHONE_RE    = re.compile(r"[\+\(]?[\d][\d\s\-\(\)\.]{8,}[\d]")   # 10+ digit patterns
_URL_RE      = re.compile(r"https?://\S+|www\.\S+")

_REDACT_EMAIL = "[EMAIL]"
_REDACT_PHONE = "[PHONE]"
_REDACT_URL   = "[URL]"

# ── Rating columns to preserve ────────────────────────────────────────────────

RATING_COLS = [
    "overall_rating", "work_life_balance", "culture_values",
    "diversity_inclusion", "career_opp", "comp_benefits",
    "senior_mgmt", "recommend", "ceo_approv", "outlook",
]


def _redact(text: str) -> str:
    """Remove emails, phone-like numbers, and URLs from a text string."""
    if not isinstance(text, str) or not text.strip():
        return text
    text = _URL_RE.sub(_REDACT_URL, text)
    text = _EMAIL_RE.sub(_REDACT_EMAIL, text)
    text = _PHONE_RE.sub(_REDACT_PHONE, text)
    return text


def _hash_firm(firm: str) -> str:
    """Stable 12-char SHA-256 prefix of the lowercased firm name."""
    return hashlib.sha256(str(firm).strip().lower().encode()).hexdigest()[:12]


def _make_row_id(firm: str, date: str, job_title: str, headline: str, pros: str, cons: str) -> str:
    """Stable 16-char SHA-256 prefix from key fields."""
    key = "|".join(str(v).strip() for v in [firm, date, job_title, headline, pros, cons])
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _combine_text(headline: str, pros: str, cons: str) -> str:
    """Build combined review text in the spec format."""
    parts = []
    h = str(headline or "").strip()
    p = str(pros     or "").strip()
    c = str(cons     or "").strip()
    if h and h.lower() not in ("nan", "none"):
        parts.append(h)
    if p and p.lower() not in ("nan", "none"):
        parts.append(f"PROS: {p}")
    if c and c.lower() not in ("nan", "none"):
        parts.append(f"CONS: {c}")
    return "\n".join(parts)


def _parse_date(series: pd.Series) -> pd.Series:
    """Parse date_review column to UTC-aware timestamps."""
    parsed = pd.to_datetime(series, errors="coerce", utc=False)
    try:
        return parsed.dt.tz_localize("UTC")
    except Exception:
        return parsed


def load_glassdoor(
    path: str | Path,
    anonymize_firm: bool = True,
    keep_location: bool = False,
    min_text_length: int = 30,
    chunk_size: int = 50_000,
    source_file: str = "",
) -> pd.DataFrame:
    """
    Load, redact, and normalise the Glassdoor Kaggle reviews CSV.

    Parameters
    ----------
    path              Path to glassdoor_reviews.csv
    anonymize_firm    If True: replace firm with SHA-256 hash (firm_id); drop raw firm
    keep_location     If False: set location to empty string in output
    min_text_length   Drop rows where combined text is shorter than this
    chunk_size        Rows per chunk for the chunked CSV reader
    source_file       Label for the source_file column

    Returns
    -------
    DataFrame with common pipeline schema + Glassdoor rating columns.
    Returns empty DataFrame if file not found.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Glassdoor file not found: %s", path)
        return pd.DataFrame()

    logger.info("Loading Glassdoor reviews from %s (chunk_size=%d)...", path.name, chunk_size)

    chunks_out = []
    total_read = 0

    # Try UTF-8 first, fall back to latin-1 (common for Glassdoor exports).
    # Retry up to 3 times on PermissionError — Windows Defender can briefly
    # lock large files after a recent access.
    for attempt in range(1, 4):
        try:
            try:
                reader = pd.read_csv(
                    path, chunksize=chunk_size, low_memory=False,
                    on_bad_lines="skip", encoding="utf-8",
                )
                chunks_out, total_read = _process_chunks(
                    reader, anonymize_firm, keep_location, min_text_length, source_file or str(path)
                )
            except UnicodeDecodeError:
                logger.info("UTF-8 decode failed, retrying with latin-1...")
                reader = pd.read_csv(
                    path, chunksize=chunk_size, low_memory=False,
                    on_bad_lines="skip", encoding="latin-1",
                )
                chunks_out, total_read = _process_chunks(
                    reader, anonymize_firm, keep_location, min_text_length, source_file or str(path)
                )
            break  # success — exit retry loop
        except PermissionError as exc:
            if attempt == 3:
                logger.error("Glassdoor file locked after 3 attempts: %s", exc)
                return pd.DataFrame()
            wait = attempt * 5
            logger.warning("Glassdoor file locked (attempt %d/3) — retrying in %ds...", attempt, wait)
            time.sleep(wait)

    if not chunks_out:
        logger.warning("Glassdoor: no rows passed filters (file may be empty or all rows too short)")
        return pd.DataFrame()

    df = pd.concat(chunks_out, ignore_index=True)
    df = df.drop_duplicates(subset=["id"], keep="first")

    logger.info(
        "Glassdoor: read %d rows -> kept %d after redaction, quality filter, dedup",
        total_read, len(df),
    )
    return df


def _process_chunks(
    reader,
    anonymize_firm: bool,
    keep_location: bool,
    min_text_length: int,
    source_file: str,
) -> tuple[list[pd.DataFrame], int]:
    total_read = 0
    chunks_out = []

    for chunk in reader:
        total_read += len(chunk)

        # Normalise column names to lowercase
        chunk.columns = [c.strip().lower() for c in chunk.columns]

        # Require the essential text columns; skip chunk if missing
        if not {"headline", "pros", "cons"}.issubset(chunk.columns):
            logger.warning("Chunk missing headline/pros/cons columns — skipped")
            continue

        # Fill NA in text fields before processing
        for col in ["headline", "pros", "cons"]:
            chunk[col] = chunk[col].fillna("").astype(str)

        # ── PII redaction ─────────────────────────────────────────────────
        chunk["headline"] = chunk["headline"].apply(_redact)
        chunk["pros"]     = chunk["pros"].apply(_redact)
        chunk["cons"]     = chunk["cons"].apply(_redact)

        # ── Combined text ─────────────────────────────────────────────────
        chunk["text"] = [
            _combine_text(h, p, c)
            for h, p, c in zip(chunk["headline"], chunk["pros"], chunk["cons"])
        ]

        # Apply redaction to combined text too (catches cross-field patterns)
        chunk["text"] = chunk["text"].apply(_redact)

        # ── Text quality gate ─────────────────────────────────────────────
        chunk = chunk[chunk["text"].str.len() >= min_text_length].copy()
        if chunk.empty:
            continue

        # ── Stable row ID ─────────────────────────────────────────────────
        firm_col     = chunk.get("firm",       pd.Series([""] * len(chunk), index=chunk.index))
        date_col     = chunk.get("date_review",pd.Series([""] * len(chunk), index=chunk.index))
        jt_col       = chunk.get("job_title",  pd.Series([""] * len(chunk), index=chunk.index))

        chunk["id"] = [
            _make_row_id(f, d, j, h, p, c)
            for f, d, j, h, p, c in zip(
                firm_col.fillna(""), date_col.fillna(""), jt_col.fillna(""),
                chunk["headline"], chunk["pros"], chunk["cons"],
            )
        ]

        # ── Firm anonymisation ────────────────────────────────────────────
        if "firm" in chunk.columns:
            chunk["firm_id"] = chunk["firm"].fillna("").apply(_hash_firm)
            if anonymize_firm:
                chunk = chunk.drop(columns=["firm"])
        else:
            chunk["firm_id"] = ""

        # ── Location ──────────────────────────────────────────────────────
        if not keep_location:
            chunk["location"] = ""
        elif "location" not in chunk.columns:
            chunk["location"] = ""

        # ── created_utc ───────────────────────────────────────────────────
        if "date_review" in chunk.columns:
            chunk["created_utc"] = _parse_date(chunk["date_review"])
        else:
            chunk["created_utc"] = pd.NaT

        # ── Rating columns (numeric) ──────────────────────────────────────
        for col in RATING_COLS:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            else:
                chunk[col] = pd.NA

        # ── Common schema columns ─────────────────────────────────────────
        chunk["type"]        = "review"
        chunk["source"]      = "glassdoor"
        chunk["subreddit"]   = ""
        chunk["title"]       = chunk["headline"]
        chunk["author"]      = ""
        chunk["score"]       = pd.array([pd.NA] * len(chunk), dtype="Int64")
        chunk["url"]         = ""
        chunk["source_file"] = source_file

        # Rename job_title / current so they survive column selection
        if "job_title" in chunk.columns:
            chunk["job_title"] = chunk["job_title"].fillna("").astype(str)
        else:
            chunk["job_title"] = ""

        if "current" in chunk.columns:
            chunk["current"] = chunk["current"].fillna("").astype(str)
        else:
            chunk["current"] = ""

        chunks_out.append(chunk)

    return chunks_out, total_read