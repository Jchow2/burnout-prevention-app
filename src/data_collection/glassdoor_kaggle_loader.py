"""
src/data_collection/glassdoor_kaggle_loader.py

Loads the davidgauthier/glassdoor-job-reviews Kaggle dataset
(glassdoor_reviews.csv — 292MB, ~850K rows).

This is a SEPARATE loader from glassdoor_loader.py because the Kaggle
dataset has different column names from your competition dataset.
Both feed into the same merger.py normalizer via the SOURCE_RUNNERS
registry in run_pipeline.py.

Confirmed file from dir output:
  data/raw/glassdoor_kaggle/glassdoor_reviews.csv  (292MB)

Known columns in davidgauthier dataset (from public notebooks):
  firm, date_review, job_title, current, location,
  overall_rating, work_life_balance, culture_values,
  diversity_inclusion, career_opp, comp_benefits,
  senior_mgmt, recommend, ceo_approv, outlook,
  headline, pros, cons

Target companies for NeuroTrace (warehouse / blue-collar focus):
  Amazon, Walmart, Target, UPS, FedEx, USPS,
  XPO Logistics, DHL, Home Depot, Lowe's
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import DATA_RAW  # type: ignore

import logging
import hashlib
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Companies to keep — warehouse, logistics, retail, trades
# ---------------------------------------------------------------------------

TARGET_COMPANIES = [
    "amazon", "walmart", "target", "ups", "fedex", "usps",
    "xpo", "xpo logistics", "dhl", "home depot", "lowe",
    "costco", "kroger", "dollar general", "dollar tree",
    "sysco", "chewy", "wayfair", "chewy.com",
    "fulfillment", "warehouse", "distribution",
]

# Minimum overall_rating word count for review text
GLASSDOOR_KAGGLE_MIN_LENGTH = 20


# ---------------------------------------------------------------------------
# Column mapping for davidgauthier dataset
# ---------------------------------------------------------------------------

KAGGLE_COLUMN_MAP = {
    "firm":                "company_name",
    "date_review":         "rating_date",
    "job_title":           "employee_job_title",
    "current":             "employee_status",
    "location":            "location",
    "overall_rating":      "rating_overall",
    "work_life_balance":   "rating_work_life",
    "culture_values":      "rating_culture",
    "diversity_inclusion": "rating_diversity",
    "career_opp":          "rating_career",
    "comp_benefits":       "rating_comp",
    "senior_mgmt":         "rating_mgmt",
    "recommend":           "recommend_to_friend",
    "ceo_approv":          "ceo_approval",
    "outlook":             "business_outlook",
    "headline":            "summary_text",
    "pros":                "pros_text",
    "cons":                "cons_text",
}


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_glassdoor_kaggle(
    filepath: Path = DATA_RAW / "glassdoor_kaggle" / "glassdoor_reviews.csv",
    company_filter: Optional[list[str]] = TARGET_COMPANIES,
    min_length: int = GLASSDOOR_KAGGLE_MIN_LENGTH,
    chunksize: int = 50_000,
) -> pd.DataFrame:
    """
    Load the Kaggle Glassdoor reviews dataset in chunks (292MB file).

    Uses chunked reading to avoid loading 850K rows into memory at once.
    Filters to target companies during loading so memory stays manageable.

    Args:
        filepath:       Path to glassdoor_reviews.csv.
        company_filter: List of company name substrings to keep (lowercase).
                        Pass None to load all companies.
        min_length:     Minimum combined text length to keep a row.
        chunksize:      Rows per chunk — tune down if you hit memory issues.

    Returns:
        Normalized DataFrame matching the merger.py base schema.
    """
    if not filepath.exists():
        logger.warning(f"Glassdoor Kaggle file not found: {filepath}")
        return pd.DataFrame()

    logger.info(f"Loading {filepath.name} in chunks of {chunksize:,}...")

    chunks = []
    total_read = 0
    total_kept = 0

    try:
        reader = pd.read_csv(
            filepath,
            chunksize=chunksize,
            low_memory=False,
            on_bad_lines="skip",
            encoding="utf-8",
        )
    except UnicodeDecodeError:
        reader = pd.read_csv(
            filepath,
            chunksize=chunksize,
            low_memory=False,
            on_bad_lines="skip",
            encoding="latin-1",
        )

    for chunk in reader:
        total_read += len(chunk)

        # Normalize column names
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        chunk = chunk.rename(columns=KAGGLE_COLUMN_MAP)

        # Company filter
        if company_filter and "company_name" in chunk.columns:
            pattern = "|".join(company_filter)
            mask = chunk["company_name"].str.lower().str.contains(
                pattern, na=False, regex=True
            )
            chunk = chunk[mask].copy()

        if chunk.empty:
            continue

        # Build combined text from pros + cons + headline
        chunk["text"] = chunk.apply(_combine_text, axis=1)

        # Drop rows with insufficient text
        chunk = chunk[chunk["text"].str.len() >= min_length].copy()

        if chunk.empty:
            continue

        # Generate review IDs
        chunk["review_id"] = chunk["text"].apply(
            lambda t: f"gdk_{hashlib.md5(str(t)[:120].encode()).hexdigest()[:12]}"
        )
        chunk["source"] = "glassdoor_kaggle"
        chunk["subreddit"] = None

        # Parse ratings as numeric
        for col in chunk.columns:
            if col.startswith("rating_"):
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        # Parse dates
        if "rating_date" in chunk.columns:
            chunk["rating_date"] = pd.to_datetime(
                chunk["rating_date"], errors="coerce"
            )

        # Detect warehouse roles
        if "employee_job_title" in chunk.columns:
            chunk["is_warehouse"] = chunk["employee_job_title"].apply(
                _detect_warehouse
            )
        else:
            chunk["is_warehouse"] = None

        # Fill missing base columns
        for col in ["employee_job_title", "employee_status", "company_name"]:
            if col not in chunk.columns:
                chunk[col] = "Unknown"

        total_kept += len(chunk)
        chunks.append(chunk)

    if not chunks:
        logger.warning(
            "Glassdoor Kaggle: no rows matched target companies. "
            "Check company_filter or set it to None to load all."
        )
        return pd.DataFrame()

    combined = pd.concat(chunks, ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"], keep="first")

    logger.info(
        f"Glassdoor Kaggle: read {total_read:,} rows → "
        f"kept {len(combined):,} after filtering and deduplication"
    )

    # Company breakdown
    if "company_name" in combined.columns:
        top = combined["company_name"].value_counts().head(10)
        for company, count in top.items():
            logger.info(f"  {company}: {count:,}")

    return combined


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _combine_text(row: pd.Series) -> str:
    """Merge headline + pros + cons into a single review text."""
    parts = []
    headline = str(row.get("summary_text", "") or "").strip()
    pros     = str(row.get("pros_text",    "") or "").strip()
    cons     = str(row.get("cons_text",    "") or "").strip()

    if headline and headline.lower() not in ("nan", "none", ""):
        parts.append(headline)
    if pros and pros.lower() not in ("nan", "none", ""):
        parts.append(f"Pros: {pros}")
    if cons and cons.lower() not in ("nan", "none", ""):
        parts.append(f"Cons: {cons}")

    return " ".join(parts).strip()


def _detect_warehouse(job_title: str) -> bool:
    """Heuristic: is this job title likely a warehouse/fulfillment role?"""
    if pd.isna(job_title) or not job_title:
        return False
    title_lower = str(job_title).lower()
    warehouse_terms = [
        "warehouse", "fulfillment", "picker", "packer", "stower",
        "sorter", "loader", "unloader", "forklift", "associate",
        "tier 1", "tier 3", "fc", "delivery", "driver", "handler",
        "operations", "logistics", "shipping", "receiving",
    ]
    return any(term in title_lower for term in warehouse_terms)


# ---------------------------------------------------------------------------
# CLI — quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    df = load_glassdoor_kaggle()

    if not df.empty:
        print(f"\nLoaded {len(df):,} rows")
        print(f"Columns: {list(df.columns)}")
        print("\nSample:")
        print(df[["company_name", "employee_job_title",
                   "rating_overall", "text"]].head(5).to_string())
    else:
        print("No data loaded — check file path and company_filter above.")
