"""
src/preprocessing/labeler.py

Assigns sentiment labels and extracts keyword theme features.

Labeling pipeline:
    1. TextBlob polarity → baseline positive/neutral/negative
    2. Keyword theme counts (burnout, physical, management, workload, environment)
    3. Metadata enrichment (is_warehouse, word count, subjectivity, confidence)

These labels and features feed directly into the ML model training stage.
"""
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import (
    KEYWORD_THEMES,
    TEXTBLOB_POSITIVE_THRESHOLD,
    TEXTBLOB_NEGATIVE_THRESHOLD,
    CHECKIN_BURNOUT_THEMES,
    TREND_THRESHOLDS,
    PEAK_SEASON_MONTHS,
)

import logging

import pandas as pd 
from textblob import TextBlob 

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TextBlob sentiment
# ---------------------------------------------------------------------------

def _textblob_label(polarity: float) -> str:
    """Map TextBlob polarity to three-class label."""
    if polarity > TEXTBLOB_POSITIVE_THRESHOLD:
        return "positive"
    elif polarity < TEXTBLOB_NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def compute_textblob_features(text: str) -> dict:
    """
    Compute TextBlob polarity, subjectivity, and derived label.

    Returns:
        dict with polarity, subjectivity, sentiment_label, confidence
    """
    if not text:
        return {
            "polarity": 0.0,
            "subjectivity": 0.0,
            "sentiment_label": "neutral",
            "confidence": 0.0,
        }

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    label = _textblob_label(polarity)

    # Confidence: distance from the neutral boundary
    if label == "positive":
        confidence = min((polarity - TEXTBLOB_POSITIVE_THRESHOLD) / 0.9, 1.0)
    elif label == "negative":
        confidence = min((TEXTBLOB_NEGATIVE_THRESHOLD - polarity) / 0.9, 1.0)
    else:
        confidence = 1.0 - abs(polarity) / max(
            abs(TEXTBLOB_POSITIVE_THRESHOLD), 0.01
        )

    return {
        "polarity": round(polarity, 4),
        "subjectivity": round(subjectivity, 4),
        "sentiment_label": label,
        "confidence": round(max(confidence, 0.0), 4),
    }


# ---------------------------------------------------------------------------
# Keyword theme detection
# ---------------------------------------------------------------------------

def count_keyword_themes(text: str) -> dict:
    """
    Count occurrences of each keyword theme in the text.

    Returns:
        dict mapping "{theme}_keywords" → int count
    """
    text_lower = (text or "").lower()
    counts = {}
    for theme, keywords in KEYWORD_THEMES.items():
        counts[f"{theme}_keywords"] = sum(
            1 for kw in keywords if kw in text_lower
        )
    return counts


# ---------------------------------------------------------------------------
# Warehouse detection
# ---------------------------------------------------------------------------

WAREHOUSE_TERMS = [
    "warehouse", "fulfillment", "fc", "picker", "packer", "stower",
    "sorter", "loader", "unloader", "forklift", "dock", "conveyor",
    "delivery", "driver", "handler", "shipping", "receiving",
    "distribution center", "sort center", "logistics",
]

CORPORATE_TERMS = [
    "corporate", "office", "manager", "director", "engineer",
    "software", "hr ", "human resources", "analyst", "marketing",
    "finance", "legal", "executive", "vp ", "vice president",
]


def detect_warehouse(row: pd.Series) -> bool:
    """
    Heuristic warehouse vs. corporate detection.
    Checks job title first, then falls back to text content.
    """
    # Check job title if available
    title = str(row.get("employee_job_title", "") or "").lower()
    if title:
        if any(t in title for t in WAREHOUSE_TERMS):
            return True
        if any(t in title for t in CORPORATE_TERMS):
            return False

    # Fall back to text analysis
    text = str(row.get("text", "") or "").lower()
    warehouse_hits = sum(1 for t in WAREHOUSE_TERMS if t in text)
    corporate_hits = sum(1 for t in CORPORATE_TERMS if t in text)
    return warehouse_hits > corporate_hits


# ---------------------------------------------------------------------------
# Full labeling pipeline
# ---------------------------------------------------------------------------

def label_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
) -> pd.DataFrame:
    """
    Apply the full labeling pipeline to a DataFrame.

    Adds columns:
        - sentiment_label (positive / neutral / negative)
        - polarity, subjectivity, confidence
        - burnout_keywords, physical_keywords, management_keywords,
          workload_keywords, environment_keywords
        - is_warehouse (boolean)
        - word_count

    Args:
        df: DataFrame with a text column (should already be cleaned).

    Returns:
        Enriched DataFrame with all labeling columns added.
    """
    df = df.copy()
    initial = len(df)

    logger.info(f"Labeling {initial} reviews...")

    # TextBlob sentiment
    tb_results = df[text_column].apply(compute_textblob_features)
    tb_df = pd.DataFrame(tb_results.tolist())
    for col in tb_df.columns:
        df[col] = tb_df[col].values

    # Keyword themes
    kw_results = df[text_column].apply(count_keyword_themes)
    kw_df = pd.DataFrame(kw_results.tolist())
    for col in kw_df.columns:
        df[col] = kw_df[col].values

    # Warehouse detection
    df["is_warehouse"] = df.apply(detect_warehouse, axis=1)

    # Word count
    df["word_count"] = df[text_column].str.split().str.len()

    # Label distribution summary
    dist = df["sentiment_label"].value_counts()
    logger.info(f"Label distribution:\n{dist.to_string()}")

    return df


# ---------------------------------------------------------------------------
# Check-in debrief labeling
# ---------------------------------------------------------------------------

def label_checkin_debrief(free_text: str) -> dict:
    """
    Analyze a single shift debrief entry using TextBlob sentiment and
    CHECKIN_BURNOUT_THEMES keyword counts.

    Keys in domain_signals match BAT-12 subscale names and the OLBI
    disengagement subscale so the output plugs directly into
    _boosted_categories_from_scales in intervention_recommender.py.

    Args:
        free_text: The worker's raw debrief string (may be empty or None).

    Returns:
        dict with has_debrief, sentiment features, per-domain keyword signals,
        and word count.
    """
    if not free_text or not free_text.strip():
        return {
            "has_debrief": False,
            "sentiment": None,
            "domain_signals": {},
            "word_count": 0,
        }

    sentiment = compute_textblob_features(free_text)
    text_lower = free_text.lower()

    domain_signals = {}
    for domain, keywords in CHECKIN_BURNOUT_THEMES.items():
        hits = [kw for kw in keywords if kw in text_lower]
        domain_signals[domain] = {
            "count": len(hits),
            "keywords_found": hits,
        }

    return {
        "has_debrief": True,
        "sentiment": sentiment,
        "domain_signals": domain_signals,
        "word_count": len(free_text.split()),
    }


# ---------------------------------------------------------------------------
# Trend feature calculation (daily / weekly / seasonal)
# ---------------------------------------------------------------------------

def compute_trend_features(score_history: list, worker_type: str = "general") -> dict:
    """
    Compute trend labels and alert flags from a list of BurnoutScore objects
    or plain dicts, most recent first.

    Covers three time horizons:
        Daily  — trend direction vs. 7-day rolling average
        Weekly — consecutive high-day streak + alert flag
        Seasonal — whether the current month falls in a known peak period
                   for the worker's sector

    Args:
        score_history: List of BurnoutScore objects (with .score) or dicts
                       (with 'score' key), most recent first.
        worker_type:   Sector key for PEAK_SEASON_MONTHS lookup
                       ('warehouse', 'retail', or 'general').

    Returns:
        dict with trend, consecutive_high_days, alert_flag,
        peak_season_context, window_avg, and window_days.
    """
    from datetime import datetime

    if not score_history:
        return {
            "trend": None,
            "consecutive_high_days": 0,
            "alert_flag": False,
            "peak_season_context": False,
            "window_avg": None,
            "window_days": 0,
        }

    def _score(s):
        return s.score if hasattr(s, "score") else s.get("score", 0)

    window = TREND_THRESHOLDS["history_window_days"]
    current = _score(score_history[0])
    prior = score_history[1:window]

    window_scores = [_score(s) for s in prior]
    window_avg = sum(window_scores) / len(window_scores) if window_scores else None

    # Daily trend direction
    trend = "stable"
    if window_avg is not None:
        if current > window_avg + TREND_THRESHOLDS["worsening_delta"]:
            trend = "worsening"
        elif current < window_avg - TREND_THRESHOLDS["improving_delta"]:
            trend = "improving"

    # Weekly streak — consecutive days above the high threshold
    high_threshold = TREND_THRESHOLDS["consecutive_high_threshold"]
    consecutive = 0
    for s in score_history:
        if _score(s) > high_threshold:
            consecutive += 1
        else:
            break

    alert_flag = consecutive >= TREND_THRESHOLDS["alert_consecutive_days"]

    # Seasonal context
    current_month = datetime.now().month
    peak_months = PEAK_SEASON_MONTHS.get(worker_type, [])
    peak_season_context = current_month in peak_months

    return {
        "trend": trend,
        "consecutive_high_days": consecutive,
        "alert_flag": alert_flag,
        "peak_season_context": peak_season_context,
        "window_avg": round(window_avg, 1) if window_avg is not None else None,
        "window_days": len(window_scores),
    }
