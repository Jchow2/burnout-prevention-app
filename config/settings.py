"""
config/settings.py — Single source of truth for the entire pipeline.

All credentials come from environment variables (via .env file).
All paths, keyword lists, subreddit targets, and pipeline parameters live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv # type: ignore

load_dotenv()

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reddit API credentials
# ---------------------------------------------------------------------------
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv(
    "REDDIT_USER_AGENT",
    "WorkforceSentimentAnalyzer/1.0 (by u/your_username)",
)

# ---------------------------------------------------------------------------
# Subreddit targets (aligned with your README)
# ---------------------------------------------------------------------------
SUBREDDIT_GROUPS = {
    "warehouse": ["Warehouseworkers", "Warehousing", "AmazonFC", "walmart", "Target"],
    "shipping": ["UPS", "FedEx", "USPS"],
    "manufacturing": ["manufacturing"],
    "general": [
        "antiwork", "WorkReform", "Workplace", "jobs",
        "TalesFromRetail", "TalesFromYourServer",
    ],
}
ALL_SUBREDDITS = [s for subs in SUBREDDIT_GROUPS.values() for s in subs]

SUB_TO_CATEGORY = {}
for cat, subs in SUBREDDIT_GROUPS.items():
    for s in subs:
        SUB_TO_CATEGORY[s.lower()] = cat

# ---------------------------------------------------------------------------
# Keyword themes (shared across all pipeline stages)
# ---------------------------------------------------------------------------
KEYWORD_THEMES = {
    "burnout": [
        "stress", "exhausted", "overwhelmed", "burned out", "burnout",
        "drained", "breaking point", "mental health",
    ],
    "physical_demands": [
        "pain", "fatigue", "physically demanding", "injury", "hurt",
        "sore", "back pain", "repetitive strain", "carpal tunnel",
    ],
    "management_issues": [
        "micromanage", "poor leadership", "unfair", "favoritism",
        "toxic manager", "no support", "abusive", "retaliation",
    ],
    "workload": [
        "pressure", "quotas", "overtime", "understaffed", "overworked",
        "too much work", "mandatory overtime", "short staffed",
    ],
    "environment": [
        "toxic", "hostile", "negative culture", "hostile environment",
        "terrible culture", "unsafe", "dangerous",
    ],
}

# Flat complaint keywords (for filtering in collectors)
COMPLAINT_KEYWORDS = [
    "hate my job", "toxic", "unsafe", "injury", "injured",
    "overtime", "mandatory overtime", "overworked",
    "understaffed", "short staffed",
    "minimum wage", "no raise", "no benefits",
    "bully", "harassed", "abusive", "write up", "pip",
    "burnout", "burned out", "quit", "quitting",
]

# Flairs that signal complaint posts on Reddit
COMPLAINT_FLAIRS = {
    "rant", "vent", "venting", "storytime", "story time",
    "bad boss", "wage theft", "quit", "i quit", "success (i quit)",
    "advice", "need advice", "question", "rant/vent", "discussion",
}

# ---------------------------------------------------------------------------
# Sentiment labeling thresholds (TextBlob polarity)
# ---------------------------------------------------------------------------
TEXTBLOB_POSITIVE_THRESHOLD = 0.1
TEXTBLOB_NEGATIVE_THRESHOLD = -0.1
# polarity >  0.1 → positive
# polarity < -0.1 → negative
# otherwise        → neutral

# ---------------------------------------------------------------------------
# Pipeline defaults
# ---------------------------------------------------------------------------
REDDIT_MAX_POSTS_PER_SUB = 500
GLASSDOOR_MIN_REVIEW_LENGTH = 20
MIN_REVIEW_LENGTH = 20            # Global minimum for any source
VADER_SENTIMENT_THRESHOLD = -0.05  # For Reddit pre-filtering

# ---------------------------------------------------------------------------
# Check-in pipeline settings
# (separate from the external data pipeline above)
# ---------------------------------------------------------------------------

# Burnout domain keywords for shift debrief free-text analysis.
# Keys match BAT-12 subscale names and the OLBI disengagement subscale so
# that labeler.py output can feed directly into _boosted_categories_from_scales.
CHECKIN_BURNOUT_THEMES = {
    "exhaustion": [
        "exhausted", "drained", "wiped out", "no energy", "can't go on",
        "running on empty", "dead on my feet", "worn out", "burnt out",
        "nothing left", "empty", "depleted",
    ],
    "mental_distance": [
        "don't care", "checked out", "going through the motions",
        "numb", "disconnected", "not present", "zoned out", "autopilot",
        "couldn't care less", "just a job",
    ],
    "cognitive_impairment": [
        "can't think", "brain fog", "can't concentrate", "forgetful",
        "making mistakes", "slow", "confused", "can't focus",
        "keep forgetting", "head isn't in it",
    ],
    "emotional_impairment": [
        "frustrated", "irritable", "snapping", "short tempered",
        "angry", "upset", "overwhelmed", "on edge", "lost my patience",
    ],
    "disengagement": [
        "don't want to be here", "hate this job", "quit", "leaving",
        "not worth it", "given up", "counting down", "just a number",
        "don't care anymore", "looking for something else",
    ],
}

# Thresholds for daily/weekly trend detection
TREND_THRESHOLDS = {
    "worsening_delta": 8,               # Score must rise this much above window avg
    "improving_delta": 8,               # Score must drop this much below window avg
    "consecutive_high_threshold": 60,   # Score above this counts as a "high day"
    "alert_consecutive_days": 3,        # Flag alert after this many consecutive high days
    "history_window_days": 7,           # Rolling window for trend baseline
}

# When to surface periodic BAT-12 / OLBI scales to the worker
SCALE_CADENCE = {
    "bat12_trigger_consecutive_high": 3,  # Prompt BAT-12 after 3 consecutive high days
    "bat12_trigger_score_threshold": 65,  # Or when compound score exceeds this
    "olbi_trigger_consecutive_high": 5,   # OLBI is longer — trigger less frequently
    "olbi_min_days_between": 14,          # Don't re-prompt OLBI within 2 weeks
}

# Months that typically show elevated scores by worker sector.
# Adds contextual framing to trend labels ("score elevated — expected for Q4").
PEAK_SEASON_MONTHS = {
    "warehouse": [10, 11, 12, 1],   # Q4 peak + January returns surge
    "retail":    [11, 12, 1],       # Holiday rush
    "general":   [],                # No assumed peak for unclassified workers
}

# ---------------------------------------------------------------------------
# Output schema — the 18 columns your model expects
# ---------------------------------------------------------------------------
UNIFIED_SCHEMA = [
    "review_id",
    "text",
    "sentiment_label",
    "source",
    "subreddit",               # Reddit only; NaN for others
    "employee_job_title",
    "employee_status",
    "company_name",
    "rating_date",
    "rating_overall",
    "polarity",
    "subjectivity",
    "confidence",
    "burnout_keywords",
    "physical_keywords",
    "management_keywords",
    "workload_keywords",
    "is_warehouse",
]
