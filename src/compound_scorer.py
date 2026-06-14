"""
src/scoring/compound_scorer.py

Compound Burnout Scorer — the core algorithm.

Takes a CheckInResponse and produces a 0-100 burnout score that models
the INTERACTION between physical, emotional, and financial stress.

Key insight: these dimensions compound each other. A worker with moderate
pain AND moderate financial stress is at higher risk than someone with
severe pain alone, because the anxiety about affording treatment amplifies
the pain's impact on mood and sleep. This interaction effect is what no
existing burnout tool models.

Scoring approach:
    1. Normalize each dimension to 0-100
    2. Compute weighted base score
    3. Apply compound interaction multiplier
    4. Factor in consecutive-day trends (if history available)
    5. Output: score (0-100), risk level, dominant dimension, trend
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class BurnoutScore:
    """Result of compound burnout scoring."""

    # Core score
    score: float                    # 0-100 composite
    risk_level: str                 # "low" | "moderate" | "high" | "critical"

    # Dimension breakdown (each 0-100)
    body_score: float
    mind_score: float
    life_score: float

    # Compound analysis
    dominant_dimension: str         # Which dimension drives the score most
    compound_multiplier: float      # How much interaction amplifies the score
    interaction_pairs: List[str]     # Which dimensions are compounding

    # Trend (if history provided)
    trend: Optional[str] = None     # "improving" | "stable" | "worsening"
    consecutive_high_days: int = 0  # Days in a row with score > 60

    # Burnout scale scores (BAT-12 / OLBI) — present when periodic assessment was completed
    burnout_scales: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 1),
            "risk_level": self.risk_level,
            "dimensions": {
                "body": round(self.body_score, 1),
                "mind": round(self.mind_score, 1),
                "life": round(self.life_score, 1),
            },
            "dominant_dimension": self.dominant_dimension,
            "compound_multiplier": round(self.compound_multiplier, 2),
            "interaction_pairs": self.interaction_pairs,
            "trend": self.trend,
            "consecutive_high_days": self.consecutive_high_days,
            "burnout_scales": self.burnout_scales,
        }


# ---------------------------------------------------------------------------
# Scoring parameters
# ---------------------------------------------------------------------------

# Dimension weights (sum to 1.0)
# Body is weighted slightly higher because physical injury risk is immediate
DIMENSION_WEIGHTS = {
    "body": 0.40,
    "mind": 0.35,
    "life": 0.25,
}

# Compound interaction: when two dimensions are BOTH elevated (> threshold),
# the overall score gets amplified. This models the feedback loop.
COMPOUND_THRESHOLD = 55  # Dimension score above this triggers interaction
COMPOUND_PAIRS = {
    ("body", "mind"): 1.15,    # Pain + anxiety feedback loop
    ("body", "life"): 1.12,    # Pain + financial worry (can't afford to rest)
    ("mind", "life"): 1.10,    # Emotional drain + financial stress
}
# If ALL THREE are elevated, apply an additional multiplier
TRIPLE_COMPOUND = 1.25

# Risk level thresholds
RISK_THRESHOLDS = {
    "low": (0, 30),
    "moderate": (30, 55),
    "high": (55, 75),
    "critical": (75, 100),
}


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def _normalize_dimension(raw_score: float) -> float:
    """Convert 0-10 dimension average to 0-100 scale."""
    # 0 → 0, 5.0 → 50, 10.0 → 100
    return max(0, min(100, raw_score * 10))


def _classify_risk(score: float) -> str:
    """Map 0-100 score to risk level."""
    for level, (low, high) in RISK_THRESHOLDS.items():
        if low <= score < high:
            return level
    return "critical"


def compute_burnout_score(
    checkin,
    history: Optional[list] = None,
) -> BurnoutScore:
    """
    Compute compound burnout score from a check-in response.

    Args:
        checkin: A CheckInResponse object (or any object with
                 body_score, mind_score, life_score properties).
        history: Optional list of previous BurnoutScore objects
                 (most recent first) for trend detection.

    Returns:
        BurnoutScore with composite score, risk level, and analysis.
    """
    # Step 1: Normalize each dimension to 0-100
    body = _normalize_dimension(checkin.body_score)
    mind = _normalize_dimension(checkin.mind_score)
    life = _normalize_dimension(checkin.life_score)

    dimensions = {"body": body, "mind": mind, "life": life}

    # Step 2: Weighted base score
    base_score = (
        body * DIMENSION_WEIGHTS["body"]
        + mind * DIMENSION_WEIGHTS["mind"]
        + life * DIMENSION_WEIGHTS["life"]
    )

    # Step 3: Compound interaction multiplier
    multiplier = 1.0
    active_pairs = []

    elevated = {dim for dim, val in dimensions.items() if val > COMPOUND_THRESHOLD}

    for (dim_a, dim_b), pair_mult in COMPOUND_PAIRS.items():
        if dim_a in elevated and dim_b in elevated:
            multiplier = max(multiplier, pair_mult)
            active_pairs.append(f"{dim_a}+{dim_b}")

    # Triple compound: all three dimensions elevated
    if len(elevated) == 3:
        multiplier = max(multiplier, TRIPLE_COMPOUND)
        active_pairs = ["body+mind+life"]

    compound_score = min(100, base_score * multiplier)

    # Step 4: Identify dominant dimension
    dominant = max(dimensions, key=dimensions.get)

    # Step 5: Trend detection from history
    trend = None
    consecutive_high = 0

    if history and len(history) >= 2:
        recent_scores = [h.score for h in history[:7]]  # Last week
        avg_recent = sum(recent_scores) / len(recent_scores)

        if compound_score > avg_recent + 8:
            trend = "worsening"
        elif compound_score < avg_recent - 8:
            trend = "improving"
        else:
            trend = "stable"

        # Count consecutive high days
        for h in history:
            if h.score > 60:
                consecutive_high += 1
            else:
                break

        if compound_score > 60:
            consecutive_high += 1  # Include today

    return BurnoutScore(
        score=compound_score,
        risk_level=_classify_risk(compound_score),
        body_score=body,
        mind_score=mind,
        life_score=life,
        dominant_dimension=dominant,
        compound_multiplier=multiplier,
        interaction_pairs=active_pairs,
        trend=trend,
        consecutive_high_days=consecutive_high,
        burnout_scales=getattr(checkin, "burnout_scales", None),
    )
