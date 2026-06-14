"""
src/checkin/checkin_engine.py

The 60-second daily check-in for frontline workers.

Based on occupational stress literature recommendations:
    - 6 core questions maximum (brevity beats completeness for adherence)
    - 0-10 scales (intuitive, granular, matches clinical pain scales workers know)
    - Non-clinical wording (energy, soreness, sleep — not burnout, depression)
    - One safety-critical binary question (shift safety energy)
    - Two intervention preference questions (personalization layer)

Question design informed by:
    - Strongest predictors in occupational stress literature: pain, fatigue,
      sleep disruption, stress, emotional exhaustion
    - "Your Brain on Art" (Magsamen & Ross, 2023) — neuroarts micro-intervention
      should be matched to the worker's preferred modality
    - Brevity principle: high adherence in real-world settings requires
      4-6 items max with simple response options

Cadence design:
    - Daily:   6 core questions (physical worn-down, soreness, sleep, stress,
               mental drain, safety energy)
    - Weekly:  Extended questions (enjoyment, irritability, motivation, recovery)
               — triggered when daily scores stay elevated
    - After intervention: "Did that help?" (thumbs up/down)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict
import uuid

from src.scoring.burnout_scales import score_bat12, score_olbi


# ---------------------------------------------------------------------------
# Check-in data model
# ---------------------------------------------------------------------------

@dataclass
class CheckInResponse:
    """A single completed check-in from a worker."""

    # Identity & timing
    checkin_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # --- Core 6 questions (0-10 scale, higher = worse) ---

    # Q1: "How physically worn down do you feel right now?"
    physical_worn_down: int = 0

    # Q2: "How much pain or soreness is interfering with your work or mood today?"
    pain_soreness: int = 0

    # Q3: "How mentally drained or emotionally exhausted do you feel right now?"
    mentally_drained: int = 0

    # Q4: "How well did you sleep last night?" (0 = terribly, 10 = great — INVERTED)
    sleep_quality: int = 10

    # Q5: "How stressed do you feel about work right now?"
    work_stress: int = 0

    # Q6: "Do you feel like you have enough energy to finish your shift safely?"
    shift_safety_energy: str = "yes"   # "yes" | "somewhat" | "no"

    # --- Intervention preference (personalization layer) ---

    # "Would you rather have something calming, energizing, or distracting?"
    intervention_mood: str = "calming"  # "calming" | "energizing" | "distracting"

    # "Do you want sound, visuals, or a quick reflection?"
    intervention_modality: str = "reflection"  # "sound" | "visuals" | "reflection"

    # --- Context ---
    time_available_min: int = 10     # Minutes available: 5 | 10 | 15
    setting: str = "break_room"      # break_room | commute | home | other
    shift_type: str = "day"          # day | night | swing | rotating

    # --- Optional free-text ---
    free_text: Optional[str] = None  # "Shift debrief" journal entry

    # --- Optional burnout scale responses (periodic deeper assessment) ---
    bat12_responses: Optional[Dict[str, int]] = None  # BAT-12 items bat1..bat12 (1-5)
    olbi_responses: Optional[Dict[str, int]] = None   # OLBI items olbi1..olbi16 (1-4)

    bat12_scores: Optional[dict] = None
    olbi_scores: Optional[dict] = None

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []

        # 0-10 scale questions (sleep_quality included — validated same range)
        for fld in [
            "physical_worn_down", "pain_soreness", "mentally_drained",
            "sleep_quality", "work_stress",
        ]:
            val = getattr(self, fld)
            if not isinstance(val, int) or val < 0 or val > 10:
                errors.append(f"{fld} must be 0-10, got {val}")

        # Safety energy
        if self.shift_safety_energy not in ("yes", "somewhat", "no"):
            errors.append("shift_safety_energy must be yes/somewhat/no")

        # Intervention preferences
        if self.intervention_mood not in ("calming", "energizing", "distracting"):
            errors.append("intervention_mood must be calming/energizing/distracting")
        if self.intervention_modality not in ("sound", "visuals", "reflection"):
            errors.append("intervention_modality must be sound/visuals/reflection")

        # Context
        if self.time_available_min not in (5, 10, 15):
            errors.append("time_available_min must be 5, 10, or 15")
        if self.setting not in ("break_room", "commute", "home", "other"):
            errors.append("setting must be break_room/commute/home/other")

        if self.bat12_responses is not None:
            errors.extend(self._validate_bat12())

        if self.olbi_responses is not None:
            errors.extend(self._validate_olbi())

        return errors

    def _validate_bat12(self) -> list[str]:
        errors = []
        for i in range(1, 13):
            key = f"bat{i}"
            if key not in self.bat12_responses:
                errors.append(f"Missing BAT-12 item: {key}")
                continue
            val = self.bat12_responses[key]
            if not isinstance(val, int) or not (1 <= val <= 5):
                errors.append(f"{key} must be 1-5, got {val}")
        return errors

    def _validate_olbi(self) -> list[str]:
        errors = []
        for i in range(1, 17):
            key = f"olbi{i}"
            if key not in self.olbi_responses:
                errors.append(f"Missing OLBI item: {key}")
                continue
            val = self.olbi_responses[key]
            if not isinstance(val, int) or not (1 <= val <= 4):
                errors.append(f"{key} must be 1-4, got {val}")
        return errors

    def compute_scale_scores(self) -> None:
        """Score BAT-12 and OLBI responses if provided."""
        if self.bat12_responses is not None:
            self.bat12_scores = score_bat12(self.bat12_responses).to_dict()
        if self.olbi_responses is not None:
            self.olbi_scores = score_olbi(self.olbi_responses).to_dict()

    # ------------------------------------------------------------------
    # Dimension scores — mapped to 0-10 for the compound scorer
    # ------------------------------------------------------------------

    @property
    def body_score(self) -> float:
        """
        Physical strain dimension (0-10 scale, higher = worse).
        Combines: physical worn-down, pain/soreness, inverted sleep.
        """
        # Invert sleep: 10 (great) -> 0 strain, 0 (terrible) -> 10 strain
        sleep_strain = 10 - self.sleep_quality
        return (self.physical_worn_down + self.pain_soreness + sleep_strain) / 3

    @property
    def mind_score(self) -> float:
        """
        Mental/emotional strain dimension (0-10 scale, higher = worse).
        Combines: mentally drained, work stress, safety energy flag.
        """
        # Convert safety energy to numeric: yes=0, somewhat=4, no=8
        safety_numeric = {"yes": 0, "somewhat": 4, "no": 8}.get(
            self.shift_safety_energy, 4
        )
        return (self.mentally_drained + self.work_stress + safety_numeric) / 3

    @property
    def life_score(self) -> float:
        """
        Life impact dimension (0-10 scale, higher = worse).

        In the daily 6-question check-in, we don't directly ask about
        financial or schedule stress (those are weekly questions). For the
        daily score, life_score is derived from the overall strain pattern:
        high physical + high mental with poor sleep signals life impact.
        """
        # Heuristic: when body and mind are both strained and sleep is poor,
        # life outside work is being affected
        sleep_strain = 10 - self.sleep_quality
        combined = (self.body_score + self.mind_score) / 2

        # Amplify if sleep is bad (sleep disruption is the bridge between
        # work strain and life impact)
        if sleep_strain >= 6:
            return min(10, combined * 1.2)
        return combined

    @property
    def burnout_scales(self) -> Optional[dict]:
        """Combined BAT-12 / OLBI scores, or None if no scales were completed."""
        if self.bat12_scores is None and self.olbi_scores is None:
            return None
        return {"bat12": self.bat12_scores, "olbi": self.olbi_scores}

    def to_dict(self) -> dict:
        """Serialize for storage / API response."""
        return {
            "checkin_id": self.checkin_id,
            "timestamp": self.timestamp,
            "core_responses": {
                "physical_worn_down": self.physical_worn_down,
                "pain_soreness": self.pain_soreness,
                "mentally_drained": self.mentally_drained,
                "sleep_quality": self.sleep_quality,
                "work_stress": self.work_stress,
                "shift_safety_energy": self.shift_safety_energy,
            },
            "dimensions": {
                "body": round(self.body_score, 2),
                "mind": round(self.mind_score, 2),
                "life": round(self.life_score, 2),
            },
            "burnout_scales": self.burnout_scales,
            "intervention_preferences": {
                "mood": self.intervention_mood,
                "modality": self.intervention_modality,
            },
            "context": {
                "time_available_min": self.time_available_min,
                "setting": self.setting,
                "shift_type": self.shift_type,
            },
            "free_text": self.free_text,
        }


# ---------------------------------------------------------------------------
# Check-in question definitions (for frontend rendering)
# ---------------------------------------------------------------------------

# Core daily questions — presented in this order
CHECKIN_QUESTIONS = [
    {
        "id": "physical_worn_down",
        "question": "How physically worn down do you feel right now?",
        "type": "slider",
        "min": 0,
        "max": 10,
        "min_label": "Not at all",
        "max_label": "Extremely",
        "icon": "body",
    },
    {
        "id": "pain_soreness",
        "question": "How much pain or soreness is interfering with your work or mood today?",
        "type": "slider",
        "min": 0,
        "max": 10,
        "min_label": "None",
        "max_label": "Severe",
        "icon": "alert-circle",
    },
    {
        "id": "mentally_drained",
        "question": "How mentally drained or emotionally exhausted do you feel right now?",
        "type": "slider",
        "min": 0,
        "max": 10,
        "min_label": "Fresh",
        "max_label": "Completely empty",
        "icon": "brain",
    },
    {
        "id": "sleep_quality",
        "question": "How well did you sleep last night?",
        "type": "slider",
        "min": 0,
        "max": 10,
        "min_label": "Terribly",
        "max_label": "Great",
        "icon": "moon",
        "inverted": True,  # Higher = better (opposite of other questions)
    },
    {
        "id": "work_stress",
        "question": "How stressed do you feel about work right now?",
        "type": "slider",
        "min": 0,
        "max": 10,
        "min_label": "Calm",
        "max_label": "Overwhelmed",
        "icon": "activity",
    },
    {
        "id": "shift_safety_energy",
        "question": "Do you feel like you have enough energy to finish your shift safely?",
        "type": "choice",
        "options": [
            {"value": "yes", "label": "Yes"},
            {"value": "somewhat", "label": "Somewhat"},
            {"value": "no", "label": "No"},
        ],
        "icon": "shield",
    },
]

# Intervention preference questions — presented after core questions
INTERVENTION_PREFERENCE_QUESTIONS = [
    {
        "id": "intervention_mood",
        "question": "What would help you most right now?",
        "type": "choice",
        "options": [
            {"value": "calming", "label": "Something calming"},
            {"value": "energizing", "label": "Something energizing"},
            {"value": "distracting", "label": "Something distracting"},
        ],
        "icon": "target",
    },
    {
        "id": "intervention_modality",
        "question": "What sounds good?",
        "type": "choice",
        "options": [
            {"value": "sound", "label": "Sound / music"},
            {"value": "visuals", "label": "Something visual"},
            {"value": "reflection", "label": "A quick reflection"},
        ],
        "icon": "layers",
    },
]

# Context questions — presented once during onboarding, editable anytime
CONTEXT_QUESTIONS = [
    {
        "id": "time_available_min",
        "question": "How much time do you have?",
        "type": "choice",
        "options": [
            {"value": 5, "label": "5 min"},
            {"value": 10, "label": "10 min"},
            {"value": 15, "label": "15 min"},
        ],
        "icon": "clock",
    },
    {
        "id": "setting",
        "question": "Where are you?",
        "type": "choice",
        "options": [
            {"value": "break_room", "label": "Break room"},
            {"value": "commute", "label": "Commute"},
            {"value": "home", "label": "Home"},
            {"value": "other", "label": "Other"},
        ],
        "icon": "map-pin",
    },
]

# Weekly extended questions — triggered when daily scores stay elevated
# (not part of the daily flow, used for deeper assessment)
WEEKLY_EXTENDED_QUESTIONS = [
    {
        "id": "enjoyment",
        "question": "Did you enjoy anything about work this week?",
        "type": "slider",
        "min": 0,
        "max": 10,
        "min_label": "Not at all",
        "max_label": "Quite a bit",
    },
    {
        "id": "irritability",
        "question": "How irritable or short-tempered have you been this week?",
        "type": "slider",
        "min": 0,
        "max": 10,
        "min_label": "Not at all",
        "max_label": "Very",
    },
    {
        "id": "motivation",
        "question": "How motivated do you feel about going to work?",
        "type": "slider",
        "min": 0,
        "max": 10,
        "min_label": "Dreading it",
        "max_label": "Ready to go",
    },
    {
        "id": "recovery",
        "question": "Do you feel like you recovered on your days off?",
        "type": "choice",
        "options": [
            {"value": "yes", "label": "Yes, fully"},
            {"value": "somewhat", "label": "Somewhat"},
            {"value": "no", "label": "Not really"},
        ],
    },
    {
        "id": "financial_stress",
        "question": "How worried are you about money right now?",
        "type": "slider",
        "min": 0,
        "max": 10,
        "min_label": "Not worried",
        "max_label": "Constantly worried",
    },
    {
        "id": "schedule_impact",
        "question": "How much is your work schedule affecting your life outside work?",
        "type": "slider",
        "min": 0,
        "max": 10,
        "min_label": "No impact",
        "max_label": "Major impact",
    },
]
