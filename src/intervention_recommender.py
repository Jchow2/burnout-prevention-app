"""
src/interventions/intervention_recommender.py

Intervention Recommender — presents all 5 evidence-based activity
categories and lets the worker choose what resonates with them.

Design philosophy:
    - Never prescribe. Always offer choices.
    - Workers know what they need better than an algorithm does.
    - Autonomy is the antidote to the lack of control that drives burnout.
    - Each intervention is backed by published research.
    - Framing avoids clinical/therapy language — uses direct, practical terms.

The 5 categories:
    1. Body sketch    — interoception / body awareness drawing
    2. Shift debrief  — expressive writing (Pennebaker method)
    3. Sound reset    — tempo-matched audio for stress reduction
    4. 3-thing notice — cognitive shift / psychological detachment
    5. Quick challenge — gamified micro-creative task

Each category has variants for different time windows (5/10/15 min)
and settings (break room, commute, home).
"""

from dataclasses import dataclass, field  # noqa: F401
from typing import Optional, Set


@dataclass
class Intervention:
    """A single intervention option presented to the worker."""

    id: str
    category: str              # body_sketch | shift_debrief | sound_reset | three_thing | quick_challenge
    name: str                  # Display name
    tagline: str               # One-line hook (what the worker sees first)
    description: str           # How to do it (2-3 sentences)
    why_it_works: str          # Research-backed explanation (plain language)
    duration_min: int          # Minutes required
    needs_audio: bool          # Does it require sound?
    needs_writing: bool        # Does it require typing/writing?
    best_for: list[str]        # Which burnout dimensions it targets: body/mind/life
    settings: list[str]        # Where it works: break_room/commute/home
    research_citation: str     # Source for the evidence


# ---------------------------------------------------------------------------
# Intervention catalog
# ---------------------------------------------------------------------------

INTERVENTIONS = [
    # ===== BODY SKETCH =====
    Intervention(
        id="body_sketch_5",
        category="body_sketch",
        name="Body sketch",
        tagline="Draw where you feel it",
        description=(
            "On your phone or a scrap of paper, quickly outline a stick figure. "
            "Mark where you feel tension, pain, or tightness. Circle the worst spot. "
            "That's it — just notice and mark."
        ),
        why_it_works=(
            "This is called interoception — tuning into your body's signals. "
            "Research shows that naming where you feel pain actually reduces "
            "how intense it feels. Takes 3 minutes, no skill needed."
        ),
        duration_min=5,
        needs_audio=False,
        needs_writing=False,
        best_for=["body"],
        settings=["break_room", "commute", "home"],
        research_citation="Mehling et al., 2012 — Multidimensional Assessment of Interoceptive Awareness",
    ),
    Intervention(
        id="body_sketch_10",
        category="body_sketch",
        name="Body sketch + reset",
        tagline="Map it, then move it",
        description=(
            "Draw where you feel tension (3 min). Then for each marked spot, "
            "do a slow 10-second stretch targeting that area. Breathe out as "
            "you stretch. Repeat for your top 3 spots."
        ),
        why_it_works=(
            "Combining body awareness with targeted movement breaks the "
            "pain-tension cycle. Even brief stretching reduces cortisol and "
            "loosens muscles that tighten under stress."
        ),
        duration_min=10,
        needs_audio=False,
        needs_writing=False,
        best_for=["body"],
        settings=["break_room", "home"],
        research_citation="Magsamen & Ross, 2023 — Your Brain on Art; Mehling et al., 2012",
    ),

    # ===== SHIFT DEBRIEF =====
    Intervention(
        id="shift_debrief_5",
        category="shift_debrief",
        name="Shift debrief",
        tagline="Get it out of your head",
        description=(
            "Write freely for 3-5 minutes about your shift. Don't edit, "
            "don't worry about spelling. Just dump whatever's on your mind. "
            "Nobody sees this but you."
        ),
        why_it_works=(
            "Expressive writing has been studied for 30+ years. Writing about "
            "stressful experiences — even for just a few minutes — measurably "
            "reduces stress hormones and improves mood the next day."
        ),
        duration_min=5,
        needs_audio=False,
        needs_writing=True,
        best_for=["mind"],
        settings=["break_room", "commute", "home"],
        research_citation="Pennebaker & Beall, 1986 — Expressive Writing research program",
    ),
    Intervention(
        id="shift_debrief_10",
        category="shift_debrief",
        name="Shift debrief + reframe",
        tagline="Write it, then flip it",
        description=(
            "Write about your shift for 5 minutes (whatever comes to mind). "
            "Then re-read what you wrote and underline one thing you handled "
            "well — even something small. Write one sentence about that."
        ),
        why_it_works=(
            "Adding a reframe step after expressive writing helps your brain "
            "process the stress AND recognize your own competence. This combo "
            "reduces rumination — the replay loop that keeps stress alive."
        ),
        duration_min=10,
        needs_audio=False,
        needs_writing=True,
        best_for=["mind", "life"],
        settings=["break_room", "commute", "home"],
        research_citation="Pennebaker, 1997; Seligman et al., 2005 — positive psychology interventions",
    ),

    # ===== SOUND RESET =====
    Intervention(
        id="sound_reset_5",
        category="sound_reset",
        name="Sound reset",
        tagline="Reset your nervous system",
        description=(
            "Put in one earbud. Listen to a track at 60 BPM (we'll suggest one) "
            "for 4 minutes. Close your eyes if you can. Let the rhythm slow "
            "you down."
        ),
        why_it_works=(
            "Music at 60 beats per minute synchronizes with your resting heart rate "
            "and activates your parasympathetic nervous system — the 'rest and digest' "
            "mode. Research shows this lowers cortisol within minutes."
        ),
        duration_min=5,
        needs_audio=True,
        needs_writing=False,
        best_for=["mind", "body"],
        settings=["commute", "home"],
        research_citation="Thoma et al., 2013 — Music on stress; Magsamen & Ross, 2023",
    ),
    Intervention(
        id="sound_reset_10",
        category="sound_reset",
        name="Sound reset + breathe",
        tagline="Sound meets breath",
        description=(
            "Listen to a 60 BPM track for 5 minutes. Match your breathing to the "
            "beat: inhale for 4 beats, exhale for 6 beats. Don't force it — just "
            "let the music guide your pace."
        ),
        why_it_works=(
            "Combining slow music with extended exhale breathing doubles the "
            "calming effect. The extended exhale activates your vagus nerve, "
            "which directly signals your body to lower heart rate and blood pressure."
        ),
        duration_min=10,
        needs_audio=True,
        needs_writing=False,
        best_for=["mind", "body"],
        settings=["commute", "home"],
        research_citation="Thoma et al., 2013; Zaccaro et al., 2018 — respiratory vagal stimulation",
    ),

    # ===== 3-THING NOTICE =====
    Intervention(
        id="three_thing_5",
        category="three_thing",
        name="3-thing notice",
        tagline="Shift your brain out of work mode",
        description=(
            "Name 3 things you noticed today that have nothing to do with work. "
            "A color, a sound, something someone was wearing, the weather — "
            "anything. Say them out loud or type them."
        ),
        why_it_works=(
            "This is a psychological detachment exercise. Your brain gets stuck "
            "in 'work mode' loops. Forcing it to recall non-work details breaks "
            "the loop and helps you mentally clock out."
        ),
        duration_min=5,
        needs_audio=False,
        needs_writing=False,
        best_for=["mind", "life"],
        settings=["break_room", "commute", "home"],
        research_citation="Sonnentag & Fritz, 2015 — Recovery Experience Questionnaire",
    ),
    Intervention(
        id="three_thing_10",
        category="three_thing",
        name="3-thing notice + expand",
        tagline="Notice, then describe one in detail",
        description=(
            "Name 3 non-work things you noticed today. Then pick the most "
            "interesting one and describe it in detail — what did it look like, "
            "sound like, feel like? Spend 5 minutes on just that one thing."
        ),
        why_it_works=(
            "Extended attention to a non-work detail activates your brain's "
            "default mode network — the same network that lights up during "
            "creative thinking and rest. This is the opposite of the task-focused "
            "mode you've been in all shift."
        ),
        duration_min=10,
        needs_audio=False,
        needs_writing=True,
        best_for=["mind", "life"],
        settings=["commute", "home"],
        research_citation="Sonnentag & Fritz, 2015; Immordino-Yang et al., 2012 — default mode network",
    ),

    # ===== QUICK CHALLENGE =====
    Intervention(
        id="quick_challenge_5",
        category="quick_challenge",
        name="Quick challenge",
        tagline="30-second creative sprint",
        description=(
            "Set a 30-second timer. Draw the first thing that comes to mind — "
            "an animal, your lunch, your shoe, anything. Time's up? Do another. "
            "Repeat 3-5 times. No quality judgment — speed is the point."
        ),
        why_it_works=(
            "Timed creative bursts shift your brain from the stress response "
            "(threat-focused) to play mode (exploration-focused). The time "
            "pressure makes it a game, not 'art.' Research shows even brief "
            "creative engagement reduces cortisol regardless of skill level."
        ),
        duration_min=5,
        needs_audio=False,
        needs_writing=False,
        best_for=["mind"],
        settings=["break_room", "home"],
        research_citation="Kaimal et al., 2016 — cortisol reduction from art-making",
    ),
    Intervention(
        id="quick_challenge_10",
        category="quick_challenge",
        name="Quick challenge + story",
        tagline="Draw it, then give it a backstory",
        description=(
            "Do 3 quick 30-second drawings. Then pick your favorite and "
            "write 3 sentences about it — give it a name, a story, a reason "
            "to exist. Turn a doodle into a character."
        ),
        why_it_works=(
            "Adding narrative to visual creation engages multiple brain regions "
            "simultaneously — visual, language, imagination. This 'whole-brain' "
            "activation is what makes creative activities more restorative than "
            "passive rest like scrolling your phone."
        ),
        duration_min=10,
        needs_audio=False,
        needs_writing=True,
        best_for=["mind", "body"],
        settings=["break_room", "home"],
        research_citation="Kaimal et al., 2016; Magsamen & Ross, 2023",
    ),
]

# Category metadata for UI display
CATEGORIES = {
    "body_sketch": {
        "name": "Body sketch",
        "icon": "pencil",
        "color": "teal",
        "short_desc": "Draw where you feel tension",
    },
    "shift_debrief": {
        "name": "Shift debrief",
        "icon": "edit",
        "color": "purple",
        "short_desc": "Write it out, let it go",
    },
    "sound_reset": {
        "name": "Sound reset",
        "icon": "headphones",
        "color": "blue",
        "short_desc": "Let rhythm calm your system",
    },
    "three_thing": {
        "name": "3-thing notice",
        "icon": "eye",
        "color": "amber",
        "short_desc": "Shift your brain out of work mode",
    },
    "quick_challenge": {
        "name": "Quick challenge",
        "icon": "zap",
        "color": "coral",
        "short_desc": "30-second creative sprint",
    },
}


# ---------------------------------------------------------------------------
# Scale-aware targeting helpers
# ---------------------------------------------------------------------------

# Subscale elevation threshold (normalized 0-10 scale)
_SUBSCALE_HIGH = 6.0

# Maps elevated BAT-12 / OLBI subscales to the intervention categories they
# most directly address, based on what each activity restores.
_SUBSCALE_CATEGORY_MAP: dict[str, list[str]] = {
    # BAT-12
    "exhaustion":            ["body_sketch", "sound_reset"],
    "mental_distance":       ["three_thing", "quick_challenge"],
    "cognitive_impairment":  ["sound_reset", "shift_debrief"],
    "emotional_impairment":  ["shift_debrief", "three_thing"],
    # OLBI
    "disengagement":         ["quick_challenge", "three_thing"],
    # OLBI exhaustion key shares name with BAT-12 — both map the same way
}


def _boosted_categories_from_scales(burnout_scales: Optional[dict]) -> Set[str]:
    """
    Return the set of category IDs that subscale scores specifically elevate.

    Reads normalized_0_10 values from bat12 and olbi subscale dicts and
    returns any category mapped to a subscale whose score exceeds the
    elevation threshold.
    """
    if not burnout_scales:
        return set()

    boosted: Set[str] = set()

    for scale_key in ("bat12", "olbi"):
        scale_data = burnout_scales.get(scale_key)
        if not scale_data:
            continue
        subscales = scale_data.get("subscales", {})
        for subscale_name, subscale_scores in subscales.items():
            if subscale_scores is None:
                continue
            normalized = subscale_scores.get("normalized_0_10")
            if normalized is not None and normalized >= _SUBSCALE_HIGH:
                for cat in _SUBSCALE_CATEGORY_MAP.get(subscale_name, []):
                    boosted.add(cat)

    return boosted


# ---------------------------------------------------------------------------
# Filtering and presentation
# ---------------------------------------------------------------------------

def get_available_interventions(
    time_available_min: int = 10,
    setting: str = "break_room",
    dominant_dimension: Optional[str] = None,
    burnout_score=None,
) -> list[dict]:
    """
    Return all 5 intervention categories with the best-fit variant for each,
    filtered by time and setting.

    The worker always sees all 5 categories. Within each category, we pick
    the variant that fits their available time and setting. If a category
    has no variant that fits, we still show it but mark it as needing
    more time or a different setting.

    Args:
        time_available_min: Worker's reported available time.
        setting: Worker's current setting.
        dominant_dimension: If provided, categories targeting this dimension
                          are shown first (but all 5 are always shown).
        burnout_score: Optional BurnoutScore object. When present, BAT-12 and
                       OLBI subscale elevations further refine which categories
                       are marked as relevant.

    Returns:
        List of 5 dicts, each representing one category with its best variant.
    """
    # Extract subscale-boosted categories if a full BurnoutScore was provided
    scale_boosted = _boosted_categories_from_scales(
        getattr(burnout_score, "burnout_scales", None)
    )

    results = []

    for cat_id, cat_meta in CATEGORIES.items():
        # Find all variants for this category
        variants = [iv for iv in INTERVENTIONS if iv.category == cat_id]

        # Filter to variants that fit time and setting
        fitting = [
            v for v in variants
            if v.duration_min <= time_available_min
            and setting in v.settings
        ]

        # Pick the longest-duration fitting variant (maximize benefit)
        if fitting:
            best = max(fitting, key=lambda v: v.duration_min)
            available = True
        else:
            # Fall back to the shortest variant in the category
            best = min(variants, key=lambda v: v.duration_min)
            available = False

        # Relevance: dominant dimension match OR subscale elevation targets
        # this category. Both paths are shown first — all 5 always appear.
        dimension_match = dominant_dimension in best.best_for if dominant_dimension else False
        scale_match = cat_id in scale_boosted
        is_relevant = dimension_match or scale_match or (not dominant_dimension and not scale_boosted)

        results.append({
            "category_id": cat_id,
            "category_name": cat_meta["name"],
            "category_icon": cat_meta["icon"],
            "category_color": cat_meta["color"],
            "category_desc": cat_meta["short_desc"],
            "intervention": {
                "id": best.id,
                "name": best.name,
                "tagline": best.tagline,
                "description": best.description,
                "why_it_works": best.why_it_works,
                "duration_min": best.duration_min,
                "needs_audio": best.needs_audio,
                "needs_writing": best.needs_writing,
                "research_citation": best.research_citation,
            },
            "available": available,
            "is_relevant": is_relevant,
            "relevance_reason": (
                "subscale" if scale_match and not dimension_match
                else "dimension" if dimension_match
                else None
            ),
            "unavailable_reason": (
                None if available
                else f"Needs {best.duration_min} min or different setting"
            ),
        })

    # Sort: relevant categories first, then by availability
    results.sort(key=lambda r: (not r["is_relevant"], not r["available"]))

    return results


@dataclass
class InterventionFeedback:
    """Worker's response after completing an intervention."""
    intervention_id: str
    helpful: bool              # Thumbs up / thumbs down
    checkin_id: str             # Links back to the check-in that triggered it
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "intervention_id": self.intervention_id,
            "helpful": self.helpful,
            "checkin_id": self.checkin_id,
            "timestamp": self.timestamp,
        }
