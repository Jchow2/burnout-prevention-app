"""
End-to-end integration tests for the ShiftGuard backend.

Simulates a complete backend request across three scenarios:
  - High-burnout worker (with BAT-12 + OLBI responses)
  - Low-burnout worker
  - Trend detection (score history)

The integration flow is:
  raw text
    → clean_text          (text_cleaner)
    → compute_textblob_features / label_checkin_debrief  (labeler)
    → CheckInResponse.validate()  (checkin_engine)
    → compute_scale_scores()      (burnout_scales via checkin_engine)
    → compute_burnout_score()     (compound_scorer)
    → get_available_interventions()  (intervention_recommender)

Run all:
    python -m unittest discover -s src/tests -p "test_*.py" -v

Run this file only:
    python -m unittest src.tests.test_backend_integration -v
"""

from src.text_cleaner import clean_text
from src.labeler import compute_textblob_features, label_checkin_debrief
from src.checkin_engine import CheckInResponse
from src.scoring.burnout_scales import score_bat12, score_olbi
from src.compound_scorer import compute_burnout_score
from src.intervention_recommender import get_available_interventions

import sys
import unittest
from pathlib import Path

# Ensure project root is on sys.path regardless of how the runner is invoked.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Shared sample texts
# ---------------------------------------------------------------------------

_HIGH_BURNOUT_TEXT = (
    "I have been absolutely exhausted all week. My back is killing me from "
    "the constant lifting. I can barely sleep at night because the pain is "
    "so bad. Management just pushes us harder and harder with no regard for "
    "our wellbeing. I am completely burned out and I dread every single shift."
)

_LOW_BURNOUT_TEXT = (
    "Had a pretty decent week overall. Work has been manageable and the team "
    "is supportive. I got enough sleep and feel fairly good about things. "
    "Nothing major to complain about right now."
)


# ===========================================================================
# Scenario 1 — High-burnout worker (includes BAT-12 + OLBI)
# ===========================================================================

class TestBackendIntegrationHighBurnout(unittest.TestCase):
    """
    Full pipeline: high-burnout worker who fills in BAT-12 and OLBI
    in addition to the daily check-in.
    """

    def setUp(self):
        self.raw_text = _HIGH_BURNOUT_TEXT
        self.checkin = CheckInResponse(
            physical_worn_down=9,
            pain_soreness=8,
            mentally_drained=9,
            sleep_quality=2,              # inverted: 2 = almost no sleep
            work_stress=9,
            shift_safety_energy="no",
            time_available_min=10,
            setting="break_room",
            shift_type="night",
            free_text=self.raw_text,
            bat12_responses={f"bat{i}": 4 for i in range(1, 13)},  # 1–5, all 4
            olbi_responses={f"olbi{i}": 3 for i in range(1, 17)},  # 1–4, all 3
        )

    # ── Step 1: text cleaning ────────────────────────────────────────────────

    def test_step1_clean_text_returns_non_empty_string(self):
        cleaned = clean_text(self.raw_text, source="reddit")
        self.assertIsInstance(cleaned, str)
        self.assertGreater(len(cleaned.split()), 5)

    # ── Step 2: text labeling ────────────────────────────────────────────────

    def test_step2_textblob_features_has_sentiment_label(self):
        cleaned = clean_text(self.raw_text, source="reddit")
        features = compute_textblob_features(cleaned)
        self.assertIn("sentiment_label", features)
        self.assertIn(features["sentiment_label"], {"positive", "neutral", "negative"})

    def test_step2_debrief_analysis_detects_burnout_signals(self):
        result = label_checkin_debrief(self.raw_text)
        self.assertTrue(result["has_debrief"])
        self.assertGreater(result["word_count"], 0)
        # The text contains burnout-related words; at least one domain should hit
        any_domain_hit = any(
            v["count"] > 0 for v in result["domain_signals"].values()
        )
        self.assertTrue(any_domain_hit, "Expected at least one domain signal hit")

    # ── Step 3: check-in validation ──────────────────────────────────────────

    def test_step3_checkin_validates_cleanly(self):
        errors = self.checkin.validate()
        self.assertEqual(
            errors, [],
            f"Unexpected validation errors: {errors}",
        )

    # ── Step 4: burnout scale scoring ────────────────────────────────────────

    def test_step4_bat12_scores_populated(self):
        self.checkin.compute_scale_scores()
        self.assertIsNotNone(self.checkin.bat12_scores)

        d = self.checkin.bat12_scores
        self.assertEqual(d["scale"], "BAT-12")
        self.assertEqual(d["item_count"], 12)
        self.assertIn("subscales", d)
        for sub in ("exhaustion", "mental_distance", "cognitive_impairment",
                    "emotional_impairment"):
            self.assertIn(sub, d["subscales"])

    def test_step4_bat12_total_normalized_in_range(self):
        self.checkin.compute_scale_scores()
        val = self.checkin.bat12_scores["total"]["normalized_0_10"]
        self.assertIsNotNone(val)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 10.0)

    def test_step4_olbi_scores_populated(self):
        self.checkin.compute_scale_scores()
        self.assertIsNotNone(self.checkin.olbi_scores)

        d = self.checkin.olbi_scores
        self.assertEqual(d["scale"], "OLBI")
        self.assertEqual(d["item_count"], 16)
        for sub in ("exhaustion", "disengagement"):
            self.assertIn(sub, d["subscales"])

    def test_step4_olbi_total_normalized_in_range(self):
        self.checkin.compute_scale_scores()
        val = self.checkin.olbi_scores["total"]["normalized_0_10"]
        self.assertIsNotNone(val)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 10.0)

    # ── Step 5: compound score ───────────────────────────────────────────────

    def test_step5_compound_score_in_valid_range(self):
        self.checkin.compute_scale_scores()
        burnout = compute_burnout_score(self.checkin)
        self.assertIsInstance(burnout.score, (int, float))
        self.assertGreaterEqual(burnout.score, 0)
        self.assertLessEqual(burnout.score, 100)

    def test_step5_high_burnout_produces_high_or_critical_risk(self):
        self.checkin.compute_scale_scores()
        burnout = compute_burnout_score(self.checkin)
        self.assertIn(
            burnout.risk_level, {"high", "critical"},
            f"Expected high/critical risk for max-stress inputs, got {burnout.risk_level}",
        )

    # ── Step 6: recommendations ──────────────────────────────────────────────

    def test_step6_returns_five_recommendations(self):
        self.checkin.compute_scale_scores()
        burnout = compute_burnout_score(self.checkin)
        recs = get_available_interventions(
            time_available_min=self.checkin.time_available_min,
            setting=self.checkin.setting,
            dominant_dimension=burnout.dominant_dimension,
            burnout_score=burnout,
        )
        self.assertEqual(len(recs), 5)

    def test_step6_recommendations_have_required_structure(self):
        self.checkin.compute_scale_scores()
        burnout = compute_burnout_score(self.checkin)
        recs = get_available_interventions(
            time_available_min=self.checkin.time_available_min,
            setting=self.checkin.setting,
            dominant_dimension=burnout.dominant_dimension,
            burnout_score=burnout,
        )
        for rec in recs:
            self.assertIn("category_id", rec)
            self.assertIn("intervention", rec)
            self.assertIn("available", rec)
            iv = rec["intervention"]
            self.assertIn("id", iv)
            self.assertIn("name", iv)
            self.assertIn("duration_min", iv)

    def test_step6_at_least_one_recommendation_is_available(self):
        self.checkin.compute_scale_scores()
        burnout = compute_burnout_score(self.checkin)
        recs = get_available_interventions(
            time_available_min=10,
            setting="break_room",
            dominant_dimension=burnout.dominant_dimension,
        )
        self.assertTrue(
            any(r["available"] for r in recs),
            "No available interventions for 10 min / break_room",
        )

    # ── Full end-to-end assertion (compound score + ≥1 recommendation) ───────

    def test_full_pipeline_produces_compound_score_and_recommendations(self):
        """
        Master integration assertion: final output must include a numeric
        compound score and at least one intervention recommendation object.
        """
        # 1. Clean
        cleaned = clean_text(self.raw_text, source="reddit")
        self.assertTrue(cleaned)

        # 2. Label
        debrief = label_checkin_debrief(self.raw_text)
        self.assertTrue(debrief["has_debrief"])

        # 3. Validate
        errors = self.checkin.validate()
        self.assertEqual(errors, [])

        # 4. Scale scores
        self.checkin.compute_scale_scores()
        self.assertIsNotNone(self.checkin.bat12_scores)

        # 5. Compound score
        burnout = compute_burnout_score(self.checkin)
        self.assertGreater(burnout.score, 0)

        # 6. Recommendations
        recs = get_available_interventions(
            time_available_min=self.checkin.time_available_min,
            setting=self.checkin.setting,
            dominant_dimension=burnout.dominant_dimension,
            burnout_score=burnout,
        )

        # Final assertions: compound score is numeric, ≥1 recommendation exists
        self.assertIsInstance(burnout.score, (int, float))
        self.assertGreater(len(recs), 0)
        # At least one recommendation must have a valid intervention id
        ids = [r["intervention"]["id"] for r in recs]
        self.assertTrue(all(isinstance(i, str) and len(i) > 0 for i in ids))


# ===========================================================================
# Scenario 2 — Low-burnout worker (no scale responses)
# ===========================================================================

class TestBackendIntegrationLowBurnout(unittest.TestCase):
    """
    Full pipeline: worker with low burnout, no periodic scale assessment.
    """

    def setUp(self):
        self.raw_text = _LOW_BURNOUT_TEXT
        self.checkin = CheckInResponse(
            physical_worn_down=2,
            pain_soreness=1,
            mentally_drained=2,
            sleep_quality=8,
            work_stress=2,
            shift_safety_energy="yes",
            time_available_min=5,
            setting="commute",
        )

    def test_checkin_validates_cleanly(self):
        self.assertEqual(self.checkin.validate(), [])

    def test_compound_score_is_low_or_moderate(self):
        burnout = compute_burnout_score(self.checkin)
        self.assertLessEqual(
            burnout.score, 55,
            f"Expected low/moderate score for low-burnout inputs, got {burnout.score}",
        )
        self.assertIn(burnout.risk_level, {"low", "moderate"})

    def test_recommendations_still_return_five(self):
        burnout = compute_burnout_score(self.checkin)
        recs = get_available_interventions(
            time_available_min=self.checkin.time_available_min,
            setting=self.checkin.setting,
            dominant_dimension=burnout.dominant_dimension,
        )
        self.assertEqual(len(recs), 5)

    def test_full_pipeline_low_burnout(self):
        cleaned = clean_text(self.raw_text, source="reddit")
        self.assertTrue(cleaned)

        errors = self.checkin.validate()
        self.assertEqual(errors, [])

        burnout = compute_burnout_score(self.checkin)
        self.assertIsInstance(burnout.score, (int, float))
        self.assertGreaterEqual(burnout.score, 0)
        self.assertLessEqual(burnout.score, 100)

        recs = get_available_interventions(
            time_available_min=self.checkin.time_available_min,
            setting=self.checkin.setting,
            dominant_dimension=burnout.dominant_dimension,
        )
        self.assertEqual(len(recs), 5)


# ===========================================================================
# Scenario 3 — Trend detection with score history
# ===========================================================================

class TestBackendIntegrationWithHistory(unittest.TestCase):
    """
    Compound scorer's trend logic requires a history list of previous
    BurnoutScore objects.  These tests verify it produces the right
    trend labels given controlled inputs.
    """

    def _build_score(self, physical=5, pain=5, drained=5,
                     sleep=5, stress=5, safety="somewhat"):
        checkin = CheckInResponse(
            physical_worn_down=physical,
            pain_soreness=pain,
            mentally_drained=drained,
            sleep_quality=sleep,
            work_stress=stress,
            shift_safety_energy=safety,
        )
        return compute_burnout_score(checkin)

    def test_trend_worsening_when_today_spikes_above_history(self):
        # Build a history of consistently low scores
        low = self._build_score(1, 1, 1, 9, 1, "yes")
        history = [low] * 7

        # Today is catastrophically high
        high_checkin = CheckInResponse(
            physical_worn_down=10,
            pain_soreness=10,
            mentally_drained=10,
            sleep_quality=0,
            work_stress=10,
            shift_safety_energy="no",
        )
        today = compute_burnout_score(high_checkin, history=history)
        self.assertEqual(
            today.trend, "worsening",
            f"Expected worsening trend, got {today.trend} "
            f"(today={today.score:.1f}, history avg ≈ {low.score:.1f})",
        )

    def test_trend_improving_when_today_drops_below_history(self):
        # History of high scores
        high = self._build_score(9, 9, 9, 1, 9, "no")
        history = [high] * 7

        # Today is very low
        low_checkin = CheckInResponse(
            physical_worn_down=0,
            pain_soreness=0,
            mentally_drained=0,
            sleep_quality=10,
            work_stress=0,
            shift_safety_energy="yes",
        )
        today = compute_burnout_score(low_checkin, history=history)
        self.assertEqual(
            today.trend, "improving",
            f"Expected improving trend, got {today.trend} "
            f"(today={today.score:.1f}, history avg ≈ {high.score:.1f})",
        )

    def test_trend_stable_when_score_unchanged(self):
        mid = self._build_score(5, 5, 5, 5, 5, "somewhat")
        history = [mid] * 7
        today   = compute_burnout_score(
            CheckInResponse(
                physical_worn_down=5,
                pain_soreness=5,
                mentally_drained=5,
                sleep_quality=5,
                work_stress=5,
                shift_safety_energy="somewhat",
            ),
            history=history,
        )
        self.assertEqual(today.trend, "stable")

    def test_consecutive_high_days_counted_correctly(self):
        high = self._build_score(9, 9, 9, 1, 9, "no")
        # 4 consecutive high-scoring days in history
        history = [high] * 4
        today   = compute_burnout_score(
            CheckInResponse(
                physical_worn_down=9,
                pain_soreness=9,
                mentally_drained=9,
                sleep_quality=1,
                work_stress=9,
                shift_safety_energy="no",
            ),
            history=history,
        )
        # Must have counted at least 4 consecutive high days
        self.assertGreaterEqual(today.consecutive_high_days, 4)


# ===========================================================================
# Scenario 4 — Scale scoring in isolation (unit-level integration)
# ===========================================================================

class TestBurnoutScaleIntegration(unittest.TestCase):
    """
    Directly exercises score_bat12 and score_olbi — the functions that
    compute_scale_scores() delegates to — to verify the integration
    path from raw dict → serialisable output.
    """

    def test_bat12_high_scores_above_bat12_low(self):
        high = score_bat12({f"bat{i}": 5 for i in range(1, 13)})
        low  = score_bat12({f"bat{i}": 1 for i in range(1, 13)})
        self.assertGreater(
            high.total.normalized_0_10,
            low.total.normalized_0_10,
        )

    def test_olbi_high_scores_non_trivially(self):
        # All items at max (4) — after reverse-scoring some items, total
        # should still be non-None and in range
        result = score_olbi({f"olbi{i}": 4 for i in range(1, 17)})
        self.assertIsNotNone(result.total.normalized_0_10)
        self.assertGreaterEqual(result.total.normalized_0_10, 0.0)
        self.assertLessEqual(result.total.normalized_0_10, 10.0)

    def test_scale_scores_fed_into_checkin_burnout_scales_property(self):
        checkin = CheckInResponse(
            bat12_responses={f"bat{i}": 4 for i in range(1, 13)},
            olbi_responses={f"olbi{i}": 3 for i in range(1, 17)},
        )
        self.assertIsNone(checkin.burnout_scales,
                          "burnout_scales should be None before compute_scale_scores()")
        checkin.compute_scale_scores()
        scales = checkin.burnout_scales
        self.assertIsNotNone(scales)
        self.assertIn("bat12", scales)
        self.assertIn("olbi",  scales)

    def test_burnout_scales_feed_into_compound_scorer(self):
        checkin = CheckInResponse(
            physical_worn_down=7,
            pain_soreness=7,
            mentally_drained=7,
            sleep_quality=3,
            work_stress=7,
            shift_safety_energy="no",
            bat12_responses={f"bat{i}": 4 for i in range(1, 13)},
            olbi_responses={f"olbi{i}": 3 for i in range(1, 17)},
        )
        checkin.compute_scale_scores()
        burnout = compute_burnout_score(checkin)
        # burnout_scales on the BurnoutScore object should mirror what
        # checkin.burnout_scales contains
        self.assertIsNotNone(burnout.burnout_scales)
        self.assertIn("bat12", burnout.burnout_scales)
        self.assertIn("olbi",  burnout.burnout_scales)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)