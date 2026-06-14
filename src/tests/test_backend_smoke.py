"""
Smoke tests for each backend module.

One class per module. Each test targets a real exported symbol — names were
verified by reading the source files before writing any test.

Run all:
    python -m unittest discover -s src/tests -p "test_*.py" -v

Run this file only:
    python -m unittest src.tests.test_backend_smoke -v
"""

import pandas as pd

from src.text_cleaner import clean_text, clean_dataframe
from src.labeler import (
    compute_textblob_features,
    count_keyword_themes,
    label_checkin_debrief,
)
from src.checkin_engine import CheckInResponse
from src.scoring.burnout_scales import (
    score_bat12,
    score_olbi,
    Bat12Score,
    OlbiScore,
    ScaleSubscore,
)
from src.compound_scorer import compute_burnout_score, BurnoutScore
from src.intervention_recommender import get_available_interventions
from src.merger import assert_no_youtube, merge_sources, ALLOWED_SOURCES

import sys
import math  # noqa: F401
import unittest
from pathlib import Path

# Ensure project root is on sys.path regardless of how the runner is invoked.
# File lives at src/tests/test_backend_smoke.py → three .parent calls → root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Sentiment model depends on sentence_transformers (large dep).
# Import is guarded so the rest of the suite runs even if it is missing.
try:
    from src.sentiment_model import WorkforceSentimentModel as _WSM
    _HAS_SENTIMENT_MODEL = True
except ImportError:
    _WSM = None
    _HAS_SENTIMENT_MODEL = False

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Full BAT-12 response sets  (bat1..bat12, scale 1–5)
_BAT12_HIGH = {f"bat{i}": 4 for i in range(1, 13)}
_BAT12_LOW  = {f"bat{i}": 2 for i in range(1, 13)}

# Full OLBI response sets (olbi1..olbi16, scale 1–4)
_OLBI_HIGH  = {f"olbi{i}": 3 for i in range(1, 17)}
_OLBI_LOW   = {f"olbi{i}": 2 for i in range(1, 17)}


def _high_checkin() -> CheckInResponse:
    return CheckInResponse(
        physical_worn_down=9,
        pain_soreness=8,
        mentally_drained=9,
        sleep_quality=1,          # inverted: 1 = barely slept
        work_stress=9,
        shift_safety_energy="no",
        time_available_min=10,
        setting="break_room",
    )


def _low_checkin() -> CheckInResponse:
    return CheckInResponse(
        physical_worn_down=1,
        pain_soreness=1,
        mentally_drained=1,
        sleep_quality=9,          # inverted: 9 = great sleep
        work_stress=1,
        shift_safety_energy="yes",
        time_available_min=10,
        setting="break_room",
    )


# ===========================================================================
# 1. text_cleaner — clean_text / clean_dataframe
# ===========================================================================

class TestTextCleaner(unittest.TestCase):

    def test_returns_non_empty_string_for_normal_input(self):
        result = clean_text("I worked a brutal twelve hour shift today and I am exhausted.")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_empty_string_returns_empty(self):
        self.assertEqual(clean_text(""), "")

    def test_nan_float_returns_empty(self):
        self.assertEqual(clean_text(float("nan")), "")

    def test_url_is_stripped(self):
        result = clean_text("Check https://example.com for details about this awful job.")
        self.assertNotIn("https://", result)
        self.assertNotIn("example.com", result)

    def test_email_is_stripped(self):
        result = clean_text("Contact hr@company.com about the new unfair policy today.")
        self.assertNotIn("@company.com", result)

    def test_reddit_markdown_stripped_with_source(self):
        result = clean_text(
            "**Very tired** from work, feeling *completely broken* today.",
            source="reddit",
        )
        self.assertNotIn("**", result)
        self.assertNotIn("*", result)
        self.assertIn("Very tired", result)

    def test_text_shorter_than_3_words_returns_empty(self):
        # clean_text drops text with fewer than 3 words after cleaning
        self.assertEqual(clean_text("hi ok"), "")

    def test_unicode_smart_quotes_normalized(self):
        result = clean_text(
            "“Hello world” — this job is genuinely terrible for everyone."
        )
        self.assertNotIn("“", result)
        self.assertNotIn("”", result)
        self.assertNotIn("—", result)

    def test_clean_dataframe_drops_short_rows(self):
        df = pd.DataFrame({
            "text":   ["hi", "I genuinely hate this awful terrible job so much"],
            "source": ["reddit", "reddit"],
        })
        result = clean_dataframe(df, min_words=5)
        self.assertEqual(len(result), 1)
        self.assertIn("hate", result["text"].iloc[0])

    def test_clean_dataframe_deduplicates(self):
        text = "I am absolutely exhausted and burned out from this warehouse job."
        df = pd.DataFrame({
            "text":   [text, text],
            "source": ["reddit", "glassdoor"],
        })
        result = clean_dataframe(df)
        self.assertEqual(len(result), 1)


# ===========================================================================
# 2. labeler — compute_textblob_features / label_checkin_debrief
# ===========================================================================

class TestLabeler(unittest.TestCase):

    def test_textblob_features_returns_expected_keys(self):
        result = compute_textblob_features("I enjoy my job and love the friendly team.")
        for key in ("polarity", "subjectivity", "sentiment_label", "confidence"):
            self.assertIn(key, result)

    def test_textblob_positive_text_labelled_positive(self):
        result = compute_textblob_features(
            "Amazing benefits, great management, wonderful team culture here."
        )
        self.assertEqual(result["sentiment_label"], "positive")
        self.assertGreater(result["polarity"], 0)

    def test_textblob_negative_text_labelled_negative(self):
        result = compute_textblob_features(
            "Terrible management, horrible conditions, awful miserable job."
        )
        self.assertEqual(result["sentiment_label"], "negative")
        self.assertLess(result["polarity"], 0)

    def test_textblob_empty_text_returns_neutral_zero(self):
        result = compute_textblob_features("")
        self.assertEqual(result["sentiment_label"], "neutral")
        self.assertEqual(result["polarity"], 0.0)
        self.assertEqual(result["confidence"], 0.0)

    def test_keyword_themes_returns_dict_with_theme_keys(self):
        result = count_keyword_themes(
            "I am completely burned out and exhausted from this overwork situation."
        )
        self.assertIsInstance(result, dict)
        # All expected theme keys must be present
        for key in ("burnout_keywords", "physical_demands_keywords", "management_issues_keywords",
                    "workload_keywords", "environment_keywords"):
            self.assertIn(key, result)

    def test_keyword_themes_burnout_hit(self):
        result = count_keyword_themes("I feel burned out and drained every single day.")
        self.assertGreater(result["burnout_keywords"], 0)

    def test_label_checkin_debrief_empty_string(self):
        result = label_checkin_debrief("")
        self.assertFalse(result["has_debrief"])
        self.assertEqual(result["word_count"], 0)
        self.assertIsNone(result["sentiment"])

    def test_label_checkin_debrief_none(self):
        result = label_checkin_debrief(None)
        self.assertFalse(result["has_debrief"])

    def test_label_checkin_debrief_with_text(self):
        result = label_checkin_debrief(
            "Totally exhausted and drained today. Cannot go on like this at work."
        )
        self.assertTrue(result["has_debrief"])
        self.assertIsNotNone(result["sentiment"])
        self.assertIn("sentiment_label", result["sentiment"])
        self.assertIsInstance(result["domain_signals"], dict)
        self.assertGreater(result["word_count"], 0)

    def test_label_checkin_debrief_domain_signals_have_expected_keys(self):
        result = label_checkin_debrief(
            "Exhausted and drained. Can't focus. Counting down until I quit."
        )
        # domain_signals keys must match BAT-12/OLBI subscale names
        for domain in ("exhaustion", "mental_distance", "cognitive_impairment",
                       "emotional_impairment", "disengagement"):
            self.assertIn(domain, result["domain_signals"])


# ===========================================================================
# 3. checkin_engine — CheckInResponse
# ===========================================================================

class TestCheckinEngine(unittest.TestCase):

    def test_default_checkin_passes_validation(self):
        errors = CheckInResponse().validate()
        self.assertEqual(errors, [])

    def test_high_burnout_checkin_passes_validation(self):
        errors = _high_checkin().validate()
        self.assertEqual(errors, [])

    def test_low_burnout_checkin_passes_validation(self):
        errors = _low_checkin().validate()
        self.assertEqual(errors, [])

    def test_physical_worn_down_above_10_is_invalid(self):
        errors = CheckInResponse(physical_worn_down=11).validate()
        self.assertTrue(any("physical_worn_down" in e for e in errors))

    def test_negative_score_is_invalid(self):
        errors = CheckInResponse(pain_soreness=-1).validate()
        self.assertTrue(any("pain_soreness" in e for e in errors))

    def test_invalid_safety_energy_value(self):
        errors = CheckInResponse(shift_safety_energy="maybe").validate()
        self.assertTrue(any("shift_safety_energy" in e for e in errors))

    def test_invalid_time_available(self):
        errors = CheckInResponse(time_available_min=7).validate()
        self.assertTrue(any("time_available_min" in e for e in errors))

    def test_invalid_setting(self):
        errors = CheckInResponse(setting="spaceship").validate()
        self.assertTrue(any("setting" in e for e in errors))

    def test_body_score_in_0_10(self):
        for checkin in [_high_checkin(), _low_checkin(), CheckInResponse()]:
            self.assertGreaterEqual(checkin.body_score, 0, "body_score below 0")
            self.assertLessEqual(checkin.body_score, 10, "body_score above 10")

    def test_mind_score_in_0_10(self):
        for checkin in [_high_checkin(), _low_checkin(), CheckInResponse()]:
            self.assertGreaterEqual(checkin.mind_score, 0)
            self.assertLessEqual(checkin.mind_score, 10)

    def test_life_score_in_0_10(self):
        for checkin in [_high_checkin(), _low_checkin(), CheckInResponse()]:
            self.assertGreaterEqual(checkin.life_score, 0)
            self.assertLessEqual(checkin.life_score, 10)

    def test_high_burnout_body_score_above_low(self):
        self.assertGreater(_high_checkin().body_score, _low_checkin().body_score)

    def test_compute_scale_scores_bat12_populates_scores(self):
        checkin = CheckInResponse(bat12_responses=_BAT12_HIGH)
        self.assertEqual(checkin.validate(), [])
        checkin.compute_scale_scores()
        self.assertIsNotNone(checkin.bat12_scores)
        self.assertIn("subscales", checkin.bat12_scores)
        self.assertIn("total", checkin.bat12_scores)

    def test_compute_scale_scores_olbi_populates_scores(self):
        checkin = CheckInResponse(olbi_responses=_OLBI_HIGH)
        self.assertEqual(checkin.validate(), [])
        checkin.compute_scale_scores()
        self.assertIsNotNone(checkin.olbi_scores)
        self.assertIn("subscales", checkin.olbi_scores)

    def test_to_dict_has_required_top_level_keys(self):
        d = _high_checkin().to_dict()
        for key in ("checkin_id", "timestamp", "core_responses", "dimensions",
                    "intervention_preferences", "context"):
            self.assertIn(key, d)

    def test_to_dict_dimensions_have_body_mind_life(self):
        d = _high_checkin().to_dict()
        for dim in ("body", "mind", "life"):
            self.assertIn(dim, d["dimensions"])

    def test_bat12_validation_catches_out_of_range(self):
        bad = {f"bat{i}": 6 for i in range(1, 13)}   # 6 > max of 5
        errors = CheckInResponse(bat12_responses=bad).validate()
        self.assertTrue(len(errors) > 0)

    def test_olbi_validation_catches_out_of_range(self):
        bad = {f"olbi{i}": 5 for i in range(1, 17)}  # 5 > max of 4
        errors = CheckInResponse(olbi_responses=bad).validate()
        self.assertTrue(len(errors) > 0)


# ===========================================================================
# 4a. burnout_scales — BAT-12  (score_bat12)
# ===========================================================================

class TestBurnoutScalesBat12(unittest.TestCase):

    def test_returns_bat12_score_instance(self):
        self.assertIsInstance(score_bat12(_BAT12_HIGH), Bat12Score)

    def test_all_four_subscales_present(self):
        result = score_bat12(_BAT12_HIGH)
        for sub in ("exhaustion", "mental_distance", "cognitive_impairment",
                    "emotional_impairment"):
            self.assertIsInstance(getattr(result, sub), ScaleSubscore)

    def test_total_normalized_in_0_10(self):
        for responses in (_BAT12_HIGH, _BAT12_LOW):
            val = score_bat12(responses).total.normalized_0_10
            self.assertIsNotNone(val)
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 10.0)

    def test_high_responses_score_above_low_responses(self):
        high = score_bat12(_BAT12_HIGH).total.normalized_0_10
        low  = score_bat12(_BAT12_LOW).total.normalized_0_10
        self.assertGreater(high, low)

    def test_item_count_is_12(self):
        self.assertEqual(score_bat12(_BAT12_HIGH).item_count, 12)

    def test_partial_responses_do_not_raise(self):
        # Providing only 6 of 12 items should score gracefully
        partial = {f"bat{i}": 3 for i in range(1, 7)}
        result = score_bat12(partial)
        self.assertIsInstance(result, Bat12Score)

    def test_empty_responses_returns_none_for_means(self):
        result = score_bat12({})
        self.assertIsNone(result.total.normalized_0_10)

    def test_to_dict_shape(self):
        d = score_bat12(_BAT12_HIGH).to_dict()
        self.assertEqual(d["scale"], "BAT-12")
        self.assertEqual(d["item_count"], 12)
        self.assertIn("subscales", d)
        self.assertIn("total", d)
        for sub in ("exhaustion", "mental_distance", "cognitive_impairment",
                    "emotional_impairment"):
            self.assertIn(sub, d["subscales"])


# ===========================================================================
# 4b. burnout_scales — OLBI  (score_olbi)
# ===========================================================================

class TestBurnoutScalesOlbi(unittest.TestCase):

    def test_returns_olbi_score_instance(self):
        self.assertIsInstance(score_olbi(_OLBI_HIGH), OlbiScore)

    def test_exhaustion_and_disengagement_subscales_present(self):
        result = score_olbi(_OLBI_HIGH)
        self.assertIsInstance(result.exhaustion, ScaleSubscore)
        self.assertIsInstance(result.disengagement, ScaleSubscore)

    def test_total_normalized_in_0_10(self):
        for responses in (_OLBI_HIGH, _OLBI_LOW):
            val = score_olbi(responses).total.normalized_0_10
            self.assertIsNotNone(val)
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 10.0)

    def test_item_count_is_16(self):
        self.assertEqual(score_olbi(_OLBI_HIGH).item_count, 16)

    def test_partial_responses_do_not_raise(self):
        partial = {f"olbi{i}": 2 for i in range(1, 9)}
        result = score_olbi(partial)
        self.assertIsInstance(result, OlbiScore)

    def test_empty_responses_returns_none_for_means(self):
        result = score_olbi({})
        self.assertIsNone(result.total.normalized_0_10)

    def test_to_dict_shape(self):
        d = score_olbi(_OLBI_HIGH).to_dict()
        self.assertEqual(d["scale"], "OLBI")
        self.assertEqual(d["item_count"], 16)
        self.assertIn("subscales", d)
        for sub in ("exhaustion", "disengagement"):
            self.assertIn(sub, d["subscales"])


# ===========================================================================
# 5. compound_scorer — compute_burnout_score
# ===========================================================================

class TestCompoundScorer(unittest.TestCase):

    def test_returns_burnout_score_instance(self):
        self.assertIsInstance(compute_burnout_score(_high_checkin()), BurnoutScore)

    def test_score_always_between_0_and_100(self):
        for checkin in [_high_checkin(), _low_checkin(), CheckInResponse()]:
            result = compute_burnout_score(checkin)
            self.assertGreaterEqual(result.score, 0,   f"score < 0 for {checkin}")
            self.assertLessEqual(result.score,   100,  f"score > 100 for {checkin}")

    def test_high_burnout_scores_above_low_burnout(self):
        high = compute_burnout_score(_high_checkin()).score
        low  = compute_burnout_score(_low_checkin()).score
        self.assertGreater(high, low)

    def test_risk_level_is_valid_string(self):
        valid = {"low", "moderate", "high", "critical"}
        for checkin in [_high_checkin(), _low_checkin()]:
            self.assertIn(compute_burnout_score(checkin).risk_level, valid)

    def test_high_burnout_risk_level_is_high_or_critical(self):
        result = compute_burnout_score(_high_checkin())
        self.assertIn(result.risk_level, {"high", "critical"})

    def test_low_burnout_risk_level_is_low_or_moderate(self):
        result = compute_burnout_score(_low_checkin())
        self.assertIn(result.risk_level, {"low", "moderate"})

    def test_dominant_dimension_is_valid(self):
        valid = {"body", "mind", "life"}
        result = compute_burnout_score(_high_checkin())
        self.assertIn(result.dominant_dimension, valid)

    def test_dimension_scores_in_0_100(self):
        result = compute_burnout_score(_high_checkin())
        for val in (result.body_score, result.mind_score, result.life_score):
            self.assertGreaterEqual(val, 0)
            self.assertLessEqual(val, 100)

    def test_compound_multiplier_at_least_one(self):
        result = compute_burnout_score(_high_checkin())
        self.assertGreaterEqual(result.compound_multiplier, 1.0)

    def test_interaction_pairs_is_list(self):
        result = compute_burnout_score(_high_checkin())
        self.assertIsInstance(result.interaction_pairs, list)

    def test_trend_is_none_without_history(self):
        result = compute_burnout_score(_high_checkin(), history=None)
        self.assertIsNone(result.trend)

    def test_trend_is_set_with_sufficient_history(self):
        history = [compute_burnout_score(_high_checkin()) for _ in range(5)]
        result  = compute_burnout_score(_high_checkin(), history=history)
        self.assertIn(result.trend, {"worsening", "improving", "stable"})

    def test_to_dict_has_required_keys(self):
        d = compute_burnout_score(_high_checkin()).to_dict()
        for key in ("score", "risk_level", "dimensions", "dominant_dimension",
                    "compound_multiplier", "interaction_pairs", "trend"):
            self.assertIn(key, d)


# ===========================================================================
# 6. intervention_recommender — get_available_interventions
# ===========================================================================

class TestInterventionRecommender(unittest.TestCase):

    _EXPECTED_CATEGORIES = frozenset({
        "body_sketch", "shift_debrief", "sound_reset",
        "three_thing", "quick_challenge",
    })

    def test_always_returns_exactly_five_items(self):
        for time, setting in [(5, "commute"), (10, "break_room"), (15, "home")]:
            result = get_available_interventions(
                time_available_min=time, setting=setting
            )
            self.assertEqual(
                len(result), 5,
                f"Expected 5 results, got {len(result)} for time={time} setting={setting}",
            )

    def test_category_ids_are_the_five_expected(self):
        result = get_available_interventions()
        ids = {r["category_id"] for r in result}
        self.assertEqual(ids, self._EXPECTED_CATEGORIES)

    def test_each_result_has_required_keys(self):
        for item in get_available_interventions():
            for key in ("category_id", "category_name", "intervention",
                        "available", "is_relevant", "unavailable_reason"):
                self.assertIn(key, item, f"Missing '{key}' in result")

    def test_nested_intervention_has_required_keys(self):
        for item in get_available_interventions():
            iv = item["intervention"]
            for key in ("id", "name", "tagline", "description",
                        "duration_min", "needs_audio", "needs_writing",
                        "research_citation"):
                self.assertIn(key, iv, f"Missing '{key}' in intervention dict")

    def test_relevant_categories_sorted_before_non_relevant(self):
        burnout = compute_burnout_score(_high_checkin())
        result  = get_available_interventions(
            time_available_min=10,
            setting="break_room",
            dominant_dimension=burnout.dominant_dimension,
            burnout_score=burnout,
        )
        flags = [r["is_relevant"] for r in result]
        # Once we see a False, no True should follow
        seen_false = False
        for f in flags:
            if not f:
                seen_false = True
            if seen_false and f:
                self.fail(
                    "A non-relevant category appeared before a relevant one: "
                    + str(flags)
                )

    def test_at_least_one_category_available_for_10min_break_room(self):
        result = get_available_interventions(
            time_available_min=10, setting="break_room"
        )
        self.assertTrue(any(r["available"] for r in result))

    def test_unavailable_items_have_unavailable_reason(self):
        # 5 min commute — sound_reset needs audio but commute is valid;
        # body_sketch_10 needs 10 min, so it won't fit in 5 min.
        result = get_available_interventions(
            time_available_min=5, setting="commute"
        )
        for item in result:
            if not item["available"]:
                self.assertIsNotNone(
                    item["unavailable_reason"],
                    f"unavailable_reason is None for {item['category_id']}",
                )

    def test_available_items_have_null_unavailable_reason(self):
        result = get_available_interventions(
            time_available_min=15, setting="home"
        )
        for item in result:
            if item["available"]:
                self.assertIsNone(
                    item["unavailable_reason"],
                    f"unavailable_reason should be None for available {item['category_id']}",
                )


# ===========================================================================
# 7. merger — assert_no_youtube / merge_sources / ALLOWED_SOURCES
# ===========================================================================

class TestMerger(unittest.TestCase):

    def test_assert_no_youtube_raises_on_youtube_source(self):
        df = pd.DataFrame({"source": ["youtube", "reddit"]})
        with self.assertRaises(ValueError) as ctx:
            assert_no_youtube(df, stage="test")
        self.assertIn("youtube", str(ctx.exception).lower())

    def test_assert_no_youtube_raises_on_unknown_source(self):
        df = pd.DataFrame({"source": ["twitter"]})
        with self.assertRaises(ValueError):
            assert_no_youtube(df, stage="test")

    def test_assert_no_youtube_does_not_raise_for_valid_sources(self):
        df = pd.DataFrame({"source": ["reddit", "glassdoor", "reddit"]})
        assert_no_youtube(df, stage="test")   # must not raise

    def test_assert_no_youtube_skips_when_no_source_column(self):
        df = pd.DataFrame({"text": ["hello world"]})
        assert_no_youtube(df, stage="test")   # must not raise

    def test_allowed_sources_contains_reddit_and_glassdoor(self):
        self.assertIn("reddit",    ALLOWED_SOURCES)
        self.assertIn("glassdoor", ALLOWED_SOURCES)

    def test_allowed_sources_excludes_youtube(self):
        self.assertNotIn("youtube", ALLOWED_SOURCES)

    def test_merge_sources_reddit_only(self):
        df = pd.DataFrame({
            "review_id": ["r1", "r2"],
            "text": [
                "Working here is an absolute nightmare with no proper breaks.",
                "Exhausted from the constant mandatory overtime every single week.",
            ],
        })
        merged = merge_sources({"reddit": df})
        self.assertFalse(merged.empty)
        self.assertTrue((merged["source"] == "reddit").all())

    def test_merge_sources_reddit_and_glassdoor(self):
        reddit_df = pd.DataFrame({
            "review_id": ["r1"],
            "text": ["Terrible working conditions at this warehouse every day."],
        })
        gd_df = pd.DataFrame({
            "review_id": ["g1"],
            "text": ["Great company with excellent healthcare and PTO benefits."],
        })
        merged = merge_sources({"reddit": reddit_df, "glassdoor": gd_df})
        self.assertEqual(len(merged), 2)
        self.assertIn("reddit",    merged["source"].values)
        self.assertIn("glassdoor", merged["source"].values)

    def test_merge_sources_skips_empty_dataframe(self):
        reddit_df = pd.DataFrame({
            "review_id": ["r1"],
            "text": ["Really tough job with very long mandatory overtime shifts."],
        })
        merged = merge_sources({"reddit": reddit_df, "glassdoor": pd.DataFrame()})
        self.assertFalse(merged.empty)
        self.assertEqual(len(merged), 1)

    def test_merge_sources_all_empty_returns_empty_with_base_columns(self):
        merged = merge_sources({"reddit": pd.DataFrame(), "glassdoor": pd.DataFrame()})
        self.assertTrue(merged.empty)

    def test_merge_sources_deduplication(self):
        same_text = "I absolutely hate working here and want to quit immediately."
        df = pd.DataFrame({
            "review_id": ["r1"],
            "text": [same_text],
        })
        gd = pd.DataFrame({
            "review_id": ["g1"],
            "text": [same_text],
        })
        merged = merge_sources({"reddit": df, "glassdoor": gd}, deduplicate=True)
        self.assertEqual(len(merged), 1)


# ===========================================================================
# 8. sentiment_model — optional (guarded import)
# ===========================================================================

@unittest.skipUnless(
    _HAS_SENTIMENT_MODEL,
    "sentence_transformers not installed — skipping WorkforceSentimentModel tests",
)
class TestSentimentModelOptional(unittest.TestCase):

    def test_class_is_importable(self):
        self.assertIsNotNone(_WSM)

    def test_has_extract_features_callable(self):
        # Confirms the primary feature-extraction method exists without
        # instantiating the model (which would trigger a model download)
        self.assertTrue(
            callable(getattr(_WSM, "extract_features", None)),
            "WorkforceSentimentModel.extract_features should be callable",
        )

    def test_has_prepare_training_data_callable(self):
        self.assertTrue(
            callable(getattr(_WSM, "prepare_training_data", None)),
            "WorkforceSentimentModel.prepare_training_data should be callable",
        )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)