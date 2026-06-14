# Backend Test Plan — ShiftGuard

Discovered by reading each module directly. All function and class names are
verified against the actual source files, not guessed.

---

## Discovered Public APIs

### `src/scoring/burnout_scales.py`

| Symbol | Kind | Signature | Notes |
|---|---|---|---|
| `score_bat12` | function | `(responses: Dict[str, int]) -> Bat12Score` | Keys: `bat1`..`bat12`, values `1–5` |
| `score_olbi` | function | `(responses: Dict[str, int]) -> OlbiScore` | Keys: `olbi1`..`olbi16`, values `1–4` |
| `Bat12Score` | dataclass | `.exhaustion`, `.mental_distance`, `.cognitive_impairment`, `.emotional_impairment`, `.total`, `.item_count`, `.to_dict()` | Each subscale is a `ScaleSubscore` |
| `OlbiScore` | dataclass | `.exhaustion`, `.disengagement`, `.total`, `.item_count`, `.to_dict()` | Each subscale is a `ScaleSubscore` |
| `ScaleSubscore` | dataclass | `.raw_mean`, `.normalized_0_10`, `.n_items` | `normalized_0_10` is always in `[0, 10]` |

> **The symbol that was failing (`compute_burnout_scores`) does not exist.**
> The correct functions are `score_bat12` and `score_olbi`.

---

### `src/checkin_engine.py`

| Symbol | Kind | Notes |
|---|---|---|
| `CheckInResponse` | dataclass | Core daily check-in model |
| `.validate()` | method | Returns `list[str]` — empty = valid |
| `.compute_scale_scores()` | method | Runs `score_bat12` / `score_olbi` if `bat12_responses` / `olbi_responses` are set; populates `.bat12_scores` / `.olbi_scores` |
| `.body_score` | property | `float`, `0–10`; combines `physical_worn_down`, `pain_soreness`, inverted `sleep_quality` |
| `.mind_score` | property | `float`, `0–10`; combines `mentally_drained`, `work_stress`, `shift_safety_energy` numeric |
| `.life_score` | property | `float`, `0–10`; derived from `body`+`mind` with sleep amplifier |
| `.to_dict()` | method | Serializable dict with `checkin_id`, `timestamp`, `core_responses`, `dimensions`, etc. |

**Valid field values**

| Field | Type | Valid values |
|---|---|---|
| `physical_worn_down` … `work_stress` | `int` | `0–10` |
| `sleep_quality` | `int` | `0–10` (inverted: 10 = great) |
| `shift_safety_energy` | `str` | `"yes"` / `"somewhat"` / `"no"` |
| `intervention_mood` | `str` | `"calming"` / `"energizing"` / `"distracting"` |
| `intervention_modality` | `str` | `"sound"` / `"visuals"` / `"reflection"` |
| `time_available_min` | `int` | `5`, `10`, or `15` |
| `setting` | `str` | `"break_room"` / `"commute"` / `"home"` / `"other"` |

---

### `src/compound_scorer.py`

| Symbol | Kind | Signature | Notes |
|---|---|---|---|
| `compute_burnout_score` | function | `(checkin, history=None) -> BurnoutScore` | `checkin` needs `.body_score`, `.mind_score`, `.life_score` properties |
| `BurnoutScore` | dataclass | `.score` (float, 0–100), `.risk_level`, `.body_score`, `.mind_score`, `.life_score`, `.dominant_dimension`, `.compound_multiplier`, `.interaction_pairs`, `.trend`, `.consecutive_high_days`, `.burnout_scales`, `.to_dict()` | |

**Risk levels:** `"low"` (0–30), `"moderate"` (30–55), `"high"` (55–75), `"critical"` (75–100)

**Compound multipliers** apply when two or more dimensions exceed 55:
- `body+mind`: 1.15×, `body+life`: 1.12×, `mind+life`: 1.10×, all three: 1.25×

---

### `src/intervention_recommender.py`

| Symbol | Kind | Signature | Notes |
|---|---|---|---|
| `get_available_interventions` | function | `(time_available_min=10, setting="break_room", dominant_dimension=None, burnout_score=None) -> list[dict]` | **Always returns exactly 5 dicts** (one per category) |
| `Intervention` | dataclass | `.id`, `.category`, `.name`, `.tagline`, `.description`, `.why_it_works`, `.duration_min`, `.needs_audio`, `.needs_writing`, `.best_for`, `.settings`, `.research_citation` | |
| `InterventionFeedback` | dataclass | `.intervention_id`, `.helpful`, `.checkin_id`, `.timestamp` | Post-intervention thumbs up/down |
| `INTERVENTIONS` | list | 10 `Intervention` instances (2 variants × 5 categories) | |
| `CATEGORIES` | dict | 5 category metadata dicts | |

**Each result dict keys:** `category_id`, `category_name`, `category_icon`, `category_color`, `category_desc`, `intervention` (nested dict), `available`, `is_relevant`, `relevance_reason`, `unavailable_reason`

**5 categories:** `body_sketch`, `shift_debrief`, `sound_reset`, `three_thing`, `quick_challenge`

---

### `src/labeler.py`

| Symbol | Kind | Signature | Returns |
|---|---|---|---|
| `compute_textblob_features` | function | `(text: str) -> dict` | `{polarity, subjectivity, sentiment_label, confidence}` |
| `count_keyword_themes` | function | `(text: str) -> dict` | `{burnout_keywords, physical_keywords, management_keywords, workload_keywords, environment_keywords}` |
| `label_checkin_debrief` | function | `(free_text: str) -> dict` | `{has_debrief, sentiment, domain_signals, word_count}` |
| `label_dataframe` | function | `(df, text_column="text") -> pd.DataFrame` | Adds sentiment + keyword cols to DataFrame |
| `compute_trend_features` | function | `(score_history, worker_type="general") -> dict` | `{trend, consecutive_high_days, alert_flag, peak_season_context, window_avg, window_days}` |

---

### `src/merger.py`

| Symbol | Kind | Signature | Notes |
|---|---|---|---|
| `merge_sources` | function | `(dataframes: dict, deduplicate=True) -> pd.DataFrame` | Keys must be registered source names |
| `assert_no_youtube` | function | `(df, stage="merge") -> None` | Raises `ValueError` if `source` col contains values outside `ALLOWED_SOURCES` |
| `register_source` | function | `(name, normalizer)` | Registers a new normalizer at runtime |
| `ALLOWED_SOURCES` | constant | `{"reddit", "glassdoor"}` | Enforced guardrail |

---

### `src/text_cleaner.py`

| Symbol | Kind | Signature | Notes |
|---|---|---|---|
| `clean_text` | function | `(text: str, source: str = "unknown") -> str` | Source-aware; returns `""` for empty/too-short |
| `clean_dataframe` | function | `(df, text_column="text", source_column="source", min_words=10) -> pd.DataFrame` | Drops short/empty/duplicate rows |

---

### `src/sentiment_model.py`

| Symbol | Kind | Notes |
|---|---|---|
| `WorkforceSentimentModel` | class | Requires `sentence_transformers` (heavy dep) |
| `__init__(model_name)` | method | Calls `SentenceTransformer(model_name)` immediately — downloads model on first run |
| `extract_features(text)` | method | Returns `{embedding, polarity, subjectivity, review_length, *theme_counts}` |
| `prepare_training_data(df)` | method | Converts labeled DataFrame → `(X, y)` for sklearn |

> Tests for this module are **guarded with `@unittest.skipUnless`**. They check
> importability and the presence of key methods only — no instantiation, no
> model download.

---

## Test Coverage

### `src/tests/test_backend_smoke.py`

| Test class | Module tested | What it checks |
|---|---|---|
| `TestTextCleaner` | `text_cleaner` | Returns string; empty/NaN input; URL/email stripping; Reddit markdown; short text filtering; Unicode normalization |
| `TestLabeler` | `labeler` | TextBlob feature keys; positive/negative/empty text; keyword theme counts; debrief analysis empty and non-empty |
| `TestCheckinEngine` | `checkin_engine` | Default valid; high-burnout valid; out-of-range errors; invalid enum errors; dimension properties in 0–10; BAT-12 scoring integration; `to_dict()` shape |
| `TestBurnoutScalesBat12` | `burnout_scales` | Returns `Bat12Score`; all subscales present; `normalized_0_10` in `[0,10]`; high > low; item count = 12; partial responses; `to_dict()` shape |
| `TestBurnoutScalesOlbi` | `burnout_scales` | Returns `OlbiScore`; subscales present; `normalized_0_10` in `[0,10]`; item count = 16; `to_dict()` shape |
| `TestCompoundScorer` | `compound_scorer` | Returns `BurnoutScore`; score in `[0,100]`; high > low; risk level valid; dominant dimension valid; dimension scores in `[0,100]`; multiplier ≥ 1; trend with/without history; `to_dict()` keys |
| `TestInterventionRecommender` | `intervention_recommender` | Always 5 results; required keys on each result and nested intervention; relevant first; unavailable marking; expected category IDs |
| `TestMerger` | `merger` | `assert_no_youtube` raises on `"youtube"` source; passes on `reddit`/`glassdoor`; no source column is safe; `ALLOWED_SOURCES` values; reddit-only merge; reddit+glassdoor merge; empty df skipped |
| `TestSentimentModelOptional` | `sentiment_model` | Class importable; `extract_features` callable; `prepare_training_data` callable — **skipped if `sentence_transformers` unavailable** |

### `src/tests/test_backend_integration.py`

| Test class | Scenario | Flow |
|---|---|---|
| `TestBackendIntegrationHighBurnout` | High-burnout worker | `clean_text` → `compute_textblob_features` → `CheckInResponse.validate()` → `compute_scale_scores()` (BAT-12 + OLBI) → `compute_burnout_score()` → `get_available_interventions()` |
| `TestBackendIntegrationLowBurnout` | Low-burnout worker | Same flow; asserts `risk_level` in `{low, moderate}` |
| `TestBackendIntegrationWithHistory` | Trend detection | Builds 7-item history; asserts `trend == "worsening"` when today spikes; asserts `trend == "stable"` when consistent |

---

## Running the tests

```bash
# Full suite (from project root, using venv)
.venv/Scripts/python -m unittest discover -s src/tests -p "test_*.py" -v

# Smoke tests only
.venv/Scripts/python -m unittest src.tests.test_backend_smoke -v

# Integration tests only
.venv/Scripts/python -m unittest src.tests.test_backend_integration -v
```