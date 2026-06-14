# ShiftGuard: Burnout Detection and Neuroarts Recovery for Blue-Collar Workers

## Executive Summary

ShiftGuard is an open-source app designed to detect compound burnout in blue-collar workers, including warehouse, shipping, manufacturing, and construction employees, and to recommend short, evidence-based creative recovery activities matched to the worker's current state, available time, and setting. The app combines a 60-second daily check-in with a neuroarts-based intervention engine, offering a practical alternative to the clinical wellness tools that most workers ignore or distrust.

## Project Rationale

### Why blue-collar workers

Most digital wellness tools are designed for office environments. They assume reliable internet access, unstructured time, private workspaces, and comfort with therapeutic language. Blue-collar workers face a fundamentally different burnout profile: physically demanding tasks, chronic pain, rotating or overnight shifts, financial instability, and strong cultural stigma around seeking help. These stressors compound in ways that standard single-dimension burnout scales were not built to capture. A worker finishing a 12-hour warehouse shift during peak season is unlikely to open an app that asks them to "journal about their emotions" or "schedule a therapy session." ShiftGuard is built for the reality of break rooms, commutes, and exhaustion.

### Why stigma-free language and short interventions matter

In manual labor settings, anything that sounds clinical or therapeutic risks being dismissed or mocked. Workers who might benefit most from support are often the ones most resistant to language that frames them as patients. ShiftGuard avoids terms like "mental health assessment," "therapy," or "mindfulness exercise." Instead, it uses plain, practical framing: "How physically worn down do you feel right now?" and "Try this 3-minute reset." Interventions are designed to be completable in 5 to 15 minutes during a break, on a commute, or at home after a shift.

### Why detection plus intervention

Detection alone creates a dead end. An app that tells a worker "you are burned out" without offering anything actionable adds awareness without agency. Worse, it can increase anxiety by naming a problem the worker has no clear path to address. ShiftGuard pairs every check-in result with a matched recovery recommendation, closing the loop between recognizing a problem and doing something about it. The intervention recommender never prescribes a single activity. It presents all five intervention categories and lets the worker choose what resonates, because autonomy is the antidote to the lack of control that drives burnout.

## Research Foundation

### Compound burnout in manual labor

Burnout in blue-collar populations is not a single-axis phenomenon. It emerges from the interaction of physical strain (heavy lifting, repetitive motion, standing for long periods), poor or irregular sleep, emotional exhaustion, financial pressure, and limited recovery time. These factors reinforce each other: chronic pain disrupts sleep, poor sleep lowers emotional resilience, and emotional exhaustion makes physical work feel harder. The compound scorer models this explicitly. A worker with moderate pain and moderate financial stress is at higher risk than someone with severe pain alone, because the anxiety about affording treatment amplifies the pain's impact on mood and sleep. This interaction effect is the core insight no existing burnout tool captures.

### Sleep as a bridge between physical and emotional strain

Sleep quality acts as a mediating variable between physical workload and emotional burnout. Workers in physically demanding roles who also experience disrupted sleep, whether from shift rotation, pain, or stress, show significantly higher rates of emotional exhaustion and cognitive fatigue. This is why ShiftGuard's daily check-in includes sleep quality as one of its six core questions rather than treating it as a secondary indicator. In the compound scorer, poor sleep amplifies the life-impact dimension, modeling the way sleep disruption bridges work strain and home-life deterioration.

### BAT-12 and OLBI as separate validated instruments

The Burnout Assessment Tool (BAT-12) and the Oldenburg Burnout Inventory (OLBI) are both validated measures of burnout, but they differ in structure, subscale composition, and theoretical orientation. BAT-12 captures four subscales: exhaustion, mental distance, cognitive impairment, and emotional impairment (12 items scored 1 to 5). OLBI measures two subscales: exhaustion and disengagement (16 items scored 1 to 4, with 8 reverse-scored items). Blending them into a single composite scale would compromise the psychometric integrity of both. ShiftGuard implements each as a standalone validation module in `burnout_scales.py`, preserving the original item groupings, reverse-scoring rules, and scoring procedures. Neither instrument is used as the daily check-in itself. Instead, they serve as periodic calibration and validation tools, triggered by configurable cadence rules: BAT-12 is prompted after 3 consecutive high-scoring days or when the compound score exceeds 65, while OLBI is prompted less frequently (minimum 14 days between administrations) given its longer format.

### Neuroarts as a practical recovery layer

The emerging field of neuroarts, drawing on research synthesized in works such as "Your Brain on Art" (Magsamen & Ross, 2023) and the NeuroArts Blueprint, demonstrates that brief creative activities can reduce cortisol, improve mood regulation, and support cognitive recovery. Importantly, these activities do not require artistic skill or prior experience. A 3-minute body sketch, a short sound-based reset, or a brief expressive writing prompt can produce measurable physiological and psychological shifts. For blue-collar workers, these modalities have a key advantage over traditional mindfulness or meditation: they feel like doing something, not like sitting still.

### Creative micro-interventions for short breaks

The evidence base for brief creative interventions, typically 5 to 15 minutes, shows that even very short engagement with drawing, writing, listening, or noticing tasks can interrupt stress cycles and support recovery. Each intervention in the ShiftGuard catalog is backed by published research: interoception studies (Mehling et al., 2012) for body sketching, the Pennebaker expressive writing program for shift debriefs, music-and-cortisol research (Thoma et al., 2013) for sound resets, psychological detachment literature (Sonnentag & Fritz, 2007) for the 3-thing notice, and art-making studies (Kaimal et al., 2016) for quick challenges.

## Architecture Changes

ShiftGuard is structured as a three-part system: detection, scoring and validation, and intervention recommendation. Alongside these sits a data collection and training pipeline that supplies the sentiment model.

### Detection

The detection layer is the daily check-in, a 60-second interaction consisting of six core questions on 0-to-10 slider scales (or a three-option choice for the safety question), plus two intervention preference questions, two context questions, and an optional free-text field. The check-in engine (`checkin_engine.py`) validates inputs, stores responses, and timestamps the shift context. The six core questions are: physical worn-down, pain or soreness, mentally drained, sleep quality (inverted, higher is better), work stress, and shift safety energy. These were selected based on occupational stress literature as the strongest predictors of compound burnout.

The engine also supports a weekly extended question set, triggered when daily scores stay elevated, covering enjoyment, irritability, motivation, recovery, financial stress, and schedule impact. This tiered cadence, daily core plus weekly extended, keeps the baseline interaction fast while surfacing deeper signals when needed.

### Scoring and validation

The compound scorer (`compound_scorer.py`) takes the daily check-in responses and produces a single 0-to-100 burnout score. The check-in's six questions are mapped to three dimensions: body (physical worn-down, pain/soreness, inverted sleep), mind (mentally drained, work stress, safety energy), and life (derived from the interaction of body and mind strain, amplified by poor sleep). The scorer then applies a five-step algorithm. First, each dimension is normalized to 0-to-100. Second, a weighted base score is computed (body 40%, mind 35%, life 25%, body weighted higher because physical injury risk is immediate). Third, a compound interaction multiplier is applied: when two dimensions both exceed 55, the score is amplified (body+mind: 1.15x, body+life: 1.12x, mind+life: 1.10x), and if all three are elevated, a 1.25x multiplier is applied. Fourth, the dominant dimension is identified. Fifth, trend detection compares the current score against a 7-day rolling average, flagging worsening (score more than 8 points above average), improving (more than 8 below), or stable patterns.

The score is presented to the worker as a simple gauge using green (0 to 30), amber (30 to 55), red (55 to 75), and critical (75 to 100) zones, with language like "Your body budget is low" rather than clinical labels.

If the worker provides free text, the sentiment model (`sentiment_model.py`) analyzes it using SBERT embeddings and a Gradient Boosting classifier. The model also provides an `analyze_debrief` function that maps detected keyword themes to BAT-12/OLBI subscale domain names, producing domain signals structurally compatible with the burnout scale output so they can feed directly into the intervention recommender's targeting logic.

BAT-12 and OLBI are implemented as separate validation modules in `burnout_scales.py`. They are not part of the daily check-in flow. Each module normalizes its subscale means to a 0-to-10 scale within its own interface, and raw item responses and raw subscale means are stored for psychometric auditability. These modules exist for research calibration and periodic self-assessment, not for daily use.

### Intervention recommendation

The intervention recommender (`intervention_recommender.py`) always presents all five intervention categories to the worker. It does not prescribe a single activity. Within each category, it selects the best-fit variant based on the worker's available time (5, 10, or 15 minutes) and current setting (break room, commute, home). Categories that target the worker's dominant burnout dimension are sorted first, and when BAT-12 or OLBI subscale scores are available, categories mapped to elevated subscales are additionally boosted. For example, elevated exhaustion boosts body sketch and sound reset, while elevated disengagement boosts quick challenge and 3-thing notice. Every category is always shown, even if no variant perfectly fits the current constraints, in which case the category is marked as needing more time or a different setting.

After completing an intervention, the worker gives thumbs-up or thumbs-down feedback via an `InterventionFeedback` record linked back to the originating check-in.

### Data collection and training pipeline

The project includes a full data pipeline (`run_pipeline.py`) for collecting, cleaning, labeling, and merging external worker sentiment data to train and calibrate the sentiment model. The pipeline runs in five stages.

**Collection.** Three collectors gather raw data. The YouTube transcript collector (`youtube_transcript_collector.py`) fetches auto-captions from warehouse and fulfillment worker experience videos, segmenting them into 2-minute review-length chunks. The Glassdoor loader (`glassdoor_loader.py`) normalizes CSV exports from multiple third-party scraping formats into a unified schema, detecting warehouse roles via job-title heuristics. The Reddit collector (`reddit_collector.py`) uses PRAW in read-only mode to pull posts from targeted subreddits (warehouse, shipping, manufacturing, and general labor communities), with keyword and flair filtering for complaint posts, built-in rate-limit backoff, anonymization of usernames and URLs, and peak-period flagging for holiday and Prime Day surges.

**Cleaning.** The text cleaner (`text_cleaner.py`) normalizes Unicode, strips URLs and emails, removes platform-specific formatting (Reddit markdown, YouTube auto-caption artifacts), collapses whitespace, and drops texts shorter than 10 words.

**Merging.** The merger (`merger.py`) combines all three sources into a unified 18-column schema (`UNIFIED_SCHEMA` in `settings.py`), deduplicating across sources by text content.

**Labeling.** The labeler (`labeler.py`) applies TextBlob sentiment analysis (positive/neutral/negative at configurable polarity thresholds), counts keyword theme matches across five domains (burnout, physical demands, management issues, workload, environment), detects warehouse versus corporate roles via job-title and text heuristics, and computes word counts. For check-in debrief entries, a separate `label_checkin_debrief` function maps hits against `CHECKIN_BURNOUT_THEMES`, which are keyed to BAT-12 and OLBI subscale names (exhaustion, mental distance, cognitive impairment, emotional impairment, disengagement) so the output plugs directly into the intervention recommender.

**Trend features.** The labeler also includes `compute_trend_features`, which calculates daily trend direction against a rolling 7-day window, consecutive high-day streaks with alert flags (triggered after 3 or more consecutive days above 60), and seasonal context awareness using configurable peak-season months by sector (warehouse: October through January, retail: November through January).

**Export.** The pipeline outputs a timestamped, train-ready CSV to `data/processed/`.

### Supporting infrastructure

All pipeline parameters, API credentials, keyword lists, subreddit targets, scoring thresholds, scale cadence rules, and peak-season definitions are centralized in a single settings file (`settings.py`) with environment variable support via `.env`. Project dependencies are tracked in `requirements.txt`, covering sentence-transformers, scikit-learn, TextBlob, PRAW, the YouTube transcript API, FastAPI (planned), and supporting utilities.

The current MVP runs in a single Streamlit interface (`streamlit_app.py`) with all five user-facing steps visible on one screen. Future phases include a React Native frontend and a FastAPI backend.

## Scoring Changes

### Daily check-in dimensions

The worker check-in captures six core items mapped to three compound dimensions. Body strain averages physical worn-down, pain/soreness, and inverted sleep quality. Mind strain averages mentally drained, work stress, and a numeric conversion of the shift-safety-energy flag (yes = 0, somewhat = 4, no = 8). Life impact is derived from the interaction of body and mind strain, with a 1.2x amplifier applied when sleep strain reaches 6 or higher. This three-dimension model, body, mind, and life, feeds the compound scorer's 0-to-100 algorithm.

Weekly extended questions (enjoyment, irritability, motivation, recovery, financial stress, schedule impact) are triggered when daily scores stay elevated, providing deeper signal without burdening the daily flow.

### Separation from BAT-12 and OLBI

The daily compound score is intentionally separate from BAT-12 and OLBI. The compound score is a lightweight, practical signal designed for daily use. BAT-12 and OLBI are research-grade instruments designed for periodic assessment. Blending them would compromise the validity of the validated scales and would also make the daily check-in too long. Workers complete the compound check-in every day. They complete BAT-12 or OLBI only when prompted by cadence rules or when they choose to self-assess.

### Normalization

Within their own modules, BAT-12 subscale means are normalized from the 1-to-5 response range to 0-to-10, and OLBI subscale means are normalized from the 1-to-4 range to 0-to-10. This normalization is for display and comparison purposes within each module. It does not merge the two instruments. The daily compound score uses its own independent 0-to-100 scale.

### Raw data storage

Raw item responses and raw subscale means are stored for every check-in and every validation module completion. The `ScaleSubscore` dataclass in `burnout_scales.py` preserves both `raw_mean` and `normalized_0_10` alongside the item count, ensuring that any future psychometric analysis, whether internal or by external researchers, can work from the original data rather than relying on normalized or aggregated values.

### Reliability as a separate layer

Reliability (internal consistency, test-retest stability) is a property of the data collected over time. It is computed as a separate analytic layer, not embedded in the scoring output. A normalized score of 7.2 is not "reliable" or "unreliable" on its own. Reliability is assessed by analyzing patterns across many responses, and it is reported separately from individual scores.

## Neuroarts Intervention Layer

### Why creative micro-interventions

ShiftGuard deliberately avoids generic meditation or relaxation language. Terms like "mindfulness exercise" or "guided breathing" carry connotations that many blue-collar workers find off-putting, whether because of cultural stigma, skepticism, or simple unfamiliarity. Creative micro-interventions offer the same physiological and psychological benefits, reduced cortisol, improved mood regulation, cognitive reset, but framed as practical activities rather than therapeutic exercises. The recommender's design philosophy is explicit: never prescribe, always offer choices.

### The five intervention categories

The current intervention catalog includes five categories, each with 5-minute and 10-minute variants (and in some cases 15-minute variants). Each variant specifies which settings it works in, whether it requires audio or writing, and which burnout dimensions it targets.

**Body sketch** (interoception/body awareness drawing). The worker draws a stick figure and marks where they feel tension, pain, or tightness. The 5-minute version is observation only ("just notice and mark"). The 10-minute version adds targeted stretching for each marked spot. Targets the body dimension. Works in all settings (10-minute version excludes commute). Based on Mehling et al., 2012, and Magsamen & Ross, 2023.

**Shift debrief** (expressive writing, Pennebaker method). The worker writes freely about their shift for 3 to 5 minutes. The 10-minute version adds a reframe step: re-read, underline one thing you handled well, write one sentence about it. Targets mind and life dimensions. Works in all settings. Based on Pennebaker & Beall, 1986, and Seligman et al., 2005.

**Sound reset** (tempo-matched audio for stress reduction). The worker listens to a 60 BPM track to synchronize with resting heart rate and activate the parasympathetic nervous system. The 10-minute version adds paced breathing (inhale 4 beats, exhale 6 beats). Targets mind and body dimensions. Works on commute and at home (requires audio). Based on Thoma et al., 2013, and Zaccaro et al., 2018.

**3-thing notice** (psychological detachment). The worker names three things they noticed today that have nothing to do with work. The 10-minute version adds a sensory expansion: for each thing, describe one detail about how it looked, sounded, or felt. Targets mind and life dimensions. Works in all settings. Based on Sonnentag & Fritz, 2007.

**Quick challenge** (gamified micro-creative task). A 30-second creative sprint such as "Draw the weirdest tool you used today" or "Design a logo for your crew in 60 seconds." The 10-minute version adds a creative build step. Targets mind and body dimensions. Works in break room and at home. Based on Kaimal et al., 2016, and Magsamen & Ross, 2023.

### Accessibility and stigma

All interventions are designed to be completable in 5 to 15 minutes, to require no special equipment beyond a phone, and to be usable in a break room, on a bus, or at home. None of them use therapeutic language. None of them require the worker to identify as stressed, burned out, or in need of help. They are framed as resets, challenges, or quick activities.

### Matching logic

The intervention recommender considers four factors: the worker's dominant burnout dimension from the compound score, time available (5, 10, or 15 minutes), current setting, and, when available, elevated BAT-12/OLBI subscales or debrief-derived domain signals. A subscale-to-category mapping routes elevated exhaustion toward body sketch and sound reset, elevated mental distance toward 3-thing notice and quick challenge, elevated cognitive impairment toward sound reset and shift debrief, elevated emotional impairment toward shift debrief and 3-thing notice, and elevated disengagement toward quick challenge and 3-thing notice. The worker always sees all five categories, sorted by relevance, and chooses for themselves.

## Design Principles

The following principles guide product decisions across the project.

**Brevity.** The daily check-in takes no more than 60 seconds (six core sliders plus two preference taps). Interventions are completable in 5 to 15 minutes. Nothing in the app should feel like homework.

**Autonomy.** The worker decides when to check in, whether to write free text, whether to complete a validation module, and which intervention to try. The recommender never prescribes. It always presents all five categories and lets the worker choose.

**Privacy.** Data stays on the worker's device or in their own account. No data is shared with employers. No identifying information is required. The Reddit collector anonymizes usernames, URLs, emails, and phone numbers at collection time.

**Worker-friendly language.** Every label, prompt, and recommendation is written in plain, non-clinical language. Questions use terms like "worn down," "soreness," and "drained" rather than "burnout," "depression," or "anxiety." If a phrase sounds like it belongs in a therapist's office, it gets rewritten.

**Evidence-based but non-clinical framing.** Every intervention cites published research. Every check-in question is grounded in occupational stress literature. But the app never presents itself as a clinical tool or a substitute for professional care.

**Modular validation.** BAT-12 and OLBI are implemented as independent, optional modules with configurable cadence triggers. They can be enabled or disabled without affecting the daily check-in or the intervention layer.

**Separate detection and intervention layers.** Detection (check-in and scoring) and intervention (neuroarts recommendations) are architecturally distinct. Either layer can be developed, tested, or replaced independently.

## Open Questions and Decisions

Several product and research decisions remain unresolved.

Should BAT-12 or OLBI serve as the primary validation module, or should both remain available with equal weight? BAT-12 offers a broader four-subscale structure and more granular intervention targeting. OLBI is shorter in response range (1 to 4 versus 1 to 5), though it has 16 items versus BAT-12's 12, and maps cleanly to a two-factor exhaustion/disengagement model.

Should both instruments be enabled simultaneously in a research mode, allowing researchers to compare scores across instruments for the same population?

What compound score thresholds should trigger which intervention types? The current matching logic uses dimension dominance and subscale elevation, but the specific elevation threshold (currently 6.0 on the normalized 0-to-10 scale) and the compound interaction threshold (currently 55 on the 0-to-100 scale) have not been empirically validated against worker outcomes.

Should the neuroarts layer be enabled by default for all users, or should it be an opt-in feature that workers discover after using the check-in for some time?

How should reliability be tracked over time? Options include periodic Cronbach's alpha calculations on accumulated check-in data, rolling test-retest windows, or flagging individual workers whose response patterns suggest disengagement (e.g., identical responses every day).

What is the right cadence for periodic scale administration? The current defaults (BAT-12 after 3 consecutive high days or score above 65, OLBI no more than once every 14 days) are informed guesses, not validated thresholds.

## Implementation Notes

### Core application modules

The codebase includes four core modules that form the main user-facing loop. The check-in engine (`checkin_engine.py`) defines the `CheckInResponse` dataclass with six core questions, two intervention preference fields, context fields, optional free text, and optional BAT-12/OLBI response dictionaries. It exposes computed properties for the three burnout dimensions (body, mind, life), each on a 0-to-10 scale, and a `validate()` method that enforces range and option constraints. It also defines the full question catalog as structured dictionaries for frontend rendering, including daily core, intervention preference, context, and weekly extended question sets.

The burnout scales module (`burnout_scales.py`) implements `score_bat12` and `score_olbi` as independent scoring functions. Each returns a typed result (`Bat12Score` or `OlbiScore`) containing `ScaleSubscore` objects with raw means, normalized 0-to-10 values, and item counts per subscale.

The compound scorer (`compound_scorer.py`) implements `compute_burnout_score`, which takes a check-in response and optional score history, and returns a `BurnoutScore` dataclass with the composite 0-to-100 score, risk level, per-dimension breakdown, compound multiplier details, interaction pair labels, trend direction, and consecutive high-day count.

The intervention recommender (`intervention_recommender.py`) implements `get_available_interventions`, which returns all five categories with their best-fit variants, availability flags, relevance indicators (dimension match or subscale boost), and unavailability reasons. It also defines the full `Intervention` dataclass and the `INTERVENTIONS` catalog with all variants, plus `InterventionFeedback` for post-intervention tracking.

### Data pipeline modules

The data pipeline runs through five stages orchestrated by `run_pipeline.py`. The YouTube transcript collector discovers videos via seed queries (or uses a preconfigured video ID list) and segments auto-captions into review-length chunks. The Glassdoor loader handles multiple CSV export formats through a flexible column-mapping system and combines pros, cons, summary, and advice fields into unified review text. The Reddit collector pulls from 15+ targeted subreddits across warehouse, shipping, manufacturing, and general labor categories, with complaint keyword and flair filtering, anonymization, and peak-period flagging. The text cleaner normalizes all sources through a shared pipeline. The merger combines sources into a unified 18-column schema. The labeler applies TextBlob sentiment, keyword theme counts, warehouse detection, and word counts.

### Sentiment and analysis

The sentiment model (`sentiment_model.py`) implements a `WorkforceSentimentModel` class that combines SBERT embeddings (paraphrase-MiniLM-L6-v2) with a Gradient Boosting classifier for three-class sentiment prediction. The model includes an `analyze_debrief` method that extends prediction with domain-signal extraction, mapping keyword themes to BAT-12/OLBI subscale domains and amplifying signals when overall sentiment is negative (1.3x multiplier). Domain signals are output in a format structurally compatible with `ScaleSubscore` dicts, so they can be passed directly to the intervention recommender's `_boosted_categories_from_scales` function without requiring a full psychometric assessment.

The labeler's `compute_trend_features` function provides daily trend direction, consecutive high-day streaks with configurable alert thresholds, and seasonal context awareness using sector-specific peak-month calendars.

### Configuration

All parameters are centralized in `settings.py`: API credentials (Reddit, YouTube), subreddit targets and category mappings, keyword theme dictionaries (both for the data pipeline and for check-in debrief analysis), TextBlob polarity thresholds, pipeline defaults, check-in burnout themes keyed to BAT-12/OLBI subscale names, trend thresholds, scale cadence rules, peak-season month calendars, and the unified output schema.

### Frontend and deployment

The MVP frontend is a single Streamlit app (`streamlit_app.py`) that presents all five user-facing steps (check-in, score display, intervention recommendation, trend tracking, and optional shift debrief) on one screen. The planned production architecture separates the frontend (React Native) from the backend (FastAPI, already listed in `requirements.txt`), but this transition has not yet begun.
