# WorkPulse AI — No-Rework Productization Path

## Summary

The current Streamlit MVP reads artifacts locally and runs all logic in-process.
This document describes how the architecture evolves from MVP to a real web or
mobile app without rewriting the core logic — because it is already structured
as a backend with a thin front-end on top.

---

## Three-stage path

### Stage 1 — MVP (now)

**What it is:** Streamlit reads cluster artifacts and config files directly from disk.
All business logic lives in `src/`. The app is a single-process tool for research and prototyping.

```
streamlit_app.py
  └── src/checkin_engine.py
  └── src/compound_scorer.py
  └── src/intervention_recommender.py
  └── src/cluster_router.py              ← runtime text → cluster → intervention
  └── src/analysis/personas.py           ← loads config/
  └── config/tracks.yaml
  └── config/personas.yaml
  └── data/processed/cluster_report.csv
  └── data/processed/cluster_representatives.csv
```

**Strengths:** zero infrastructure, instant iteration, shareable via `streamlit run`.
**Limits:** no user accounts, no persistent storage, no mobile client.

---

### Stage 2 — Streamlit reads from a backend API (next)

**What it is:** Extract the `src/` functions into FastAPI endpoints.
Streamlit becomes a client that calls the API over HTTP.
All product config (`tracks.yaml`, `personas.yaml`) stays as-is — the API reads it.

**Proposed endpoints (no implementation yet):**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/checkin/score` | Takes CheckInResponse JSON, returns burnout score |
| `POST` | `/route` | Takes free text + constraints, returns cluster match + interventions |
| `GET`  | `/tracks` | Returns all tracks from tracks.yaml |
| `GET`  | `/tracks/{track_id}` | Returns a single track with its interventions |
| `GET`  | `/clusters/{cluster_id}/tracks` | Returns track IDs for a cluster |
| `POST` | `/progress` | Stores a before/after completion event (user-scoped) |
| `GET`  | `/progress/{user_id}` | Returns session/user progress history |

**Transition steps:**
1. Wrap existing `src/` functions inside FastAPI route handlers — zero logic rewrite.
2. Replace `src.*` imports in `streamlit_app.py` with `httpx` calls to the API.
3. Deploy the API (Railway, Render, or AWS Lambda) and point Streamlit at it.

**Why this requires no rewrite:**
- `config/tracks.yaml` is already the product config — the API just serves it as JSON.
- `src/checkin_engine.py`, `src/compound_scorer.py` are already pure functions — wrap, don't rewrite.
- Session state in Streamlit maps 1:1 to a simple user-scoped database record.
- `src/cluster_router.route_user_input()` wraps cleanly as the `/route` endpoint.

---

### Stage 3 — React / mobile app uses the same API (later)

**What it is:** Replace the Streamlit front-end with a React web app or React Native
mobile app. The FastAPI backend from Stage 2 is unchanged.

```
React / React Native app
  └── calls FastAPI endpoints (same as Stage 2)
       └── wraps src/ functions (unchanged)
            └── reads config/tracks.yaml (unchanged)
```

**Why this requires no backend rework:**
- The API contract from Stage 2 is stable.
- `tracks.yaml` remains the single source of truth for product content.
  Updating interventions, adding tracks, or adjusting personas requires
  only a YAML edit — no code deployment.

---

### Stage 4 — Voice input (future workstream)

**What it is:** A speech-to-text layer that allows workers to describe their state
verbally. The transcribed text is passed through the existing cluster router
unchanged — no routing logic modifications needed.

```
Worker voice input (phone mic)
  └── Whisper API or browser Web Speech API → transcribed text
       └── src/cluster_router.route_user_input(transcribed_text, ...)
            └── same cluster → track → intervention routing as text input
```

**Why it matters for this user group:**
Frontline workers — warehouse pickers, logistics drivers, line workers — often cannot
type during or immediately after a shift. They may be wearing gloves, standing, physically
fatigued, or in a noisy environment. A voice interface reduces the interaction cost from
"type a paragraph" to "say 10 words."

**What this workstream would involve:**
- Integrate a speech-to-text API (Whisper, or browser `webkitSpeechRecognition`)
- Pass the transcript directly to `route_user_input()` — no router changes
- Design a voice-first UX: minimal screen interaction, audio readback for activity instructions
- Address privacy requirements: is audio stored? Who has access? What is the retention policy?
- Test for accent diversity, noisy environments, and low-signal settings

**Realistic constraints:**
- Adds a hard dependency on a third-party API (latency, cost, availability)
- Privacy policy is non-trivial if audio is processed server-side
- Out of scope for the competition MVP — flag as Stage 4 in all presentations
- Highest ROI for the highest-constraint users (physical workers who can't type)

**Lowest-friction starting point:**
Use the browser's built-in `webkitSpeechRecognition` API via a Streamlit custom component.
No server-side audio processing, no data retention — transcript is processed locally and
discarded after routing. Proof-of-concept can be built without a backend change.

---

## Business use cases

### Worker-facing

WorkPulse AI reduces the friction between "I feel terrible" and "I did something about it"
to under 60 seconds. The intervention is specific (not generic), constraint-aware (not
aspirational), and evidence-informed (not guesswork). Workers get something they can
actually do in a 5-minute break — not a journal prompt or a referral.

**Value proposition:** actionable recovery support that fits real shift constraints.

### Employer / workforce management

High burnout correlates directly with turnover, absenteeism, and safety incidents.
Warehouses, logistics companies, and large healthcare networks lose billions annually to
preventable attrition. A lightweight digital tool that measurably reduces acute strain
during shifts is a compelling EAP (Employee Assistance Program) addition — especially as
workforce mental health becomes a compliance and liability area.

**Value proposition:** retention, safety, and EAP differentiation at low cost per seat.

**Who would pay for this:**
- Large employers with high frontline headcount (Amazon, UPS, hospital systems)
- Occupational health providers building a digital care layer
- Union-negotiated wellness benefit providers

### API / platform (B2B)

The routing engine and intervention library are already structured as backend functions.
A future platform layer (see Stage 2) would allow HR systems, workforce management tools,
and occupational health providers to integrate WorkPulse AI as a service — not a standalone app.

**Value proposition:** B2B SaaS, API licensing, white-label deployment for enterprise EAP vendors.

**Defensible moat:**
The cluster archetypes are trained on real, sector-specific public discourse — not synthetic data.
That corpus is not easily replicated. Each new sector's data (healthcare, construction, agriculture)
adds a new training layer that improves routing accuracy for that sector specifically.

---

## Engagement reality note

Digital mental health tools consistently show low long-term engagement (Linardon, 2020;
Torous et al., 2021). Usage counts are poor proxies for impact. Progress tracking in
WorkPulse AI is therefore **outcome-based**: before/after ratings on Energy, Tension, and
Clarity per activity — not streaks, push reminders, or gamification badges.

Design principles that follow from this:
- **One activity = one data point.** Don't require journeys; reward individual completions.
- **Session-scoped progress.** Don't guilt users who don't return. Each session is self-contained.
- **No "days missed" counter.** Absence should be invisible, not penalised.
- **Qualitative signal over quantity.** A single Tension rating drop from 4→2 is more
  meaningful than 10 activities completed with no rating change.

---

## Config as product truth

`config/tracks.yaml` is designed to be the canonical product specification:
- Clinical or content reviewers can iterate on interventions without touching code.
- A/B testing different interventions maps to swapping YAML entries under a feature flag.
- Localisation (translated interventions for different markets) is a parallel YAML file.

The architecture treats content as data — not hardcoded strings — which means the product
can evolve without engineering involvement for content-only changes.

---

## File reference

| File | Role |
|------|------|
| `config/tracks.yaml` | Track and intervention definitions (product config) |
| `config/personas.yaml` | Persona definitions with cluster anchors |
| `src/analysis/personas.py` | Python loader for tracks and personas |
| `src/cluster_router.py` | Runtime text → cluster → intervention routing |
| `src/checkin_engine.py` | CheckInResponse schema and validation |
| `src/compound_scorer.py` | Burnout scoring logic |
| `src/intervention_recommender.py` | Constraint-aware intervention filtering |
| `streamlit_app.py` | Front-end (will become API client in Stage 2) |
| `data/processed/cluster_report.csv` | Cluster metadata |
| `data/processed/cluster_representatives.csv` | Representative texts per cluster |
