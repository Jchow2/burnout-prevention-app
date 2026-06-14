# WorkPulse AI — Burnout Detection & Constraint-Aware Interventions

Stress archetype detection and evidence-based micro-interventions for frontline workers.
Combines NLP clustering of 20,000+ real workforce posts with a 60-second check-in and
a constraint-aware routing engine that matches workers to specific recovery activities
based on their available time and physical setting.

See [docs/productization_plan.md](docs/productization_plan.md) for the full productization path (Streamlit → FastAPI → mobile → voice).

---

## Repository layout

```
neuroart-intervention-app/
├── pipeline.py               # Canonical data pipeline entrypoint
├── config.yaml               # Pipeline configuration (subreddits, paths, filters)
├── requirements.txt
├── config/
│   └── settings.py           # All pipeline constants and env-var bindings
├── src/
│   ├── data_collection/      # Ingest, normalize, filter, load Kaggle CSVs
│   ├── analysis/             # Embedding-based clustering (next modeling phase)
│   ├── scoring/
│   │   └── burnout_scales.py # BAT-12 and OLBI scoring modules
│   ├── checkin_engine.py     # Daily check-in dataclass + validation
│   ├── compound_scorer.py    # 0-100 compound burnout score
│   ├── intervention_recommender.py
│   ├── labeler.py            # TextBlob sentiment + keyword theme labeling
│   ├── merger.py             # Source merge + guardrail (reddit + glassdoor only)
│   ├── sentiment_model.py    # SBERT + GradientBoosting classifier
│   └── text_cleaner.py       # Shared text normalization
├── data/
│   ├── raw/                  # Kaggle source CSVs (not committed)
│   └── processed/            # Pipeline outputs (not committed)
├── docs/
│   ├── project_document.md
│   └── neurotrace_project_scope.docx
└── archive/
    ├── youtube/              # DEPRECATED — YouTube ingestion removed
    └── legacy/               # Superseded pipeline versions and stale duplicates
```

---

## Data sources

| Source | Format | Location |
|---|---|---|
| Reddit (antiwork + reddit_sentiment) | Kaggle CSV | `data/raw/antiwork/`, `data/raw/reddit_sentiment/` |
| Glassdoor (davidgauthier Kaggle dataset) | Kaggle CSV | `data/raw/glassdoor_kaggle/` |

**YouTube ingestion is permanently removed.** Valid sources are `reddit` and
`glassdoor` only. The merger enforces this with a runtime assertion.

---

## Streamlit demo app

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. Three tabs:

| Tab | What it shows |
|---|---|
| Check-In & Score | 60-second daily check-in → Body / Mind / Life scores + compound burnout score (0–100) |
| Interventions | All 5 neuroarts activity categories, ranked by check-in context |
| Cluster Explorer | Browse the 30 workforce narrative clusters with representative text snippets |

No external APIs. All data is read-only. The cluster tabs require the clustering pipeline to have been run first (`data/processed/cluster_report.csv` must exist).

---

## Running the pipeline

```bash
python pipeline.py --config config.yaml
```

Outputs written to `data/processed/`:
- `workforce_posts_clean.parquet`
- `workforce_comments_clean.parquet`
- `workforce_glassdoor_clean.parquet`
- `workforce_all_text.parquet` ← **canonical modeling input**

Dry-run (no files written):
```bash
python pipeline.py --config config.yaml --dry-run
```

---

## Running the tests

```bash
# Full suite (from project root)
.venv/Scripts/python -m unittest discover -s src/tests -p "test_*.py" -v

# Smoke tests only
.venv/Scripts/python -m unittest src.tests.test_backend_smoke -v

# Integration tests only
.venv/Scripts/python -m unittest src.tests.test_backend_integration -v
```

The smoke tests cover `text_cleaner`, `labeler`, `checkin_engine`, `burnout_scales`,
`compound_scorer`, `intervention_recommender`, `merger`, and `sentiment_model`
(last module skipped automatically if `sentence_transformers` is not installed).

Integration tests run the full pipeline: `clean_text` → `label_checkin_debrief` →
`CheckInResponse.validate()` → `compute_scale_scores()` → `compute_burnout_score()` →
`get_available_interventions()`.

See [docs/backend_test_plan.md](docs/backend_test_plan.md) for the full API reference and
coverage map.

---

## Embedding-based clustering

Produces cluster-annotated parquet + CSV reports from the canonical corpus.

```bash
python -m src.analysis.cluster_embeddings \
    --input data/processed/workforce_all_text.parquet \
    --output_dir data/processed \
    --sample_n 20000 \
    --mode kmeans \
    --cache_embeddings true
```

| Flag | Default | Notes |
|---|---|---|
| `--sample_n` | 20 000 | Stratified by source × type |
| `--force_full_run` | off | Process all 323 K rows |
| `--k_values` | 10 15 20 30 | k values evaluated; best chosen by silhouette |
| `--model` | `all-MiniLM-L6-v2` | Any sentence-transformers model name |
| `--cache_embeddings` | true | Saves `.npz` to `output_dir`; skips encode on reruns |

**Outputs** (all in `data/processed/`):

| File | Contents |
|---|---|
| `workforce_all_text_with_clusters.parquet` | Input rows + `cluster_id` + `cluster_label` |
| `cluster_report.csv` | Cluster size, TF-IDF top keywords, source/type breakdown |
| `cluster_representatives.csv` | 5 texts per cluster closest to centroid |
| `embeddings_<model>_<fingerprint>.npz` | Cached embeddings (not committed) |

---

## Data Sources

All datasets are sourced from Kaggle and used for **research and demo purposes only**.

| Dataset | Source | Content |
|---------|--------|---------|
| r/antiwork posts & comments | Kaggle (public Pushshift export) | Workforce frustration, job dissatisfaction, wage grievances |
| Reddit sentiment dataset | Kaggle (multi-subreddit) | Pre-labelled work-related posts and comments across warehouse / logistics subreddits |
| Glassdoor reviews | Kaggle (davidgauthier dataset) | Anonymised employer reviews with free-text pros/cons and star ratings |

**No personal data is stored in processed outputs.** Company names from Glassdoor are anonymised by the pipeline. Reddit usernames are pseudonymous by design.

Raw CSVs are not committed to this repository due to file size. See [`data/raw/README.md`](data/raw/README.md) for exact filenames, expected schemas, and download instructions.

---

## Pipeline Overview

The data pipeline transforms raw public CSVs into pre-computed cluster files used by the app.

```
Stage 1 — Ingestion
    data/raw/**/*.csv
        ↓  load_raw_data()   [pipeline.py]
    raw DataFrames (Reddit posts, comments; Glassdoor reviews)

Stage 2 — Filtering & Cleaning
        ↓  clean_text() + filter functions   [src/data_collection/]
    Subreddit allowlist · text quality gate · Unicode normalisation
    Reddit markdown stripped · Glassdoor boilerplate removed · dedup

Stage 3 — Sentiment Labeling
        ↓  add_sentiment()   [src/data_collection/sentiment.py]
    + sentiment_score (−1 to +1) · sentiment_label · sentiment_source

Stage 4 — Union & Export
        ↓  export_outputs()   [pipeline.py]
    data/processed/workforce_all_text.parquet   ← canonical modeling input

Stage 5 — Embedding
        ↓  generate_embeddings()   [pipeline.py → cluster_embeddings.py]
    sentence-transformers all-MiniLM-L6-v2 · 384-dim L2-normalised vectors
    Cached to data/processed/embeddings_*.npz

Stage 6 — Clustering
        ↓  cluster_text()   [pipeline.py → cluster_embeddings.py]
    KMeans · k selected by silhouette score · TF-IDF keywords per cluster
    data/processed/cluster_report.csv
    data/processed/cluster_representatives.csv   ← used at runtime
```

Run pipeline stages 1–4:
```bash
python pipeline.py --config config.yaml
```

Run stages 5–6 (requires sentence-transformers):
```bash
python -m src.analysis.cluster_embeddings \
    --input data/processed/workforce_all_text.parquet \
    --output_dir data/processed
```

Print a full pipeline overview in the terminal:
```bash
python -m src.data_pipeline_overview
```

---

## Runtime Flow

At app runtime, no retraining occurs. The pre-computed cluster representatives are used to match user input on the fly:

```
User free-text input
    │
    ▼  embed with all-MiniLM-L6-v2 (same model used for clustering)
    │
    ▼  cosine similarity vs. cluster_representatives.csv
    │
    ▼  cluster_id  (best matching narrative archetype)
    │
    ▼  cluster_to_track_ids()  — static YAML mapping (config/tracks.yaml)
    │
    ▼  get_available_interventions()  — filtered by time + setting
    │
    ▼  ranked neuroarts activity list
```

This is implemented in [`src/cluster_router.py`](src/cluster_router.py):

```python
from src.cluster_router import route_user_input

result = route_user_input(
    user_text="Exhausted after a double shift, can't switch off",
    time_available_min=5,
    setting="break_room",
)
print(result["explanation"])
# → "Input matched cluster 7 ('exhausted / shift / sleep', similarity=0.71).
#    Mapped to track(s): sound_reset, body_sketch."
```

The function works independently of any pre-selected persona — it routes
directly from text to cluster to intervention.

---

## Limitations

- **Static dataset**: the clustering model is trained on a fixed snapshot
  of Reddit and Glassdoor data. It does not update as new posts are published.

- **No real-time ingestion**: there is no live Reddit API or Glassdoor API
  integration. Adding new data requires re-running the full pipeline from Stage 1.

- **Approximate cluster matching**: cosine similarity against 5 representative
  texts per cluster is a heuristic. It works well for text similar to the
  training corpus but may return low-confidence matches for very unusual inputs.

- **English only**: all text processing (cleaning, tokenisation, TF-IDF, and
  the sentence-transformers model) assumes English-language input.

- **Research / demo use only**: the system is not a clinical tool and does
  not diagnose burnout. Intervention suggestions are evidence-informed but
  are not a substitute for occupational health support.