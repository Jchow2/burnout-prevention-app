"""
src/data_pipeline_overview.py

Documents how raw data becomes the processed files used by the app.
No ML logic lives here — this is a reference module for understanding
the full data lineage from source CSVs to runtime inference.

Run this file directly to print a structured overview:
    python -m src.data_pipeline_overview
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# DATA LINEAGE
# ─────────────────────────────────────────────────────────────────────────────
#
# STAGE 1 — INGESTION (pipeline.py → load_raw_data)
# ──────────────────────────────────────────────────
# Source files (not committed; obtained from Kaggle):
#
#   data/raw/antiwork/
#       the-antiwork-subreddit-dataset-posts.csv     ← reddit post metadata + text
#       the-antiwork-subreddit-dataset-comments.csv  ← reddit comment text
#
#   data/raw/antiwork/reddit_sentiment/
#       posts_df.csv         ← broader reddit sentiment dataset
#       comments.csv
#       user_posts.csv
#
#   data/raw/glassdoor_kaggle/
#       glassdoor_reviews.csv   ← workplace reviews (text + star ratings)
#
# Each loader (src/data_collection/ingest.py) reads a CSV in chunks,
# normalises column names to the shared schema (src/data_collection/normalize.py),
# and returns a DataFrame ready for filtering.
#
# ─────────────────────────────────────────────────────────────────────────────
#
# STAGE 2 — FILTERING & CLEANING (pipeline.py → _apply_reddit_filters, clean_text)
# ──────────────────────────────────────────────────────────────────────────────
# Applied filters:
#   • subreddit allowlist (warehouse, shipping, manufacturing, general)
#   • minimum text length (30 characters by default)
#   • minimum Reddit score (removes low-engagement posts)
#   • date range (configurable in config.yaml)
#   • deduplication on cleaned text field
#   • optional per-subreddit sampling cap (max_per_subreddit)
#
# Text cleaning (src/text_cleaner.py):
#   • Unicode normalisation  (smart quotes → ASCII, em-dash → hyphen)
#   • URL and email removal
#   • Reddit markdown stripping  (*bold*, ~~strikethrough~~, [links])
#   • Glassdoor boilerplate removal  (Pros:, Cons:, HTML entities)
#   • Whitespace collapse
#
# ─────────────────────────────────────────────────────────────────────────────
#
# STAGE 3 — SENTIMENT LABELING (pipeline.py → add_sentiment)
# ───────────────────────────────────────────────────────────
# Adds three columns to each row:
#   sentiment_score  : float −1 to +1  (VADER compound score by default)
#   sentiment_label  : "positive" | "neutral" | "negative"
#   sentiment_source : "vader" or "textblob"
#
# ─────────────────────────────────────────────────────────────────────────────
#
# STAGE 4 — UNION OUTPUT (pipeline.py → export_outputs)
# ──────────────────────────────────────────────────────
# Per-source files:
#   data/processed/workforce_posts_clean.parquet      Reddit posts
#   data/processed/workforce_comments_clean.parquet   Reddit comments
#   data/processed/workforce_glassdoor_clean.parquet  Glassdoor reviews
#
# Union (canonical modeling input):
#   data/processed/workforce_all_text.parquet
#       — all three sources combined, common columns only
#       — this is the input to the embedding + clustering pipeline
#
# Coverage report:
#   data/processed/subreddit_coverage_report.csv
#       — row counts and date ranges per (source, subreddit) combination
#
# ─────────────────────────────────────────────────────────────────────────────
#
# STAGE 5 — EMBEDDING (src/analysis/cluster_embeddings.py → generate_embeddings)
# ───────────────────────────────────────────────────────────────────────────────
# Model: all-MiniLM-L6-v2 (sentence-transformers)
#   • 384-dimensional dense vectors, L2-normalised
#   • Encodes all rows in workforce_all_text.parquet (stratified sample
#     of 20 000 rows by default; --force_full_run for all ~323 K)
#   • Cached to data/processed/embeddings_<model>_<fingerprint>.npz
#     so reruns skip re-encoding automatically
#
# ─────────────────────────────────────────────────────────────────────────────
#
# STAGE 6 — CLUSTERING (src/analysis/cluster_embeddings.py → cluster_text)
# ─────────────────────────────────────────────────────────────────────────
# Algorithm: KMeans with silhouette-score k selection
#   • Tries k ∈ {10, 15, 20, 30}  (configurable via --k_values)
#   • Selects k with highest average silhouette on a 5 000-row subsample
#   • Fits final KMeans on all embedded rows
#
# Outputs:
#   data/processed/workforce_all_text_with_clusters.parquet
#       — input rows + cluster_id + cluster_label columns
#
#   data/processed/cluster_report.csv
#       Columns: cluster_id, cluster_label, size, top_keywords, source_counts
#       One row per cluster.  cluster_label is the top-3 TF-IDF keywords joined
#       by " / " — a human-readable placeholder, not a curated name.
#
#   data/processed/cluster_representatives.csv
#       Columns: cluster_id, cluster_label, rank, id, source, type, text_snippet
#       5 rows per cluster — the 5 texts closest to each cluster centroid.
#       Used at runtime by src/cluster_router.py to match user input.
#
# ─────────────────────────────────────────────────────────────────────────────
#
# RUNTIME — INFERENCE (src/cluster_router.py)
# ────────────────────────────────────────────
# At app runtime (no re-training):
#
#   user_text  (free-text input, check-in debrief, or Persona Simulator)
#       │
#       ▼  embed with same all-MiniLM-L6-v2 model
#       │
#       ▼  cosine similarity vs. cluster_representatives.csv
#       │
#       ▼  cluster_id  (best matching cluster)
#       │
#       ▼  cluster_to_track_ids()  (config/tracks.yaml static mapping)
#       │
#       ▼  get_available_interventions()  (filtered by time + setting)
#       │
#       ▼  ranked neuroarts activity list
#
# ─────────────────────────────────────────────────────────────────────────────


PIPELINE_STAGES: list[dict] = [
    {
        "stage": 1,
        "name": "Ingestion",
        "function": "pipeline.load_raw_data()",
        "inputs": ["data/raw/antiwork/*.csv", "data/raw/glassdoor_kaggle/*.csv"],
        "outputs": ["post_frames", "comment_frames", "glassdoor_df"],
        "notes": "Static Kaggle CSVs only. No network calls at runtime.",
    },
    {
        "stage": 2,
        "name": "Filtering & cleaning",
        "function": "pipeline.clean_text() + _apply_reddit_filters()",
        "inputs": ["post_frames", "comment_frames", "glassdoor_df"],
        "outputs": ["posts_clean", "comments_clean", "glassdoor_clean"],
        "notes": "Subreddit gate, text quality, dedup, Unicode normalisation.",
    },
    {
        "stage": 3,
        "name": "Sentiment labeling",
        "function": "src.data_collection.sentiment.add_sentiment()",
        "inputs": ["posts_clean", "comments_clean", "glassdoor_clean"],
        "outputs": ["same DataFrames + sentiment_score / sentiment_label columns"],
        "notes": "VADER by default; TextBlob available via config.",
    },
    {
        "stage": 4,
        "name": "Union & export",
        "function": "pipeline.export_outputs()",
        "inputs": ["posts_clean", "comments_clean", "glassdoor_clean"],
        "outputs": [
            "data/processed/workforce_posts_clean.parquet",
            "data/processed/workforce_comments_clean.parquet",
            "data/processed/workforce_glassdoor_clean.parquet",
            "data/processed/workforce_all_text.parquet",
            "data/processed/subreddit_coverage_report.csv",
        ],
        "notes": "workforce_all_text.parquet is the canonical modeling input.",
    },
    {
        "stage": 5,
        "name": "Embedding",
        "function": "pipeline.generate_embeddings() / cluster_embeddings.py",
        "inputs": ["data/processed/workforce_all_text.parquet"],
        "outputs": ["embeddings array (n_rows × 384)", "data/processed/embeddings_*.npz"],
        "notes": "all-MiniLM-L6-v2; L2-normalised; cached to disk.",
    },
    {
        "stage": 6,
        "name": "Clustering",
        "function": "pipeline.cluster_text() / cluster_embeddings.py",
        "inputs": ["embeddings array"],
        "outputs": [
            "data/processed/workforce_all_text_with_clusters.parquet",
            "data/processed/cluster_report.csv",
            "data/processed/cluster_representatives.csv",
        ],
        "notes": "KMeans; k chosen by silhouette; TF-IDF keywords per cluster.",
    },
    {
        "stage": 7,
        "name": "Runtime inference",
        "function": "src.cluster_router.route_user_input()",
        "inputs": [
            "user_text (free-text)",
            "data/processed/cluster_representatives.csv (pre-computed)",
        ],
        "outputs": ["cluster_id", "track_ids", "ranked interventions"],
        "notes": "No retraining. Cosine similarity at query time only.",
    },
]


def describe() -> None:
    """Print a human-readable pipeline overview to stdout."""
    print("=" * 70)
    print("NEUROART INTERVENTION APP — DATA PIPELINE OVERVIEW")
    print("=" * 70)
    for s in PIPELINE_STAGES:
        print(f"\nStage {s['stage']}: {s['name']}")
        print(f"  Function : {s['function']}")
        print(f"  Inputs   : {', '.join(s['inputs'])}")
        print(f"  Outputs  : {', '.join(s['outputs'])}")
        print(f"  Notes    : {s['notes']}")
    print()


if __name__ == "__main__":
    describe()
