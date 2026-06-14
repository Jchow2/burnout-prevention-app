#!/usr/bin/env python3
"""
Workforce sentiment pipeline — single entrypoint.

CANONICAL DATA SOURCES: reddit + glassdoor (Kaggle-based static files only).
YouTube ingestion is permanently removed. The merger enforces this at runtime
via assert_no_youtube(); any row with source not in {reddit, glassdoor} raises.

CANONICAL MODELING INPUT: data/processed/workforce_all_text.parquet

Usage
─────
    python pipeline.py                      # uses config.yaml in project root
    python pipeline.py --config my.yaml     # custom config path
    python pipeline.py --config config.yaml --dry-run   # print counts, no writes

Outputs (written to output_dir from config)
───────────────────────────────────────────
    workforce_posts_clean.parquet        Reddit posts
    workforce_comments_clean.parquet     Reddit comments
    workforce_glassdoor_clean.parquet    Glassdoor reviews
    workforce_all_text.parquet           Union of all sources ← modeling input
    subreddit_coverage_report.csv        Coverage by source + subreddit
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.data_collection.ingest import (
    load_antiwork_posts,
    load_antiwork_comments,
    load_reddit_sentiment_posts,
    load_reddit_sentiment_comments,
    load_reddit_sentiment_user_posts,
)
from src.data_collection.normalize import (
    normalize_antiwork_posts,
    normalize_antiwork_comments,
    normalize_reddit_sentiment_posts,
    normalize_reddit_sentiment_comments,
    normalize_reddit_sentiment_user_posts,
)
from src.data_collection.filters import (
    filter_by_subreddit,
    filter_text_quality,
    filter_min_score,
    deduplicate,
    apply_date_filter,
    sample_per_subreddit,
)
from src.data_collection.glassdoor import load_glassdoor, RATING_COLS  # noqa: F401
from src.data_collection.sentiment import add_sentiment
from src.merger import assert_no_youtube

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pipeline")

# Columns included in the union workforce_all_text.parquet
# (rating columns are Glassdoor-only so excluded from the union to keep schema clean)
UNION_COLUMNS = [
    "id", "type", "source", "subreddit", "title", "text",
    "author", "created_utc", "score", "url",
    "sentiment_score", "sentiment_label", "sentiment_source",
]


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ── Coverage report ───────────────────────────────────────────────────────────

def build_coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by (source, subreddit) so Reddit and Glassdoor appear separately.
    For Glassdoor rows, subreddit is '' which shows as 'glassdoor' in the source col.
    """
    rows = []
    group_cols = [c for c in ["source", "subreddit"] if c in df.columns]
    if not group_cols:
        return pd.DataFrame()

    for keys, grp in df.groupby(group_cols):
        source  = keys[0] if isinstance(keys, tuple) else keys
        subreddit = keys[1] if isinstance(keys, tuple) and len(keys) > 1 else ""
        rows.append({
            "source":           source,
            "subreddit":        subreddit,
            "count":            len(grp),
            "pct_missing_text": round((grp["text"].str.strip() == "").mean() * 100, 2),
            "date_min":         grp["created_utc"].min() if "created_utc" in grp else pd.NaT,
            "date_max":         grp["created_utc"].max() if "created_utc" in grp else pd.NaT,
        })
    return pd.DataFrame(rows).sort_values(["source", "count"], ascending=[True, False]).reset_index(drop=True)


# ── Reddit stages ─────────────────────────────────────────────────────────────

def _load_and_normalise_reddit(cfg: dict, base: Path) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Stage 1+2 for Reddit: ingest raw files and normalise to common schema."""
    inputs = cfg.get("inputs", {})
    chunk  = cfg.get("chunk_size", 50_000)

    posts: list[pd.DataFrame] = []
    comments: list[pd.DataFrame] = []

    def _p(key: str) -> Path | None:
        val = inputs.get(key)
        return (base / val) if val else None

    if p := _p("antiwork_posts"):
        raw = load_antiwork_posts(p, chunk)
        posts.append(normalize_antiwork_posts(raw, str(p)))

    if p := _p("antiwork_comments"):
        raw = load_antiwork_comments(p, chunk)
        comments.append(normalize_antiwork_comments(raw, str(p)))

    if p := _p("reddit_sentiment_posts"):
        raw = load_reddit_sentiment_posts(p, chunk)
        posts.append(normalize_reddit_sentiment_posts(raw, str(p)))

    if p := _p("reddit_sentiment_user_posts"):
        raw = load_reddit_sentiment_user_posts(p, chunk)
        posts.append(normalize_reddit_sentiment_user_posts(raw, str(p)))

    if p := _p("reddit_sentiment_comments"):
        raw = load_reddit_sentiment_comments(p, chunk)
        comments.append(normalize_reddit_sentiment_comments(raw, str(p)))

    return posts, comments


def _apply_reddit_filters(
    frames: list[pd.DataFrame],
    allowlist: list[str],
    min_len: int,
    date_from: str | None,
    date_to: str | None,
    max_per_sub: int | None,
    min_score: int | None,
) -> pd.DataFrame:
    """Stage 3 for Reddit: subreddit gate + quality filters."""
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    logger.info("Reddit merged: %d rows before filtering", len(df))

    df = filter_by_subreddit(df, allowlist)
    df = filter_text_quality(df, min_len)
    if min_score is not None:
        df = filter_min_score(df, min_score)
    df = deduplicate(df)
    df = apply_date_filter(df, date_from, date_to)

    if max_per_sub:
        df = sample_per_subreddit(df, max_per_sub)

    df["source"] = "reddit"
    return df


# ── Glassdoor stage ───────────────────────────────────────────────────────────

def _load_and_process_glassdoor(cfg: dict, base: Path) -> pd.DataFrame:
    """
    Stage 1-3 for Glassdoor.

    Intentionally bypasses the subreddit allowlist — Glassdoor is company-
    scoped, not subreddit-scoped.  Text quality and date filters still apply.
    """
    inputs = cfg.get("inputs", {})
    gd_path_str = inputs.get("glassdoor_reviews")
    if not gd_path_str:
        return pd.DataFrame()

    gd_path = base / gd_path_str

    df = load_glassdoor(
        path            = gd_path,
        anonymize_firm  = cfg.get("glassdoor_anonymize_firm", True),
        keep_location   = cfg.get("glassdoor_keep_location",  False),
        min_text_length = cfg.get("min_text_length", 30),
        chunk_size      = cfg.get("chunk_size", 50_000),
        source_file     = str(gd_path),
    )

    if df.empty:
        return df

    # Text quality (min_length already applied inside load_glassdoor, but
    # run again in case config differs; URL-only filter is worth adding too)
    df = filter_text_quality(df, cfg.get("min_text_length", 30))
    df = deduplicate(df)

    date_from = cfg.get("date_from")
    date_to   = cfg.get("date_to")
    df = apply_date_filter(df, date_from, date_to)

    logger.info("Glassdoor final: %d rows", len(df))
    return df


# ── Main run ──────────────────────────────────────────────────────────────────

def run(cfg: dict, dry_run: bool = False) -> None:
    base    = Path(__file__).parent
    out_dir = base / cfg.get("output_dir", "data/processed")

    allowlist   = cfg.get("subreddits", [])
    min_len     = cfg.get("min_text_length", 30)
    date_from   = cfg.get("date_from")
    date_to     = cfg.get("date_to")
    max_per_sub = cfg.get("max_per_subreddit")
    min_score   = cfg.get("min_score")
    sent_mode   = cfg.get("sentiment_mode", "vader")

    # ── Stage 1-3: Reddit ─────────────────────────────────────────────────
    post_frames, comment_frames = _load_and_normalise_reddit(cfg, base)
    posts_df    = _apply_reddit_filters(post_frames,    allowlist, min_len, date_from, date_to, max_per_sub, min_score)
    comments_df = _apply_reddit_filters(comment_frames, allowlist, min_len, date_from, date_to, max_per_sub, min_score)
    logger.info("Reddit final — posts: %d, comments: %d", len(posts_df), len(comments_df))

    # ── Stage 1-3: Glassdoor ──────────────────────────────────────────────
    glassdoor_df = _load_and_process_glassdoor(cfg, base)

    # ── Stage 4: Sentiment ────────────────────────────────────────────────
    if not posts_df.empty:
        posts_df     = add_sentiment(posts_df,     sent_mode)
    if not comments_df.empty:
        comments_df  = add_sentiment(comments_df,  sent_mode)
    if not glassdoor_df.empty:
        glassdoor_df = add_sentiment(glassdoor_df, sent_mode)

    if dry_run:
        logger.info("[dry-run] No files written.")
        for label, df in [("Posts", posts_df), ("Comments", comments_df), ("Glassdoor", glassdoor_df)]:
            if not df.empty:
                print(f"\n{label} sample:")
                cols = [c for c in ["source", "subreddit", "type", "text", "sentiment_label"] if c in df.columns]
                print(df[cols].head(3).to_string())
        return

    # ── Stage 5: Write outputs ────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write_parquet(df: pd.DataFrame, name: str) -> None:
        if df.empty:
            logger.warning("Skipping %s — no rows", name)
            return
        dest = out_dir / name
        df.to_parquet(dest, index=False)
        logger.info("Wrote %s (%d rows)", dest, len(df))

    _write_parquet(posts_df,     "workforce_posts_clean.parquet")
    _write_parquet(comments_df,  "workforce_comments_clean.parquet")
    _write_parquet(glassdoor_df, "workforce_glassdoor_clean.parquet")

    # ── Union: workforce_all_text.parquet ─────────────────────────────────
    # Only common columns go into the union — Glassdoor rating columns are
    # preserved in workforce_glassdoor_clean.parquet only.
    union_frames = []
    for df in [posts_df, comments_df, glassdoor_df]:
        if df.empty:
            continue
        present = [c for c in UNION_COLUMNS if c in df.columns]
        union_frames.append(df[present])

    if union_frames:
        all_df = pd.concat(union_frames, ignore_index=True)
        assert_no_youtube(all_df, stage="pipeline.run")
        _write_parquet(all_df, "workforce_all_text.parquet")

        coverage = build_coverage_report(all_df)
        report_path = out_dir / "subreddit_coverage_report.csv"
        coverage.to_csv(report_path, index=False)
        logger.info("Wrote %s", report_path)

        print("\n── Coverage report ─────────────────────────────────")
        print(coverage.to_string(index=False))


# ── Named pipeline stage functions ────────────────────────────────────────────
#
# These are thin, named wrappers that make each pipeline stage explicitly
# callable — useful for notebooks, tests, and demo walkthroughs.
# The existing run() function continues to work unchanged; these are
# supplementary entry points for transparency and reproducibility.

def load_raw_data(cfg: dict, base: Path) -> tuple[list[pd.DataFrame], list[pd.DataFrame], pd.DataFrame]:
    """
    Stage 1 — Ingest raw source files from disk.

    Reads static Kaggle CSVs for Reddit and Glassdoor.  No network calls;
    all data must already be present in data/raw/.

    Inputs
    ------
    cfg  : pipeline config dict (from config.yaml)
    base : project root Path (file paths in cfg are relative to this)

    Outputs
    -------
    post_frames    : list of un-filtered Reddit post DataFrames
    comment_frames : list of un-filtered Reddit comment DataFrames
    glassdoor_df   : un-filtered Glassdoor DataFrame (empty if path not set)

    If a raw file is missing, the corresponding loader logs a warning and
    returns an empty DataFrame — downstream stages handle empty inputs
    gracefully so the pipeline can still run with partial data.

    STUB NOTE: If you have not yet placed raw data in data/raw/, set the
    corresponding key in config.yaml to null or leave it absent.  The stage
    will skip that source and continue.
    """
    post_frames, comment_frames = _load_and_normalise_reddit(cfg, base)
    glassdoor_df = _load_and_process_glassdoor(cfg, base)
    return post_frames, comment_frames, glassdoor_df


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 2 — Normalise and deduplicate text fields.

    Delegates to src.text_cleaner.clean_dataframe, which:
    - Normalises Unicode (smart quotes, dashes → ASCII equivalents)
    - Strips URLs, emails, and Reddit markdown (bold, strikethrough, headers)
    - Removes Glassdoor-specific boilerplate (Pros / Cons labels, HTML entities)
    - Collapses runs of whitespace
    - Drops rows with fewer than 3 words after cleaning
    - Deduplicates on the cleaned text field

    Inputs  : DataFrame with a 'text' column
    Outputs : Cleaned DataFrame — same schema, possibly fewer rows
    """
    from src.text_cleaner import clean_dataframe
    return clean_dataframe(df)


def generate_embeddings(
    df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    cache_dir: Path | None = None,
    use_cache: bool = True,
):
    """
    Stage 3 — Encode text rows as dense sentence embeddings.

    Uses sentence-transformers (all-MiniLM-L6-v2 by default) to produce
    L2-normalised 384-dimensional vectors.  Embeddings are cached to disk
    as .npz files keyed by model name + row fingerprint, so reruns skip
    re-encoding automatically.

    Inputs
    ------
    df         : DataFrame with a 'text' column (typically workforce_all_text.parquet)
    model_name : Any sentence-transformers model name.  Must match the model
                 used when the cluster centroids were computed, or similarity
                 scores at inference time will be meaningless.
    batch_size : Rows processed per GPU/CPU batch (tune for your hardware)
    cache_dir  : Directory for .npz cache files (default: data/processed/)
    use_cache  : Skip encode if a matching cache file exists

    Outputs : numpy array of shape (n_rows, embedding_dim)

    Requires: pip install sentence-transformers
    """
    from src.analysis.cluster_embeddings import load_or_compute_embeddings
    _cache = cache_dir or (Path(__file__).parent / "data" / "processed")
    return load_or_compute_embeddings(df, model_name, batch_size, _cache, use_cache)


def cluster_text(
    embeddings,
    df: pd.DataFrame,
    k_values: list[int] | None = None,
):
    """
    Stage 4 — Cluster embeddings via KMeans + silhouette-based k selection.

    Tries each value of k, selects the one with the highest silhouette score
    on a 5 000-row subsample, then fits a final KMeans on all embeddings.
    Computes TF-IDF keywords per cluster as human-readable labels.

    Inputs
    ------
    embeddings : L2-normalised numpy array (output of generate_embeddings)
    df         : Source DataFrame aligned with embeddings (same row order)
    k_values   : k values to try; default [10, 15, 20, 30]

    Outputs
    -------
    cluster_labels   : int array of shape (n_rows,) — cluster assignment per row
    centroids        : float array of shape (k, embedding_dim) — cluster centres
    tfidf_keywords   : dict mapping cluster_id → list of top-10 keyword strings
    """
    from src.analysis.cluster_embeddings import select_best_k, run_kmeans, compute_tfidf_keywords
    _k = k_values or [10, 15, 20, 30]
    best_k, _ = select_best_k(embeddings, _k)
    labels, centroids = run_kmeans(embeddings, best_k)
    texts = df["text"].fillna("").tolist()
    keywords = compute_tfidf_keywords(texts, labels)
    return labels, centroids, keywords


def export_outputs(
    out_dir: Path,
    posts_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    glassdoor_df: pd.DataFrame,
) -> None:
    """
    Stage 5 — Write all cleaned DataFrames to disk as parquet + CSV.

    Writes per-source files and the canonical union file used as the
    modeling input for embedding + clustering.

    Inputs
    ------
    out_dir      : Destination directory (created if absent)
    posts_df     : Cleaned Reddit posts
    comments_df  : Cleaned Reddit comments
    glassdoor_df : Cleaned Glassdoor reviews

    Outputs (side-effect only — files written to out_dir)
    ────────────────────────────────────────────────────
    workforce_posts_clean.parquet
    workforce_comments_clean.parquet
    workforce_glassdoor_clean.parquet
    workforce_all_text.parquet        ← canonical modeling input
    subreddit_coverage_report.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write(df: pd.DataFrame, name: str) -> None:
        if df.empty:
            logger.warning("Skipping %s — no rows", name)
            return
        path = out_dir / name
        df.to_parquet(path, index=False)
        logger.info("Wrote %s (%d rows)", path, len(df))

    _write(posts_df,     "workforce_posts_clean.parquet")
    _write(comments_df,  "workforce_comments_clean.parquet")
    _write(glassdoor_df, "workforce_glassdoor_clean.parquet")

    union_frames = [
        df[[c for c in UNION_COLUMNS if c in df.columns]]
        for df in [posts_df, comments_df, glassdoor_df]
        if not df.empty
    ]
    if union_frames:
        all_df = pd.concat(union_frames, ignore_index=True)
        assert_no_youtube(all_df, stage="export_outputs")
        _write(all_df, "workforce_all_text.parquet")
        coverage = build_coverage_report(all_df)
        coverage.to_csv(out_dir / "subreddit_coverage_report.csv", index=False)
        logger.info("Coverage report written.")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Workforce sentiment pipeline")
    parser.add_argument("--config",  default="config.yaml", help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true",   help="Run without writing files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()