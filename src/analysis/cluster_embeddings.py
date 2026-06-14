"""
src/analysis/cluster_embeddings.py

Embedding-based clustering of workforce sentiment text.

Canonical input : data/processed/workforce_all_text.parquet
Valid sources   : reddit, glassdoor

Outputs (all under --output_dir):
  workforce_all_text_with_clusters.parquet  — input rows + cluster_id + cluster_label
  cluster_report.csv                        — cluster_id, size, top_keywords, source/type breakdown
  cluster_representatives.csv              — N representative rows per cluster

Usage:
    python -m src.analysis.cluster_embeddings \\
        --input data/processed/workforce_all_text.parquet \\
        --output_dir data/processed \\
        --sample_n 20000 \\
        --mode kmeans \\
        --cache_embeddings true
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

# Project root on sys.path so src.* imports work when run as a module.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.merger import assert_no_youtube  # noqa: E402 — enforces source guardrail

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_SAMPLE_N = 20_000
_DEFAULT_BATCH_SIZE = 64
_DEFAULT_K_VALUES = [10, 15, 20, 30]
_DEFAULT_N_REPS = 5
_SILHOUETTE_SUBSAMPLE = 5_000
_TEXT_COL = "text"
_STRAT_COLS = ["source", "type"]
_RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _bool_arg(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embed and cluster workforce sentiment text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        default="data/processed/workforce_all_text.parquet",
        help="Path to input parquet file.",
    )
    p.add_argument(
        "--output_dir",
        default="data/processed",
        help="Directory for all output files.",
    )
    p.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help="Sentence-transformers model name.",
    )
    p.add_argument(
        "--sample_n",
        type=int,
        default=_DEFAULT_SAMPLE_N,
        help="Rows to sample (stratified). Ignored when --force_full_run is set.",
    )
    p.add_argument(
        "--force_full_run",
        action="store_true",
        help="Process all rows, ignoring --sample_n.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help="Batch size for sentence-transformer encoding.",
    )
    p.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=_DEFAULT_K_VALUES,
        help="KMeans k values to try. Best k selected by silhouette score.",
    )
    p.add_argument(
        "--mode",
        choices=["kmeans"],
        default="kmeans",
        help="Clustering algorithm (only kmeans supported).",
    )
    p.add_argument(
        "--n_reps",
        type=int,
        default=_DEFAULT_N_REPS,
        help="Representative texts to emit per cluster.",
    )
    p.add_argument(
        "--cache_embeddings",
        type=_bool_arg,
        default=True,
        metavar="true|false",
        help="Cache computed embeddings to disk.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Load + stratified sample
# ---------------------------------------------------------------------------

def load_and_sample(
    input_path: Path,
    sample_n: int,
    force_full: bool,
) -> pd.DataFrame:
    log.info(f"Loading {input_path} ...")
    df = pd.read_parquet(input_path)
    n_total = len(df)

    log.info(f"  Columns : {list(df.columns)}")
    log.info(f"  Total rows: {n_total:,}")
    for col in _STRAT_COLS:
        if col in df.columns:
            counts = df[col].value_counts()
            lines = "\n".join(f"    {k}: {v:,}" for k, v in counts.items())
            log.info(f"  {col}:\n{lines}")

    # Enforce source guardrail before any downstream work.
    assert_no_youtube(df, stage="cluster_embeddings.load")

    if force_full:
        log.info("--force_full_run set; using all %s rows.", f"{n_total:,}")
        return df.reset_index(drop=True)

    if n_total <= sample_n:
        log.info(
            "Dataset (%s rows) ≤ sample_n=%s; using all rows.",
            f"{n_total:,}",
            f"{sample_n:,}",
        )
        return df.reset_index(drop=True)

    log.warning(
        "Dataset has %s rows > sample_n=%s. "
        "Sampling with stratification. Pass --force_full_run to process all rows.",
        f"{n_total:,}",
        f"{sample_n:,}",
    )

    strat_cols_present = [c for c in _STRAT_COLS if c in df.columns]
    if strat_cols_present:
        strat_key = df[strat_cols_present].apply(
            lambda row: "|".join(str(row[c]) for c in strat_cols_present), axis=1
        )
        df = df.copy()
        df["_strat"] = strat_key
        parts = []
        for _name, group in df.groupby("_strat"):
            n_take = max(1, round(sample_n * len(group) / n_total))
            n_take = min(n_take, len(group))
            parts.append(group.sample(n_take, random_state=_RANDOM_SEED))
        sampled = pd.concat(parts, ignore_index=True).drop(columns=["_strat"])
        # Trim to exactly sample_n if rounding pushed us over.
        if len(sampled) > sample_n:
            sampled = sampled.sample(sample_n, random_state=_RANDOM_SEED).reset_index(drop=True)
    else:
        sampled = df.sample(sample_n, random_state=_RANDOM_SEED).reset_index(drop=True)

    log.info("  Sample size: %s", f"{len(sampled):,}")
    for col in strat_cols_present:
        counts = sampled[col].value_counts()
        lines = "\n".join(f"    {k}: {v:,}" for k, v in counts.items())
        log.info("  Sample %s:\n%s", col, lines)

    return sampled


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

def _model_tag(model_name: str) -> str:
    return (
        model_name.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def _compute_fingerprint(df: pd.DataFrame) -> str:
    if "id" in df.columns:
        raw = "|".join(df["id"].astype(str).sort_values().tolist())
    else:
        raw = "|".join(str(i) for i in sorted(df.index.tolist()))
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _cache_path(cache_dir: Path, model_name: str, fingerprint: str) -> Path:
    return cache_dir / f"embeddings_{_model_tag(model_name)}_{fingerprint}.npz"


def load_or_compute_embeddings(
    df: pd.DataFrame,
    model_name: str,
    batch_size: int,
    cache_dir: Path,
    use_cache: bool,
) -> np.ndarray:
    fingerprint = _compute_fingerprint(df)
    cache_file = _cache_path(cache_dir, model_name, fingerprint)

    if use_cache and cache_file.exists():
        log.info("Loading embeddings from cache: %s", cache_file.name)
        data = np.load(cache_file, allow_pickle=False)
        embeddings = data["embeddings"]
        log.info("  Shape: %s", embeddings.shape)
        return embeddings

    log.info(
        "Computing embeddings with '%s' (batch_size=%s) ...",
        model_name,
        batch_size,
    )
    # Lazy import — heavy dep, not needed if cache hits.
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    model = SentenceTransformer(model_name)
    texts = df[_TEXT_COL].fillna("").tolist()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalise → cosine KMeans
    )

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, embeddings=embeddings)
        log.info("Embeddings cached to %s", cache_file.name)

    return embeddings


# ---------------------------------------------------------------------------
# KMeans + silhouette k-selection
# ---------------------------------------------------------------------------

def select_best_k(
    embeddings: np.ndarray,
    k_values: list,
    seed: int = _RANDOM_SEED,
) -> tuple:
    """
    Fit KMeans on a fixed subsample for each k and return the k with the
    highest silhouette score. Also returns the full {k: score} dict.
    """
    n = len(embeddings)
    sub_n = min(_SILHOUETTE_SUBSAMPLE, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, sub_n, replace=False)
    sub = embeddings[idx]

    scores: dict = {}
    for k in k_values:
        if k >= sub_n:
            log.warning("k=%s ≥ subsample size=%s; skipping.", k, sub_n)
            continue
        log.info("  k=%s ...", k)
        km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=seed)
        labels = km.fit_predict(sub)
        if len(set(labels)) < 2:
            log.warning("  k=%s produced only one cluster; skipping silhouette.", k)
            continue
        s = silhouette_score(
            sub, labels, sample_size=min(sub_n, 3_000), random_state=seed
        )
        scores[k] = round(float(s), 4)
        log.info("    silhouette = %.4f", s)

    if not scores:
        best_k = k_values[0]
        log.warning("No valid silhouette scores computed; defaulting to k=%s.", best_k)
    else:
        best_k = max(scores, key=lambda k: scores[k])

    log.info("Best k=%s  (silhouette=%s)", best_k, scores.get(best_k, "?"))
    return best_k, scores


def run_kmeans(
    embeddings: np.ndarray,
    k: int,
    seed: int = _RANDOM_SEED,
) -> tuple:
    """Returns (cluster_labels, cluster_centroids)."""
    log.info("Fitting final KMeans k=%s on %s embeddings ...", k, f"{len(embeddings):,}")
    km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=seed)
    labels = km.fit_predict(embeddings)
    return labels, km.cluster_centers_


# ---------------------------------------------------------------------------
# TF-IDF keywords per cluster
# ---------------------------------------------------------------------------

def compute_tfidf_keywords(
    texts: list,
    cluster_labels: np.ndarray,
    n_keywords: int = 10,
) -> dict:
    """
    Sum TF-IDF weights across all documents in each cluster and return the
    top-n terms. Uses unigrams + bigrams; English stop words removed.
    """
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        min_df=3,
        max_df=0.85,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X = vectorizer.fit_transform(texts)
    vocab = np.array(vectorizer.get_feature_names_out())

    keywords: dict = {}
    for k in sorted(set(cluster_labels)):
        mask = cluster_labels == k
        # Sum TF-IDF weights across cluster documents then rank.
        cluster_tfidf = np.asarray(X[mask].sum(axis=0)).ravel()
        top_idx = cluster_tfidf.argsort()[::-1][:n_keywords]
        keywords[int(k)] = vocab[top_idx].tolist()
    return keywords


# ---------------------------------------------------------------------------
# Representative rows (closest to centroid)
# ---------------------------------------------------------------------------

def build_representatives(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    embeddings: np.ndarray,
    centroids: np.ndarray,
    cluster_label_map: dict,
    n_per_cluster: int,
) -> pd.DataFrame:
    rows = []
    for k in sorted(set(cluster_labels)):
        pos = np.where(cluster_labels == k)[0]
        centroid = centroids[k]
        dists = np.linalg.norm(embeddings[pos] - centroid, axis=1)
        top_pos = pos[dists.argsort()[:n_per_cluster]]
        for rank, i in enumerate(top_pos):
            row: dict = {
                "cluster_id": int(k),
                "cluster_label": cluster_label_map.get(int(k), ""),
                "rank": rank,
            }
            for col in ["id", "source", "type", "subreddit"]:
                if col in df.columns:
                    row[col] = df[col].iloc[i]
            row["text_snippet"] = str(df[_TEXT_COL].iloc[i] or "")[:300]
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cluster report
# ---------------------------------------------------------------------------

def build_cluster_report(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    keywords: dict,
    cluster_label_map: dict,
    sil_scores: dict,
) -> pd.DataFrame:
    rows = []
    for k in sorted(set(cluster_labels)):
        mask = cluster_labels == k
        sub = df.iloc[mask] if hasattr(df, "iloc") else df[mask]
        rows.append({
            "cluster_id": int(k),
            "cluster_label": cluster_label_map.get(int(k), ""),
            "size": int(mask.sum()),
            "top_keywords": ", ".join(keywords.get(int(k), [])),
            "source_counts": (
                sub["source"].value_counts().to_dict()
                if "source" in sub.columns else {}
            ),
            "type_counts": (
                sub["type"].value_counts().to_dict()
                if "type" in sub.columns else {}
            ),
        })
    report = pd.DataFrame(rows)
    report["silhouette_score"] = sil_scores.get(
        int(report["size"].idxmax()), None
    )
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = parse_args(argv)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)

    # ── 1. Load + sample ────────────────────────────────────────────────────
    df = load_and_sample(input_path, args.sample_n, args.force_full_run)
    log.info("Working with %s rows.", f"{len(df):,}")

    # ── 2. Embeddings ────────────────────────────────────────────────────────
    embeddings = load_or_compute_embeddings(
        df=df,
        model_name=args.model,
        batch_size=args.batch_size,
        cache_dir=output_dir,
        use_cache=args.cache_embeddings,
    )
    log.info("Embeddings shape: %s", embeddings.shape)

    # ── 3. Select best k via silhouette ──────────────────────────────────────
    log.info("Evaluating k in %s via silhouette score ...", args.k_values)
    best_k, sil_scores = select_best_k(embeddings, args.k_values)

    # ── 4. Final KMeans ──────────────────────────────────────────────────────
    cluster_labels, centroids = run_kmeans(embeddings, best_k)

    # ── 5. TF-IDF keywords + cluster labels ──────────────────────────────────
    log.info("Computing TF-IDF keywords per cluster ...")
    texts = df[_TEXT_COL].fillna("").tolist()
    keywords = compute_tfidf_keywords(texts, cluster_labels)
    # Cluster label = top-3 keywords joined by " / " (human-readable placeholder).
    cluster_label_map = {
        k: " / ".join(kws[:3]) for k, kws in keywords.items()
    }

    # ── 6. Attach cluster columns to sample DataFrame ────────────────────────
    df = df.copy()
    df["cluster_id"] = cluster_labels
    df["cluster_label"] = df["cluster_id"].map(cluster_label_map)

    # ── 7. Write outputs ─────────────────────────────────────────────────────
    out_parquet = output_dir / "workforce_all_text_with_clusters.parquet"
    df.to_parquet(out_parquet, index=False)
    log.info("Wrote %s", out_parquet)

    report_df = build_cluster_report(df, cluster_labels, keywords, cluster_label_map, sil_scores)
    out_report = output_dir / "cluster_report.csv"
    report_df.to_csv(out_report, index=False)
    log.info("Wrote %s", out_report)

    reps_df = build_representatives(
        df, cluster_labels, embeddings, centroids, cluster_label_map, args.n_reps
    )
    out_reps = output_dir / "cluster_representatives.csv"
    reps_df.to_csv(out_reps, index=False)
    log.info("Wrote %s", out_reps)

    # ── 8. Summary ───────────────────────────────────────────────────────────
    log.info("")
    log.info("=== Clustering complete ===")
    log.info("  k=%s  |  silhouette scores: %s", best_k, sil_scores)
    log.info("  Outputs written to: %s", output_dir)
    for path in (out_parquet, out_report, out_reps):
        log.info("    %s", path.name)


if __name__ == "__main__":
    main()