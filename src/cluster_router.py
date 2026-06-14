"""
src/cluster_router.py

Real-time inference layer: map free-text user input to the nearest
pre-computed cluster, then route to the appropriate intervention track.

Hybrid architecture
───────────────────
This module bridges the two halves of the system:

  STATIC LAYER (pre-computed, offline)
    Raw text (Reddit + Glassdoor)
      → clean → embed → KMeans → cluster_representatives.csv
                                  cluster_report.csv

  RUNTIME LAYER (on-the-fly, per user)
    User free-text input
      → embed (same model) → cosine similarity vs. representatives
      → cluster_id → track_ids → ranked interventions

Because both the corpus and the user input are embedded with the same
sentence-transformers model, their vectors live in the same space and
cosine similarity is a meaningful proximity measure.

Fallback: when sentence-transformers is not installed, TF-IDF cosine
similarity is used instead.  Results are less accurate but the function
still returns a valid cluster_id.

Public API
──────────
    match_input_to_cluster(user_text, ...) -> dict
        Maps raw text to the nearest cluster.
        Returns cluster_id, similarity score, and method used.

    route_user_input(user_text, ...) -> dict
        Full end-to-end routing:
        user_text → cluster → tracks → ranked interventions.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_REPS_PATH = _PROJECT_ROOT / "data" / "processed" / "cluster_representatives.csv"
_DEFAULT_MODEL = "all-MiniLM-L6-v2"


# ── Similarity helpers ─────────────────────────────────────────────────────────

def _cosine_sim_rows(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a single query vector and each row of matrix."""
    q = query / (np.linalg.norm(query) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    return (matrix / norms) @ q


def _embed_sbert(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def _embed_tfidf(query: str, corpus: list[str]) -> np.ndarray:
    """TF-IDF cosine similarity fallback — no sentence-transformers required."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(stop_words="english", max_features=5_000)
    X = vec.fit_transform([query] + corpus)
    return cosine_similarity(X[:1], X[1:]).ravel()


# ── Core public API ────────────────────────────────────────────────────────────

def match_input_to_cluster(
    user_text: str,
    reps_path: str | Path = _DEFAULT_REPS_PATH,
    model_name: str = _DEFAULT_MODEL,
) -> dict[str, Any]:
    """
    Map a free-text input to the nearest pre-computed cluster.

    The cluster representatives file (cluster_representatives.csv) is produced
    by the clustering pipeline and contains the 5 texts per cluster closest to
    each centroid.  This function embeds the user input with the same model
    and scores cosine similarity against those representatives.

    Parameters
    ----------
    user_text  : Raw text from a user — check-in debrief, typed concern, etc.
    reps_path  : Path to cluster_representatives.csv.
                 Default: data/processed/cluster_representatives.csv
    model_name : Sentence-transformers model name.  Must match the model used
                 during clustering so the embedding space is consistent.

    Returns
    -------
    dict with keys:
        cluster_id    : int   — best-matching cluster
        cluster_label : str   — human-readable TF-IDF label for that cluster
        similarity    : float — cosine similarity (0–1; higher = stronger match)
        method        : str   — "sbert" or "tfidf" (indicates fallback)
        top_matches   : list[dict] — top-3 clusters with cluster_id, label, similarity

    Raises
    ------
    FileNotFoundError
        If cluster_representatives.csv does not exist.  Run the clustering
        pipeline first:
            python -m src.analysis.cluster_embeddings \\
                --input data/processed/workforce_all_text.parquet
    """
    reps_path = Path(reps_path)
    if not reps_path.exists():
        raise FileNotFoundError(
            f"Cluster representatives not found: {reps_path}\n"
            "Run the clustering pipeline first:\n"
            "  python -m src.analysis.cluster_embeddings "
            "--input data/processed/workforce_all_text.parquet"
        )

    reps = pd.read_csv(reps_path)
    if reps.empty:
        raise ValueError("cluster_representatives.csv is empty.")

    # Build one representative document per cluster by concatenating snippets.
    cluster_texts: dict[int, list[str]] = {}
    cluster_labels: dict[int, str] = {}
    for _, row in reps.iterrows():
        cid = int(row["cluster_id"])
        cluster_texts.setdefault(cid, []).append(str(row.get("text_snippet", "")))
        if cid not in cluster_labels:
            cluster_labels[cid] = str(row.get("cluster_label", f"Cluster {cid}"))

    cluster_ids = sorted(cluster_texts.keys())
    corpus = [" ".join(cluster_texts[cid]) for cid in cluster_ids]

    method = "sbert"
    try:
        all_embeddings = _embed_sbert([user_text] + corpus, model_name)
        query_emb = all_embeddings[0]
        corpus_emb = all_embeddings[1:]
        sims = _cosine_sim_rows(query_emb, corpus_emb)
    except ImportError:
        log.warning(
            "sentence_transformers not installed — falling back to TF-IDF. "
            "Install with: pip install sentence-transformers"
        )
        method = "tfidf"
        sims = _embed_tfidf(user_text, corpus)

    ranked = np.argsort(sims)[::-1]
    best_idx = ranked[0]
    best_cluster_id = cluster_ids[best_idx]

    top_matches = [
        {
            "cluster_id": cluster_ids[i],
            "cluster_label": cluster_labels[cluster_ids[i]],
            "similarity": round(float(sims[i]), 4),
        }
        for i in ranked[:3]
    ]

    return {
        "cluster_id": best_cluster_id,
        "cluster_label": cluster_labels[best_cluster_id],
        "similarity": round(float(sims[best_idx]), 4),
        "method": method,
        "top_matches": top_matches,
    }


def route_user_input(
    user_text: str,
    time_available_min: int = 5,
    setting: str = "break_room",
    reps_path: str | Path = _DEFAULT_REPS_PATH,
    model_name: str = _DEFAULT_MODEL,
) -> dict[str, Any]:
    """
    Full end-to-end runtime inference: text → cluster → tracks → interventions.

    This is the main entry point for the hybrid AI inference layer.
    It accepts raw user text and returns a ranked list of neuroarts
    interventions, with a full explanation of the routing decision.

    Flow
    ────
        user_text
            │
            ▼
        match_input_to_cluster()   [embedding + cosine similarity]
            │
            ▼
        cluster_id
            │
            ▼
        cluster_to_track_ids()     [static YAML mapping]
            │
            ▼
        get_available_interventions()  [filtered by time + setting]
            │
            ▼
        ranked interventions + explanation

    Parameters
    ----------
    user_text         : Free-text input from the user.
    time_available_min: Minutes available (5, 10, or 15).
    setting           : "break_room", "commute", "home", or "other".
    reps_path         : Path to cluster_representatives.csv.
    model_name        : Sentence-transformers model name.

    Returns
    -------
    dict with keys:
        cluster_match  : dict        — output of match_input_to_cluster()
        track_ids      : list[str]   — intervention tracks covering this cluster
        interventions  : list[dict]  — ranked activity options
        explanation    : str         — human-readable routing summary
    """
    from src.analysis.personas import cluster_to_track_ids
    from src.intervention_recommender import get_available_interventions

    cluster_match = match_input_to_cluster(user_text, reps_path, model_name)
    cluster_id = cluster_match["cluster_id"]
    track_ids = cluster_to_track_ids(cluster_id)

    interventions = get_available_interventions(
        time_available_min=time_available_min,
        setting=setting,
        dominant_dimension="mind",  # free-text input defaults to mind dimension
        burnout_score=None,
    )

    track_summary = ", ".join(track_ids) if track_ids else "no tracks configured for this cluster"
    explanation = (
        f"Input matched cluster {cluster_id} "
        f"('{cluster_match['cluster_label']}', "
        f"similarity={cluster_match['similarity']:.2f} via {cluster_match['method']}). "
        f"Mapped to track(s): {track_summary}."
    )

    return {
        "cluster_match": cluster_match,
        "track_ids": track_ids,
        "interventions": interventions,
        "explanation": explanation,
    }
