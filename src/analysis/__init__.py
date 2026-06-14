# src/analysis/
#
# Embedding-based clustering, topic modelling, and downstream analysis of
# workforce sentiment data.
#
# Canonical input: data/processed/workforce_all_text.parquet
# Valid sources  : reddit, glassdoor
#
# Modules:
#   cluster_embeddings.py  — stratified sampling, SBERT embeddings, KMeans,
#                            TF-IDF keywords, cluster report + representatives
#
# Run:
#   python -m src.analysis.cluster_embeddings --help