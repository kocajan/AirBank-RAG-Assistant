# Data directories

`raw/` contains downloaded HTML/PDF files and is intentionally ignored by Git.

`processed/documents.jsonl` and `processed/collection_summary.json` form the normalized corpus snapshot. For a reported/reproducible experiment, keep these two files in the repository so the benchmark can be rerun against exactly the same public-data snapshot.

`index/` contains the active chatbot vector index. It can be regenerated from the processed corpus with `scripts/build_index.py`.

`evaluation/indexes/` contains temporary indexes for the 3 × 3 evaluation grid and is always regenerable.
