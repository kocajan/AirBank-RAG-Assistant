# Air Bank RAG Assistant

A portfolio proof of concept that answers questions about **publicly available Air Bank information** using retrieval-augmented generation (RAG).

The project covers the full workflow: public-data collection, cleaning, chunking, embeddings, vector retrieval, a PydanticAI chatbot, guardrails, and a reproducible comparison of multiple RAG configurations.

**Technical report:** [REPORT.md](REPORT.md)

<!-- After deployment, add the public URL here, for example:
**Live demo:** https://your-app.streamlit.app
-->

> This is an unofficial portfolio project. It is not affiliated with or endorsed by Air Bank and it cannot access customer accounts or perform banking actions.

## Highlights

- FastAPI + PydanticAI backend
- Streamlit chat interface
- RAG over public Air Bank web pages and official PDF documents
- OpenAI embeddings with a lightweight local NumPy vector index
- source links shown with every grounded answer
- short-term conversation memory
- simple safety and scope guardrails
- synthetic QA dataset generated from known source documents
- reproducible 3 × 3 evaluation of embedding models and chunking strategies
- Docker support for the full local demo

## Final experiment

The final evaluation compared three embedding models with three chunking strategies on 30 synthetic source-grounded questions.

| Embedding model | Chunking | R@1 | R@3 | R@5 | MRR@5 | nDCG@5 | Judge |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text-embedding-3-small | 800/100 | 70.0% | 93.3% | 96.7% | 0.814 | 0.853 | 4.83 |
| text-embedding-3-small | 1600/250 | 70.0% | 90.0% | 93.3% | 0.796 | 0.830 | 4.73 |
| text-embedding-3-small | 2400/400 | 80.0% | 93.3% | 96.7% | 0.862 | 0.888 | **4.97** |
| **text-embedding-3-large** | **800/100** | **83.3%** | **100.0%** | **100.0%** | **0.917** | **0.938** | 4.87 |
| text-embedding-3-large | 1600/250 | 83.3% | 93.3% | 100.0% | 0.898 | 0.924 | 4.80 |
| text-embedding-3-large | 2400/400 | 80.0% | 93.3% | 100.0% | 0.872 | 0.904 | 4.90 |
| text-embedding-ada-002 | 800/100 | 80.0% | 96.7% | 96.7% | 0.878 | 0.901 | 4.80 |
| text-embedding-ada-002 | 1600/250 | 80.0% | 93.3% | 93.3% | 0.867 | 0.884 | 4.80 |
| text-embedding-ada-002 | 2400/400 | 80.0% | 93.3% | 93.3% | 0.856 | 0.875 | 4.87 |

The final demo uses **`text-embedding-3-large` + 800/100 chunking** because it produced the strongest retrieval results overall while retaining high end-to-end answer quality.

The benchmark is intentionally controlled and synthetic; it is useful for comparing configurations, not for claiming production accuracy. See [REPORT.md](REPORT.md) for the interpretation and limitations.

## Architecture

```text
Air Bank public pages + PDFs
            │
            ▼
  collection + cleaning
            │
            ▼
        chunking
            │
            ▼
         embeddings
            │
            ▼
  local vector index (NumPy)
            │
            ▼
 PydanticAI retrieval tool
            │
            ▼
       OpenAI model
            │
            ▼
       FastAPI API
            │
            ▼
      Streamlit UI
```

## Repository structure

```text
backend/                 FastAPI API, PydanticAI agent, memory and runtime config
frontend/                Streamlit demo UI
rag/                     chunking, embeddings and local vector retrieval
data_collection/         Air Bank HTML/PDF collection and normalization
evaluation/              synthetic QA generation, metrics, judge and reports
scripts/                 CLI entry points for collection, indexing and evaluation
tests/                   unit/API tests
data/                     frozen corpus and active index (after final run)
REPORT.md                 technical write-up and experiment discussion
Dockerfile                backend container
Dockerfile.frontend       frontend container
docker-compose.yml        local two-service demo
railway.toml              backend deployment configuration
```

## Requirements

- Python 3.12 or 3.13
- [`uv`](https://docs.astral.sh/uv/)
- OpenAI API key
- optional: Docker + Docker Compose

> If you cloned this repository, **do not run `uv init`**. The repository already contains its own `pyproject.toml`.

## 1. Install the project

From the repository root:

```bash
uv sync
```

For a frozen published repository that already contains `uv.lock`:

```bash
uv sync --locked
```

Create local configuration:

```bash
cp .env.example .env
```

Add your OpenAI API key to `.env`:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.6-luna
EMBEDDING_MODEL=text-embedding-3-large
EVAL_GENERATION_MODEL=gpt-5.6-luna
EVAL_JUDGE_MODEL=gpt-5.6-luna

RAG_INDEX_DIR=data/index
RAG_TOP_K=5
RAG_MIN_SCORE=0.15
RAG_MAX_CHUNKS_PER_DOCUMENT=2
RAG_MAX_SOURCES=5

MAX_TURNS_PER_SESSION=30
MAX_SESSIONS=500

BACKEND_URL=http://localhost:8000
```

Never commit `.env`.

## 2. Prepare data from scratch

Collect up to 200 public Air Bank HTML pages plus selected official documents:

```bash
uv run python scripts/collect_data.py --max-html 200
```

Outputs:

```text
data/processed/documents.jsonl
data/processed/collection_summary.json
```

Build the final chatbot index:

```bash
uv run python scripts/build_index.py --chunk-size 800 --chunk-overlap 100
```

Quick retrieval check:

```bash
uv run python scripts/search_index.py "Jak funguje bonusové úročení spořicího účtu?"
```

## 3. Run the chatbot locally

Start the backend:

```bash
uv run uvicorn backend.main:app --reload
```

Check:

```text
http://localhost:8000/health
```

In a second terminal, start the frontend:

```bash
uv run streamlit run frontend/app.py
```

Open:

```text
http://localhost:8501
```

## 4. Reproduce the evaluation

### Generate a new synthetic QA benchmark

```bash
uv run python scripts/generate_eval_dataset.py --count 30 --seed 42
```

This creates:

```text
evaluation/dataset.jsonl
evaluation/dataset.manifest.json
```

For the reported experiment, the preferred workflow is to reuse the **frozen committed dataset** rather than regenerate it.

### Retrieval-only smoke test

```bash
uv run python scripts/run_evaluation.py --limit 3
```

### Full retrieval grid

```bash
uv run python scripts/run_evaluation.py --force-rebuild
```

### Full retrieval + answer-quality evaluation

First run a small smoke test:

```bash
uv run python scripts/run_evaluation.py --with-judge --limit 3
```

Then run the complete experiment:

```bash
uv run python scripts/run_evaluation.py --with-judge --force-rebuild
```

The experiment evaluates:

```text
Embedding models:
  text-embedding-3-small
  text-embedding-3-large
  text-embedding-ada-002

Chunking:
  800 / 100
  1600 / 250
  2400 / 400
```

Outputs are written to:

```text
evaluation/results/latest_report.md
evaluation/results/latest_summary.json
evaluation/results/latest_details.json
evaluation/results/reproducibility_manifest.json
```

After frozen `documents.jsonl` and `dataset.jsonl` exist, the convenience command is:

```bash
uv run python scripts/reproduce_experiment.py --with-judge --force-rebuild
```

## 5. Run tests

```bash
uv run pytest
```

## 6. Run with Docker

The Docker demo expects:

- `.env`
- `uv.lock`
- the active index in `data/index/`

Build both services:

```bash
docker compose build
```

Start them:

```bash
docker compose up
```

Open:

```text
Frontend: http://localhost:8501
Backend:  http://localhost:8000
Health:   http://localhost:8000/health
```

Stop with `Ctrl+C`, then:

```bash
docker compose down
```

## 7. Deployment

The intended public-demo setup is:

```text
Streamlit Community Cloud  →  Railway FastAPI backend  →  OpenAI API
                                      │
                                      └─ local packaged RAG index
```

### Backend: Railway

Deploy this repository as a Railway service using the root `Dockerfile`. `railway.toml` configures `/health` as the health check.

Set these Railway variables:

```text
OPENAI_API_KEY=<secret>
OPENAI_MODEL=gpt-5.6-luna
EMBEDDING_MODEL=text-embedding-3-large
RAG_INDEX_DIR=data/index
RAG_TOP_K=5
RAG_MIN_SCORE=0.15
RAG_MAX_CHUNKS_PER_DOCUMENT=2
RAG_MAX_SOURCES=5
MAX_TURNS_PER_SESSION=30
MAX_SESSIONS=500
```

The committed `data/index/` must contain the final built index so the backend can answer immediately after startup.

### Frontend: Streamlit Community Cloud

Deploy the same GitHub repository and use:

```text
frontend/app.py
```

as the entrypoint.

Add this secret in Streamlit Community Cloud:

```toml
BACKEND_URL = "https://<your-railway-domain>"
```

After deployment, add the public Streamlit URL near the top of this README.

## Reproducibility

The experiment is reproducible **from frozen inputs**, not guaranteed to be bit-for-bit deterministic.

For the reported experiment, preserve these files in Git:

```text
uv.lock
data/processed/documents.jsonl
data/processed/collection_summary.json
evaluation/dataset.jsonl
evaluation/dataset.manifest.json
evaluation/results/latest_report.md
evaluation/results/latest_summary.json
evaluation/results/latest_details.json
evaluation/results/reproducibility_manifest.json
```

A fresh crawl or regenerated QA dataset is a replication of the methodology, not an exact reproduction, because the Air Bank website and hosted model behavior can change over time.

Before publishing the repository, run:

```bash
uv run python scripts/check_release.py
```

The checker verifies that the expected frozen artifacts and final demo index are present and that the active index uses the selected configuration.

## Limitations

This is deliberately a small proof of concept. The main limitations are:

- synthetic rather than human-authored evaluation questions;
- document-level rather than passage-level retrieval labels;
- only 30 evaluation questions;
- dense retrieval only, without BM25/hybrid retrieval or reranking;
- simple character-based chunking;
- hosted LLM/embedding services can evolve;
- in-memory conversation sessions;
- public-data-only scope.

The production-oriented improvements and rationale are discussed in [REPORT.md](REPORT.md).

## License and disclaimer

Code is available under the [MIT License](LICENSE).

Air Bank is a trademark of its respective owner. This repository is an independent portfolio demonstration and is not an official Air Bank product.
