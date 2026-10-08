from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_REPO_FILES = (
    "README.md",
    "REPORT.md",
    "pyproject.toml",
    ".env.example",
    "Dockerfile",
    "Dockerfile.frontend",
    "docker-compose.yml",
)

FROZEN_EXPERIMENT_FILES = (
    "uv.lock",
    "data/processed/documents.jsonl",
    "data/processed/collection_summary.json",
    "evaluation/dataset.jsonl",
    "evaluation/dataset.manifest.json",
    "evaluation/results/reproducibility_manifest.json",
)

INDEX_FILES = (
    "data/index/chunks.jsonl",
    "data/index/embeddings.npy",
    "data/index/index_metadata.json",
)

EXPECTED_INDEX = {
    "embedding_model": "text-embedding-3-large",
    "chunk_size": 800,
    "chunk_overlap": 100,
}


def exists(relative: str) -> bool:
    return (ROOT / relative).is_file()


def main() -> None:
    failures: list[str] = []
    warnings: list[str] = []

    for path in REQUIRED_REPO_FILES:
        if not exists(path):
            failures.append(f"missing repository file: {path}")

    for path in FROZEN_EXPERIMENT_FILES:
        if not exists(path):
            warnings.append(f"missing frozen experiment artifact: {path}")

    for path in INDEX_FILES:
        if not exists(path):
            warnings.append(f"missing deployed chatbot index file: {path}")

    metadata_path = ROOT / "data/index/index_metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for key, expected in EXPECTED_INDEX.items():
                actual = metadata.get(key)
                if actual != expected:
                    warnings.append(
                        f"active index {key}={actual!r}; final demo expects {expected!r}"
                    )
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"could not read active index metadata: {exc}")

    env_path = ROOT / ".env"
    if env_path.is_file():
        warnings.append(".env exists locally; verify it is not committed (it is gitignored).")

    print("Air Bank RAG release check")
    print("=" * 26)

    if failures:
        print("\nFAIL")
        for item in failures:
            print(f"- {item}")
    else:
        print("\nCore repository files: OK")

    if warnings:
        print("\nBefore publishing/deploying:")
        for item in warnings:
            print(f"- {item}")
    else:
        print("Frozen experiment + active demo index: OK")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
