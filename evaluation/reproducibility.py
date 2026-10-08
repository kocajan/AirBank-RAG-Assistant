from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

from data_collection.models import SourceDocument
from evaluation.config import GRID

PROJECT_VERSION = "1.2.0"
TRACKED_PACKAGES = (
    "beautifulsoup4",
    "fastapi",
    "httpx",
    "numpy",
    "openai",
    "pydantic",
    "pydantic-ai-slim",
    "pydantic-settings",
    "pymupdf",
    "streamlit",
    "uvicorn",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_fingerprint(documents: Iterable[SourceDocument]) -> str:
    """Stable fingerprint of source identity + extracted content, ignoring fetch timestamps."""

    rows = sorted(
        (
            document.id,
            str(document.url),
            document.content_sha256,
        )
        for document in documents
    )
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def installed_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def write_evaluation_manifest(
    *,
    output_path: Path,
    documents_path: Path,
    dataset_path: Path,
    judge_enabled: bool,
    judge_model: str | None,
    limit: int | None,
    retrieval_top_k: int,
) -> Path:
    from rag.index import load_documents

    documents = load_documents(documents_path)
    dataset_manifest_path = dataset_path.with_suffix(".manifest.json")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_version": PROJECT_VERSION,
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "packages": installed_versions(),
        "inputs": {
            "documents_path": str(documents_path),
            "documents_file_sha256": sha256_file(documents_path),
            "corpus_content_fingerprint": corpus_fingerprint(documents),
            "dataset_path": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "dataset_manifest_path": (
                str(dataset_manifest_path) if dataset_manifest_path.is_file() else None
            ),
            "dataset_manifest_sha256": (
                sha256_file(dataset_manifest_path)
                if dataset_manifest_path.is_file()
                else None
            ),
        },
        "evaluation": {
            "retrieval_top_k": retrieval_top_k,
            "limit": limit,
            "judge_enabled": judge_enabled,
            "judge_model": judge_model if judge_enabled else None,
            "grid": [
                {
                    "slug": config.slug,
                    "embedding_model": config.embedding_model,
                    "chunking_name": config.chunking.name,
                    "chunk_size": config.chunking.chunk_size,
                    "chunk_overlap": config.chunking.chunk_overlap,
                }
                for config in GRID
            ],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def verify_frozen_inputs(*, documents_path: Path, dataset_path: Path) -> list[str]:
    """Return reproducibility warnings for a frozen corpus/dataset pair."""

    warnings: list[str] = []
    dataset_manifest_path = dataset_path.with_suffix(".manifest.json")
    if not dataset_manifest_path.is_file():
        warnings.append(
            f"Dataset manifest is missing: {dataset_manifest_path}. "
            "Regenerate the evaluation dataset with the current code to record input fingerprints."
        )
        return warnings

    try:
        manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"Could not read dataset manifest: {exc}")
        return warnings

    expected_dataset_sha = manifest.get("dataset_sha256")
    if expected_dataset_sha and expected_dataset_sha != sha256_file(dataset_path):
        warnings.append("evaluation/dataset.jsonl does not match its recorded SHA-256 fingerprint.")

    expected_corpus_fingerprint = manifest.get("corpus_content_fingerprint")
    if expected_corpus_fingerprint:
        from rag.index import load_documents

        current = corpus_fingerprint(load_documents(documents_path))
        if current != expected_corpus_fingerprint:
            warnings.append(
                "The current processed corpus does not match the corpus used to generate the evaluation dataset."
            )

    return warnings
