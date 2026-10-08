from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

from data_collection.models import SourceDocument
from evaluation.models import EvaluationItem, GeneratedQA
from rag.index import load_documents
from evaluation.reproducibility import corpus_fingerprint, sha256_file


GENERATION_SYSTEM_PROMPT = """
You generate evaluation questions for a RAG system about Air Bank.

You will receive an excerpt from one official Air Bank source document.
Create exactly one factual question-answer pair in Czech.

Requirements:
- The question must be naturally phrased like a user question.
- The answer must be fully supported by the supplied excerpt only.
- Prefer a concrete factual question over a vague or opinion-based one.
- Preserve important numbers, limits, dates, percentages, conditions, and exceptions.
- Do not ask about the document itself, its filename, page number, or wording.
- Do not use outside knowledge.
- Avoid questions whose answer would require information not present in the excerpt.
- Keep the ground-truth answer concise but complete enough to judge another answer.
""".strip()


def _sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]


def sample_excerpt(
    document: SourceDocument,
    *,
    rng: random.Random,
    max_chars: int = 6000,
    min_chars: int = 1200,
) -> str:
    """Sample one contiguous sentence window from a source document."""

    text = re.sub(r"\s+", " ", document.text).strip()
    if len(text) <= max_chars:
        return text

    sentences = _sentences(text)
    if not sentences:
        start = rng.randint(0, max(0, len(text) - max_chars))
        return text[start : start + max_chars].strip()

    candidate_starts = list(range(len(sentences)))
    rng.shuffle(candidate_starts)

    best = ""
    for start in candidate_starts:
        selected: list[str] = []
        length = 0
        for sentence in sentences[start:]:
            extra = len(sentence) + (1 if selected else 0)
            if selected and length + extra > max_chars:
                break
            selected.append(sentence)
            length += extra
            if length >= min_chars:
                break
        excerpt = " ".join(selected).strip()
        if len(excerpt) > len(best):
            best = excerpt
        if len(excerpt) >= min_chars:
            return excerpt

    return best or text[:max_chars].strip()


def _make_item_id(document_id: str, question: str) -> str:
    digest = hashlib.sha1(f"{document_id}:{question}".encode("utf-8")).hexdigest()[:12]
    return f"qa-{digest}"


def generate_dataset(
    *,
    documents_path: Path,
    output_path: Path,
    api_key: str,
    model: str,
    count: int,
    seed: int,
    excerpt_chars: int = 6000,
) -> list[EvaluationItem]:
    all_documents = load_documents(documents_path)
    documents = sorted(
        (doc for doc in all_documents if len(doc.text.strip()) >= 500),
        key=lambda doc: doc.id,
    )
    if not documents:
        raise ValueError("No sufficiently long source documents were found.")
    if count < 1:
        raise ValueError("count must be at least 1")
    if count > len(documents):
        raise ValueError(
            f"Requested {count} QA pairs but only {len(documents)} eligible documents exist. "
            "Use a smaller count so each QA pair comes from a different source document."
        )

    rng = random.Random(seed)
    selected = rng.sample(documents, count)
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    items: list[EvaluationItem] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for number, document in enumerate(selected, start=1):
        excerpt = sample_excerpt(document, rng=rng, max_chars=excerpt_chars)
        response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"SOURCE TITLE: {document.title}\n"
                        f"SOURCE URL: {document.url}\n\n"
                        f"SOURCE EXCERPT:\n{excerpt}"
                    ),
                },
            ],
            text_format=GeneratedQA,
        )
        generated = response.output_parsed
        if generated is None:
            raise RuntimeError(f"The model did not return a structured QA pair for {document.title!r}.")

        item = EvaluationItem(
            id=_make_item_id(document.id, generated.question),
            source_document_id=document.id,
            source_title=document.title,
            source_url=document.url,
            source_type=document.source_type,
            question=generated.question.strip(),
            ground_truth_answer=generated.answer.strip(),
            source_excerpt=excerpt,
            generated_at=datetime.now(timezone.utc),
            generation_model=model,
        )
        items.append(item)
        print(f"Generated {number}/{count}: {item.question}")

    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(item.model_dump_json() + "\n")

    manifest = {
        "count": len(items),
        "seed": seed,
        "model": model,
        "excerpt_chars": excerpt_chars,
        "documents_path": str(documents_path),
        "documents_file_sha256": sha256_file(documents_path),
        "corpus_content_fingerprint": corpus_fingerprint(all_documents),
        "output_path": str(output_path),
        "dataset_sha256": sha256_file(output_path),
        "source_document_ids": [item.source_document_id for item in items],
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return items


def load_dataset(path: Path) -> list[EvaluationItem]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    items: list[EvaluationItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                items.append(EvaluationItem.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"Invalid evaluation item at {path}:{line_number}") from exc
    return items
