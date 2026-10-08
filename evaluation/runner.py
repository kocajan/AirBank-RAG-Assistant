from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from openai import OpenAI

from backend.agent import AgentDependencies, get_agent
from evaluation.config import GRID, EvaluationConfig
from evaluation.dataset import load_dataset
from evaluation.judge import judge_answer
from evaluation.metrics import summarize_retrieval
from evaluation.models import (
    AnswerEvaluationResult,
    ConfigSummary,
    EvaluationItem,
    QueryRetrievalResult,
)
from rag.embeddings import create_openai_embedder
from rag.index import LocalVectorIndex, build_index, index_is_ready


async def ensure_eval_index(
    *,
    config: EvaluationConfig,
    documents_path: Path,
    index_root: Path,
    api_key: str,
    batch_size: int,
    force: bool,
) -> Path:
    output = index_root / config.slug
    if index_is_ready(output) and not force:
        print(f"[index] reuse {config.slug}")
        return output

    print(
        f"[index] build {config.slug}: model={config.embedding_model}, "
        f"chunk={config.chunking.chunk_size}/{config.chunking.chunk_overlap}"
    )
    embedder = create_openai_embedder(api_key=api_key, model_name=config.embedding_model)
    await build_index(
        documents_path=documents_path,
        index_dir=output,
        embedder=embedder,
        embedding_model_name=config.embedding_model,
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        batch_size=batch_size,
    )
    return output


def _rank_correct_source(item: EvaluationItem, hits) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if hit.chunk.document_id == item.source_document_id:
            return rank
    return None


async def _retrieval_results_for_model(
    *,
    model_name: str,
    configs: list[EvaluationConfig],
    dataset: list[EvaluationItem],
    index_root: Path,
    api_key: str,
    top_k: int,
) -> dict[str, list[QueryRetrievalResult]]:
    """Embed each evaluation question once per model and reuse it across chunk grids."""

    embedder = create_openai_embedder(api_key=api_key, model_name=model_name)
    question_result = await embedder.embed_documents([item.question for item in dataset])
    query_vectors = question_result.embeddings

    by_config: dict[str, list[QueryRetrievalResult]] = {}
    for config in configs:
        index = LocalVectorIndex(
            index_dir=index_root / config.slug,
            embedder=embedder,
            top_k=top_k,
            min_score=-1.0,
            max_chunks_per_document=1,
        )
        results: list[QueryRetrievalResult] = []
        for item, vector in zip(dataset, query_vectors, strict=True):
            hits = index.search_by_vector(vector, top_k=top_k)
            results.append(
                QueryRetrievalResult(
                    evaluation_item_id=item.id,
                    question=item.question,
                    correct_document_id=item.source_document_id,
                    retrieved_document_ids=[hit.chunk.document_id for hit in hits],
                    retrieved_titles=[hit.chunk.title for hit in hits],
                    rank=_rank_correct_source(item, hits),
                )
            )
        by_config[config.slug] = results
    return by_config


def _write_jsonl(path: Path, models: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for model in models:
            handle.write(model.model_dump_json() + "\n")


async def _judge_config_answers(
    *,
    config: EvaluationConfig,
    dataset: list[EvaluationItem],
    index_root: Path,
    api_key: str,
    judge_model: str,
    top_k: int,
) -> list[AnswerEvaluationResult]:
    embedder = create_openai_embedder(api_key=api_key, model_name=config.embedding_model)
    retriever = LocalVectorIndex(
        index_dir=index_root / config.slug,
        embedder=embedder,
        top_k=top_k,
        min_score=-1.0,
        max_chunks_per_document=2,
    )
    agent = get_agent()
    judge_client = OpenAI(api_key=api_key)
    results: list[AnswerEvaluationResult] = []

    for number, item in enumerate(dataset, start=1):
        deps = AgentDependencies(retriever=retriever)
        response = await agent.run(item.question, deps=deps, message_history=[])
        answer = str(response.output).strip()
        judged = judge_answer(
            client=judge_client,
            model=judge_model,
            question=item.question,
            ground_truth=item.ground_truth_answer,
            chatbot_answer=answer,
        )
        results.append(
            AnswerEvaluationResult(
                evaluation_item_id=item.id,
                question=item.question,
                ground_truth_answer=item.ground_truth_answer,
                chatbot_answer=answer,
                judge_score=judged.score,
                judge_reason=judged.reason,
            )
        )
        print(
            f"[judge] {config.slug} {number}/{len(dataset)} "
            f"score={judged.score}/5"
        )
    return results


async def run_grid_evaluation(
    *,
    dataset_path: Path,
    documents_path: Path,
    index_root: Path,
    results_dir: Path,
    api_key: str,
    judge_model: str,
    with_judge: bool,
    top_k: int = 10,
    batch_size: int = 64,
    force_rebuild: bool = False,
    limit: int | None = None,
) -> list[ConfigSummary]:
    dataset = load_dataset(dataset_path)
    if limit is not None:
        dataset = dataset[:limit]
    if not dataset:
        raise ValueError("Evaluation dataset is empty.")

    results_dir.mkdir(parents=True, exist_ok=True)
    index_root.mkdir(parents=True, exist_ok=True)

    for config in GRID:
        await ensure_eval_index(
            config=config,
            documents_path=documents_path,
            index_root=index_root,
            api_key=api_key,
            batch_size=batch_size,
            force=force_rebuild,
        )

    retrieval_by_config: dict[str, list[QueryRetrievalResult]] = {}
    for model_name in dict.fromkeys(config.embedding_model for config in GRID):
        configs = [config for config in GRID if config.embedding_model == model_name]
        print(f"[retrieval] embedding {len(dataset)} questions with {model_name}")
        retrieval_by_config.update(
            await _retrieval_results_for_model(
                model_name=model_name,
                configs=configs,
                dataset=dataset,
                index_root=index_root,
                api_key=api_key,
                top_k=top_k,
            )
        )

    summaries: list[ConfigSummary] = []
    for config in GRID:
        retrieval_results = retrieval_by_config[config.slug]
        _write_jsonl(results_dir / f"{config.slug}__retrieval.jsonl", retrieval_results)
        retrieval_metrics = summarize_retrieval(retrieval_results)

        answer_results: list[AnswerEvaluationResult] = []
        if with_judge:
            answer_results = await _judge_config_answers(
                config=config,
                dataset=dataset,
                index_root=index_root,
                api_key=api_key,
                judge_model=judge_model,
                top_k=5,
            )
            _write_jsonl(results_dir / f"{config.slug}__answers.jsonl", answer_results)

        judge_mean = mean(result.judge_score for result in answer_results) if answer_results else None
        high_rate = (
            mean(1.0 if result.judge_score >= 4 else 0.0 for result in answer_results)
            if answer_results
            else None
        )
        summaries.append(
            ConfigSummary(
                config_slug=config.slug,
                embedding_model=config.embedding_model,
                chunking_name=config.chunking.name,
                chunk_size=config.chunking.chunk_size,
                chunk_overlap=config.chunking.chunk_overlap,
                queries=len(dataset),
                mean_judge_score=judge_mean,
                judge_score_4_or_5_rate=high_rate,
                **retrieval_metrics,
            )
        )

    (results_dir / "latest_details.json").write_text(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "queries": len(dataset),
                "with_judge": with_judge,
                "judge_model": judge_model if with_judge else None,
                "configs": [summary.model_dump(mode="json") for summary in summaries],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summaries
