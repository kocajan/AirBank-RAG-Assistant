from __future__ import annotations

import json
from pathlib import Path

from evaluation.models import ConfigSummary


def _pct(value: float) -> str:
    return f"{100 * value:5.1f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "  -  " if value is None else f"{value:.{digits}f}"


def render_report(summaries: list[ConfigSummary], *, with_judge: bool) -> str:
    lines = [
        "# Air Bank RAG evaluation",
        "",
        "Retrieval relevance is evaluated at the source-document level. Because every generated question has exactly one known relevant source document, Hit@K is equivalent to Recall@K in this experiment.",
        "",
    ]

    if with_judge:
        header = (
            f"{'Embedding model':<27} {'Chunking':<8} {'size/ovl':<10} "
            f"{'R@1':>7} {'R@3':>7} {'R@5':>7} {'MRR@5':>8} {'nDCG@5':>8} "
            f"{'Judge':>7} {'>=4':>7}"
        )
    else:
        header = (
            f"{'Embedding model':<27} {'Chunking':<8} {'size/ovl':<10} "
            f"{'R@1':>7} {'R@3':>7} {'R@5':>7} {'MRR@5':>8} {'nDCG@5':>8} {'R@10':>7}"
        )

    lines.extend(["```text", header, "-" * len(header)])
    for summary in summaries:
        base = (
            f"{summary.embedding_model:<27} {summary.chunking_name:<8} "
            f"{summary.chunk_size:>4}/{summary.chunk_overlap:<4} "
            f"{_pct(summary.hit_at_1):>7} {_pct(summary.hit_at_3):>7} {_pct(summary.hit_at_5):>7} "
            f"{summary.mrr_at_5:>8.3f} {summary.ndcg_at_5:>8.3f}"
        )
        if with_judge:
            judge = _num(summary.mean_judge_score, 2)
            high = _pct(summary.judge_score_4_or_5_rate or 0.0)
            lines.append(f"{base} {judge:>7} {high:>7}")
        else:
            lines.append(f"{base} {_pct(summary.found_at_10):>7}")
    lines.append("```")

    if summaries:
        best_retrieval = max(summaries, key=lambda s: (s.mrr_at_5, s.hit_at_5, s.hit_at_1))
        lines.extend(
            [
                "",
                f"Best retrieval by MRR@5: **{best_retrieval.config_slug}** "
                f"(MRR@5={best_retrieval.mrr_at_5:.3f}, R@5={_pct(best_retrieval.hit_at_5).strip()}).",
            ]
        )
        if with_judge:
            judged = [s for s in summaries if s.mean_judge_score is not None]
            if judged:
                best_judge = max(judged, key=lambda s: (s.mean_judge_score or 0.0, s.mrr_at_5))
                lines.append(
                    f"Best end-to-end mean judge score: **{best_judge.config_slug}** "
                    f"({best_judge.mean_judge_score:.2f}/5)."
                )

    lines.extend(
        [
            "",
            "Metric notes:",
            "- R@1/R@3/R@5: fraction of questions whose known source document appears in the top K retrieved documents.",
            "- MRR@5: rewards putting the correct source as high as possible in the first five results.",
            "- nDCG@5: rank-sensitive retrieval score with one relevant source per question.",
        ]
    )
    if with_judge:
        lines.append("- Judge: mean LLM score from 1–5; >=4 is the fraction of answers judged essentially or fully correct.")
    return "\n".join(lines) + "\n"


def save_report(
    *,
    summaries: list[ConfigSummary],
    output_dir: Path,
    with_judge: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = render_report(summaries, with_judge=with_judge)
    path = output_dir / "latest_report.md"
    path.write_text(report, encoding="utf-8")
    (output_dir / "latest_summary.json").write_text(
        json.dumps([summary.model_dump(mode="json") for summary in summaries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
