from __future__ import annotations

import math
from statistics import mean

from evaluation.models import QueryRetrievalResult


def _hit_at(results: list[QueryRetrievalResult], k: int) -> float:
    if not results:
        return 0.0
    hits = sum(1 for result in results if result.rank is not None and result.rank <= k)
    return hits / len(results)


def mrr_at(results: list[QueryRetrievalResult], k: int) -> float:
    if not results:
        return 0.0
    values = [
        1.0 / result.rank if result.rank is not None and result.rank <= k else 0.0
        for result in results
    ]
    return mean(values)


def ndcg_at(results: list[QueryRetrievalResult], k: int) -> float:
    """nDCG@k for one binary-relevant source document per query."""
    if not results:
        return 0.0
    values = [
        1.0 / math.log2(result.rank + 1)
        if result.rank is not None and result.rank <= k
        else 0.0
        for result in results
    ]
    return mean(values)


def summarize_retrieval(results: list[QueryRetrievalResult]) -> dict[str, float | None]:
    found_ranks = [result.rank for result in results if result.rank is not None]
    return {
        "hit_at_1": _hit_at(results, 1),
        "hit_at_3": _hit_at(results, 3),
        "hit_at_5": _hit_at(results, 5),
        "mrr_at_5": mrr_at(results, 5),
        "ndcg_at_5": ndcg_at(results, 5),
        "found_at_10": _hit_at(results, 10),
        "mean_rank_when_found": mean(found_ranks) if found_ranks else None,
    }
