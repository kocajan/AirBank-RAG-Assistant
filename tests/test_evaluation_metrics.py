from evaluation.metrics import summarize_retrieval
from evaluation.models import QueryRetrievalResult


def _result(rank: int | None) -> QueryRetrievalResult:
    return QueryRetrievalResult(
        evaluation_item_id=f"q-{rank}",
        question="Test?",
        correct_document_id="correct",
        retrieved_document_ids=[],
        retrieved_titles=[],
        rank=rank,
    )


def test_retrieval_metrics_with_single_relevant_source() -> None:
    summary = summarize_retrieval([_result(1), _result(3), _result(None), _result(5)])

    assert summary["hit_at_1"] == 0.25
    assert summary["hit_at_3"] == 0.5
    assert summary["hit_at_5"] == 0.75
    assert 0.0 < summary["mrr_at_5"] < 1.0
    assert 0.0 < summary["ndcg_at_5"] < 1.0
    assert summary["found_at_10"] == 0.75
