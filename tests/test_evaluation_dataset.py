import random
from datetime import datetime, timezone

from data_collection.models import SourceDocument
from evaluation.dataset import sample_excerpt


def test_sample_excerpt_is_bounded_and_deterministic() -> None:
    text = " ".join(f"Sentence {i} contains useful factual information." for i in range(500))
    document = SourceDocument(
        id="doc",
        title="Title",
        url="https://www.airbank.cz/test/",
        source_type="html",
        category="Test",
        fetched_at=datetime.now(timezone.utc),
        text=text,
        content_sha256="abc",
        raw_path="data/raw/html/test.html",
    )

    first = sample_excerpt(document, rng=random.Random(42), max_chars=2000, min_chars=700)
    second = sample_excerpt(document, rng=random.Random(42), max_chars=2000, min_chars=700)

    assert first == second
    assert 700 <= len(first) <= 2000
