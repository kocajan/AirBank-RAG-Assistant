from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl


class GeneratedQA(BaseModel):
    question: str = Field(min_length=5)
    answer: str = Field(min_length=1)


class EvaluationItem(BaseModel):
    id: str
    source_document_id: str
    source_title: str
    source_url: HttpUrl
    source_type: str
    question: str
    ground_truth_answer: str
    source_excerpt: str
    generated_at: datetime
    generation_model: str


class JudgeOutput(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str = Field(min_length=1)


class QueryRetrievalResult(BaseModel):
    evaluation_item_id: str
    question: str
    correct_document_id: str
    retrieved_document_ids: list[str]
    retrieved_titles: list[str]
    rank: int | None


class AnswerEvaluationResult(BaseModel):
    evaluation_item_id: str
    question: str
    ground_truth_answer: str
    chatbot_answer: str
    judge_score: int
    judge_reason: str


class ConfigSummary(BaseModel):
    config_slug: str
    embedding_model: str
    chunking_name: str
    chunk_size: int
    chunk_overlap: int
    queries: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr_at_5: float
    ndcg_at_5: float
    found_at_10: float
    mean_rank_when_found: float | None
    mean_judge_score: float | None = None
    judge_score_4_or_5_rate: float | None = None
