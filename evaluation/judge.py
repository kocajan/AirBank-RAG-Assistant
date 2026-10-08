from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

from evaluation.models import JudgeOutput


JUDGE_SYSTEM_PROMPT = """
You are an impartial evaluator of a banking-information RAG assistant.
Judge the chatbot answer only against the user question and the supplied ground-truth answer.
Do not require identical wording. Focus on factual correctness and whether important conditions from the ground truth are preserved.

Score rubric:
5 = fully correct and complete; no material factual error
4 = essentially correct; only a minor omission or harmless imprecision
3 = partially correct; important information is missing or somewhat inaccurate
2 = mostly incorrect; contains only a small amount of correct information
1 = incorrect, contradictory, irrelevant, or refuses an answer that the ground truth clearly supports

Return a short reason for the score.
""".strip()


def judge_answer(
    *,
    client: "OpenAI",
    model: str,
    question: str,
    ground_truth: str,
    chatbot_answer: str,
) -> JudgeOutput:
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\n"
                    f"GROUND TRUTH:\n{ground_truth}\n\n"
                    f"CHATBOT ANSWER:\n{chatbot_answer}"
                ),
            },
        ],
        text_format=JudgeOutput,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("Judge model did not return a structured score.")
    return parsed
