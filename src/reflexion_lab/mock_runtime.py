from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

load_dotenv()

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}


@dataclass
class CallMetrics:
    token_estimate: int
    latency_ms: int


def use_mock_mode() -> bool:
    if os.getenv("MOCK_MODE", "").lower() in {"1", "true", "yes"}:
        return True
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key.startswith("sk-your-"):
        return True
    return False


def _format_context(example: QAExample) -> str:
    return "\n\n".join(f"[{chunk.title}]\n{chunk.text}" for chunk in example.context)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def _chat(system: str, user: str) -> tuple[str, CallMetrics]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60.0, max_retries=0)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    max_attempts = 3
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            usage = response.usage
            tokens = usage.total_tokens if usage else 0
            content = response.choices[0].message.content or ""
            return content, CallMetrics(token_estimate=tokens, latency_ms=latency_ms)
        except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    raise last_error  # type: ignore[misc]


def _mock_actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if example.qid not in FIRST_ATTEMPT_WRONG:
        return example.gold_answer
    if agent_type == "react":
        return FIRST_ATTEMPT_WRONG[example.qid]
    if attempt_id == 1 and not reflection_memory:
        return FIRST_ATTEMPT_WRONG[example.qid]
    return example.gold_answer


def _mock_evaluator(example: QAExample, answer: str) -> JudgeResult:
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
    if normalize_answer(answer) == "london":
        return JudgeResult(
            score=0,
            reason="The answer stopped at the birthplace city and never completed the second hop to the river.",
            missing_evidence=["Need to identify the river that flows through London."],
            spurious_claims=[],
        )
    return JudgeResult(
        score=0,
        reason="The final answer selected the wrong second-hop entity.",
        missing_evidence=["Need to ground the answer in the second paragraph."],
        spurious_claims=[answer],
    )


def _mock_reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    strategy = (
        "Do the second hop explicitly: birthplace city -> river through that city."
        if example.qid == "hp2"
        else "Verify the final entity against the second paragraph before answering."
    )
    return ReflectionEntry(
        attempt_id=attempt_id,
        failure_reason=judge.reason,
        lesson="A partial first-hop answer is not enough; the final answer must complete all hops.",
        next_strategy=strategy,
    )


def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> tuple[str, CallMetrics]:
    if use_mock_mode():
        answer = _mock_actor_answer(example, attempt_id, agent_type, reflection_memory)
        tokens = 120 + (len(reflection_memory) * 40)
        return answer, CallMetrics(token_estimate=tokens, latency_ms=10)

    reflection_block = ""
    if reflection_memory:
        reflection_block = "Prior reflections:\n" + "\n".join(f"- {item}" for item in reflection_memory)

    user_prompt = f"""Question: {example.question}

Context:
{_format_context(example)}

Attempt: {attempt_id}
{reflection_block}

Return only the final answer."""

    content, metrics = _chat(ACTOR_SYSTEM, user_prompt)
    return content.strip(), metrics


def evaluator(example: QAExample, answer: str) -> tuple[JudgeResult, CallMetrics]:
    if use_mock_mode():
        result = _mock_evaluator(example, answer)
        return result, CallMetrics(token_estimate=80, latency_ms=5)

    user_prompt = f"""Question: {example.question}
Gold answer: {example.gold_answer}
Predicted answer: {answer}

Context:
{_format_context(example)}"""

    content, metrics = _chat(EVALUATOR_SYSTEM, user_prompt)
    parsed = _extract_json(content)
    result = JudgeResult.model_validate(parsed)
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        result = JudgeResult(
            score=1,
            reason="Final answer matches the gold answer after normalization.",
            missing_evidence=[],
            spurious_claims=[],
        )
    return result, metrics


def reflector(
    example: QAExample,
    attempt_id: int,
    wrong_answer: str,
    judge: JudgeResult,
) -> tuple[ReflectionEntry, CallMetrics]:
    if use_mock_mode():
        result = _mock_reflector(example, attempt_id, judge)
        return result, CallMetrics(token_estimate=90, latency_ms=5)

    user_prompt = f"""Question: {example.question}
Attempt id: {attempt_id}
Wrong answer: {wrong_answer}

Evaluator feedback:
- reason: {judge.reason}
- missing_evidence: {judge.missing_evidence}
- spurious_claims: {judge.spurious_claims}

Context:
{_format_context(example)}"""

    content, metrics = _chat(REFLECTOR_SYSTEM, user_prompt)
    parsed = _extract_json(content)
    parsed["attempt_id"] = attempt_id
    result = ReflectionEntry.model_validate(parsed)
    return result, metrics
