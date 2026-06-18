from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .mock_runtime import FAILURE_MODE_BY_QID, actor_answer, evaluator, reflector
from .schemas import AttemptTrace, JudgeResult, QAExample, ReflectionEntry, RunRecord


def _infer_failure_mode(example: QAExample, judge: JudgeResult) -> str:
    if judge.score == 1:
        return "none"
    if example.qid in FAILURE_MODE_BY_QID:
        return FAILURE_MODE_BY_QID[example.qid]
    reason = judge.reason.lower()
    if "hop" in reason or "incomplete" in reason or "partial" in reason:
        return "incomplete_multi_hop"
    if "drift" in reason or "wrong entity" in reason or "second-hop" in reason:
        return "entity_drift"
    if "loop" in reason:
        return "looping"
    if "reflection" in reason or "overfit" in reason:
        return "reflection_overfit"
    return "wrong_final_answer"


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        final_judge: JudgeResult | None = None

        for attempt_id in range(1, self.max_attempts + 1):
            answer, actor_metrics = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            judge, eval_metrics = evaluator(example, answer)
            token_estimate = actor_metrics.token_estimate + eval_metrics.token_estimate
            latency_ms = actor_metrics.latency_ms + eval_metrics.latency_ms

            final_answer = answer
            final_score = judge.score
            final_judge = judge

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                token_estimate=token_estimate,
                latency_ms=latency_ms,
            )

            if judge.score == 1:
                traces.append(trace)
                break

            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection, refl_metrics = reflector(example, attempt_id, answer, judge)
                reflections.append(reflection)
                reflection_memory.append(
                    f"Attempt {attempt_id} lesson: {reflection.lesson} | Next strategy: {reflection.next_strategy}"
                )
                trace.reflection = reflection
                trace.token_estimate += refl_metrics.token_estimate
                trace.latency_ms += refl_metrics.latency_ms

            traces.append(trace)

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = _infer_failure_mode(example, final_judge) if final_judge else "wrong_final_answer"

        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
