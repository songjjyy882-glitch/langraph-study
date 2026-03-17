"""
MLB 야구 에이전트 평가 실험.

Opik evaluate()를 사용하여 에이전트의 도구 호출 정확도와 응답 품질을 평가합니다.

사용법:
    cd agent && uv run python -m app.evaluation.experiment
"""

import json
import os
import uuid
from typing import Any, Dict, List

from app.core.config import settings

# opik import 전에 환경변수 세팅 (API key 프롬프트 방지)
if settings.OPIK and settings.OPIK.URL_OVERRIDE:
    os.environ.setdefault("OPIK_URL_OVERRIDE", settings.OPIK.URL_OVERRIDE)
if settings.OPIK and settings.OPIK.PROJECT:
    os.environ.setdefault("OPIK_PROJECT_NAME", settings.OPIK.PROJECT)

import opik  # noqa: E402
from opik.evaluation.metrics import BaseMetric
from opik.evaluation.metrics.score_result import ScoreResult

from app.agents.agent import get_agent
from app.evaluation.dataset import create_dataset


# ──────────────────────────────────────────
# 커스텀 메트릭 정의
# ──────────────────────────────────────────


class ToolSelectionAccuracy(BaseMetric):
    """에이전트가 올바른 도구를 호출했는지 평가합니다."""

    name = "tool_selection_accuracy"

    def score(self, output: str, expected_tool: str = None, **kwargs) -> ScoreResult:
        if expected_tool is None:
            # 도구 호출이 불필요한 질문 — 도구를 호출하지 않았으면 정답
            tool_called = "tool_calls" in output
            return ScoreResult(
                name=self.name,
                value=1.0 if not tool_called else 0.0,
                reason="도구 호출 불필요 — " + ("호출 없음 ✓" if not tool_called else "불필요한 도구 호출 ✗"),
            )

        if expected_tool in output:
            return ScoreResult(name=self.name, value=1.0, reason=f"올바른 도구 호출: {expected_tool}")
        return ScoreResult(name=self.name, value=0.0, reason=f"기대 도구: {expected_tool}, 실제 응답에서 미발견")


class KeywordCoverage(BaseMetric):
    """응답에 기대 키워드가 포함되었는지 평가합니다."""

    name = "keyword_coverage"

    def score(self, output: str, expected_keywords: List[str] = None, **kwargs) -> ScoreResult:
        if not expected_keywords:
            return ScoreResult(name=self.name, value=1.0, reason="키워드 체크 없음")

        output_lower = output.lower()
        found = [kw for kw in expected_keywords if kw.lower() in output_lower]
        ratio = len(found) / len(expected_keywords)
        missing = [kw for kw in expected_keywords if kw.lower() not in output_lower]
        reason = f"발견: {found}" + (f", 누락: {missing}" if missing else "")
        return ScoreResult(name=self.name, value=round(ratio, 2), reason=reason)


class ResponseCompleteness(BaseMetric):
    """응답이 충분한 길이와 구조를 갖추었는지 평가합니다."""

    name = "response_completeness"

    def score(self, output: str, **kwargs) -> ScoreResult:
        # 최소 길이 체크
        if len(output) < 50:
            return ScoreResult(name=self.name, value=0.2, reason="응답이 너무 짧음")

        score = 0.0
        reasons = []

        # 길이 점수 (최대 0.4)
        if len(output) >= 200:
            score += 0.4
            reasons.append("충분한 길이")
        elif len(output) >= 100:
            score += 0.2
            reasons.append("적정 길이")

        # 숫자 데이터 포함 여부 (0.3)
        import re
        numbers = re.findall(r"\d+\.?\d*", output)
        if len(numbers) >= 3:
            score += 0.3
            reasons.append("데이터 포함")

        # 구조화 여부 — 줄바꿈이나 구분자 (0.3)
        if output.count("\n") >= 2 or ":" in output:
            score += 0.3
            reasons.append("구조화된 응답")

        return ScoreResult(name=self.name, value=round(score, 2), reason=", ".join(reasons))


# ──────────────────────────────────────────
# 에이전트 실행 태스크
# ──────────────────────────────────────────


def agent_task(item: Dict[str, Any]) -> Dict[str, Any]:
    """데이터셋 항목을 에이전트에 실행하고 결과를 반환합니다."""
    agent = get_agent()
    thread_id = str(uuid.uuid4())

    full_output = []

    # 동기 실행 (evaluate는 동기 함수 기대)
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": item["input"]}]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="updates",
    ):
        for step, event in chunk.items():
            if not event:
                continue
            # 도구 호출 기록
            messages = event.get("messages", [])
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        full_output.append(f"[tool_calls: {tc['name']}]")
                if hasattr(msg, "content") and msg.content:
                    content = msg.content
                    if isinstance(content, str):
                        full_output.append(content)

            # structured_response
            structured = event.get("structured_response")
            if structured:
                full_output.append(getattr(structured, "content", ""))

    output = "\n".join(full_output)
    return {"output": output}


# ──────────────────────────────────────────
# 실험 실행
# ──────────────────────────────────────────


def run_experiment():
    """Opik experiment를 실행합니다."""
    dataset = create_dataset()

    result = opik.evaluate(
        dataset=dataset,
        task=agent_task,
        scoring_metrics=[
            ToolSelectionAccuracy(),
            KeywordCoverage(),
            ResponseCompleteness(),
        ],
        experiment_name_prefix="mlb-agent-eval",
        project_name="jysong-project",
        experiment_config={
            "model": "gpt-4.1",
            "agent_type": "langgraph-react",
            "tools": ["search_player_stats", "compare_players", "search_game_results", "search_statcast"],
        },
        task_threads=4,
    )

    print("\n========== 실험 결과 ==========")
    print(f"실험 이름: {result.experiment_name}")
    print(f"실험 URL: {result.experiment_url}")

    return result


if __name__ == "__main__":
    run_experiment()
