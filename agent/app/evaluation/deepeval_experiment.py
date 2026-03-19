"""
DeepEval G-Eval 기반 MLB 야구 에이전트 평가 실험.

LLM-as-a-judge 방식으로 응답 품질을 정밀하게 평가합니다.

사용법:
    cd agent && uv run python -m app.evaluation.deepeval_experiment
"""

import os
import uuid
from typing import Any, Dict

from app.core.config import settings

# Opik 환경변수 세팅
if settings.OPIK and settings.OPIK.URL_OVERRIDE:
    os.environ.setdefault("OPIK_URL_OVERRIDE", settings.OPIK.URL_OVERRIDE)
if settings.OPIK and settings.OPIK.PROJECT:
    os.environ.setdefault("OPIK_PROJECT_NAME", settings.OPIK.PROJECT)

import opik  # noqa: E402
from deepeval.metrics import GEval  # noqa: E402
from deepeval.test_case import LLMTestCase, LLMTestCaseParams  # noqa: E402
from deepeval import evaluate  # noqa: E402

from app.agents.agent import get_agent  # noqa: E402
from app.core.opik_tracer import opik_tracer  # noqa: E402


# ──────────────────────────────────────────
# G-Eval 메트릭 정의
# ──────────────────────────────────────────

# 1. 도구 선택 정확도 — 질문에 맞는 올바른 도구를 호출했는지
tool_selection_metric = GEval(
    name="ToolSelectionAccuracy",
    evaluation_steps=[
        "사용자의 질문이 특정 선수 스탯, 선수 비교, 팀 경기 결과, Statcast 데이터 중 어디에 해당하는지 판단한다",
        "에이전트의 응답에서 올바른 도구(search_player_stats, compare_players, search_game_results, search_statcast)가 호출되었는지 확인한다",
        "일반 야구 지식 질문(WAR 설명 등)에는 도구를 호출하지 않아야 한다",
        "올바른 도구가 호출되었으면 높은 점수, 잘못된 도구이거나 불필요한 호출이면 낮은 점수를 준다",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.7,
)

# 2. 응답 정확성 — 야구 데이터와 분석이 정확한지
answer_accuracy_metric = GEval(
    name="AnswerAccuracy",
    evaluation_steps=[
        "응답에 포함된 야구 통계 수치(AVG, HR, ERA 등)가 합리적인 범위인지 확인한다",
        "선수 이름, 팀 이름 등 기본 정보가 정확한지 확인한다",
        "질문에서 요청한 내용에 대해 빠짐없이 답변했는지 확인한다",
        "잘못된 정보나 존재하지 않는 데이터를 포함하면 감점한다",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.6,
)

# 3. 분석 품질 — 단순 수치 나열이 아닌 인사이트를 제공하는지
analysis_quality_metric = GEval(
    name="AnalysisQuality",
    evaluation_steps=[
        "단순히 숫자만 나열했는지, 아니면 의미 있는 분석을 제공했는지 평가한다",
        "비교 질문에는 어떤 선수가 어떤 면에서 우위인지 명확한 비교가 있어야 한다",
        "Statcast 질문에는 구종별 특성, 강점/약점 분석이 포함되어야 한다",
        "야구 용어 설명 시 예시나 기준값을 함께 제공하면 가점한다",
        "응답이 한글로 자연스럽고 읽기 쉬운지 확인한다",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.5,
)

# 4. 응답 형식 준수 — 프롬프트에서 지시한 형식을 따르는지
format_compliance_metric = GEval(
    name="FormatCompliance",
    evaluation_steps=[
        "선수 스탯 응답에 주요 지표명(AVG, OBP, SLG, OPS, HR, WAR, ERA, WHIP 등)이 명시되어 있는지 확인한다",
        "팀 관련 응답에 팀 약어(LAD, NYY 등)가 포함되어 있는지 확인한다",
        "Statcast 응답에 구속, 회전수, 타구속도 등 핵심 지표가 포함되어 있는지 확인한다",
        "야구 용어 설명 시 영문 정식명칭과 한글 설명이 함께 제공되는지 확인한다",
        "응답이 구조화되어 있는지(표, 항목별 정리 등) 확인한다",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.5,
)


# ──────────────────────────────────────────
# 테스트 데이터셋
# ──────────────────────────────────────────

EVAL_QUESTIONS = [
    "오타니 쇼헤이의 2024시즌 타격 성적 알려줘",
    "2024시즌 류현진 투수 성적을 조회해줘",
    "Aaron Judge 2024 stats",
    "오타니와 저지의 2024시즌 성적 비교해줘",
    "무키 벳츠와 후안 소토 비교 분석",
    "다저스 2024시즌 최근 경기 결과 보여줘",
    "양키스 시즌 성적 알려줘",
    "오타니의 Statcast 데이터 분석해줘",
    "게릿 콜의 구종별 구속과 회전수 분석",
    "WAR이 뭐야?",
    "OPS는 어떻게 계산해?",
    "2024 MLB 시즌 홈런왕은 누구야?",
    "현 시점에서 MLB 최고의 투수와 타자는 각각 누구야?"
]


# ──────────────────────────────────────────
# 에이전트 실행
# ──────────────────────────────────────────


def run_agent(question: str) -> str:
    """에이전트에 질문하고 전체 출력을 반환합니다."""
    agent = get_agent()
    thread_id = str(uuid.uuid4())
    full_output = []

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config={
            "configurable": {"thread_id": thread_id},
            "callbacks": [opik_tracer],
        },
        stream_mode="updates",
    ):
        for step, event in chunk.items():
            if not event:
                continue
            messages = event.get("messages", [])
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        full_output.append(f"[tool_calls: {tc['name']}]")
                if hasattr(msg, "content") and msg.content:
                    if isinstance(msg.content, str):
                        full_output.append(msg.content)
            structured = event.get("structured_response")
            if structured:
                full_output.append(getattr(structured, "content", ""))

    return "\n".join(full_output)


# ──────────────────────────────────────────
# 실험 실행
# ──────────────────────────────────────────


def run_experiment():
    """DeepEval G-Eval 실험을 실행하고 결과를 Opik에도 기록합니다."""
    print("에이전트 실행 중...")

    test_cases = []
    agent_outputs = []
    for i, question in enumerate(EVAL_QUESTIONS):
        print(f"  [{i+1}/{len(EVAL_QUESTIONS)}] {question}")
        output = run_agent(question)
        agent_outputs.append(output)
        test_cases.append(
            LLMTestCase(
                input=question,
                actual_output=output,
            )
        )

    print(f"\n{len(test_cases)}개 테스트 케이스 생성 완료. G-Eval 평가 시작...\n")

    metrics = [
        tool_selection_metric,
        answer_accuracy_metric,
        analysis_quality_metric,
        format_compliance_metric,
    ]

    # DeepEval 평가 실행 — 각 메트릭을 개별 measure()로 실행하여 점수를 저장
    print("G-Eval 채점 중...")
    # score_map[i][metric_name] = (score, reason)
    score_map: Dict[int, Dict[str, tuple]] = {}
    for i, tc in enumerate(test_cases):
        score_map[i] = {}
        for metric in metrics:
            metric.measure(tc)
            score_map[i][metric.name] = (metric.score, metric.reason)
            print(f"  [{i+1}] {metric.name}: {metric.score:.2f}")

    # DeepEval evaluate()도 실행 (터미널 리포트용)
    results = evaluate(
        test_cases=test_cases,
        metrics=metrics,
    )

    # ── Opik에 G-Eval 결과 기록 ──
    print("\nOpik에 G-Eval 결과 기록 중...")
    from opik.evaluation.metrics import BaseMetric
    from opik.evaluation.metrics.score_result import ScoreResult

    opik_client = opik.Opik()
    dataset = opik_client.get_or_create_dataset(
        name="mlb-geval-experiment",
        description="DeepEval G-Eval 기반 MLB 에이전트 평가 결과",
    )

    # 데이터셋에 항목 추가 (점수 포함)
    dataset_items = []
    for i, question in enumerate(EVAL_QUESTIONS):
        item = {"input": question, "actual_output": agent_outputs[i]}
        for metric_name, (s, r) in score_map[i].items():
            item[metric_name] = s
        dataset_items.append(item)
    dataset.insert(dataset_items)

    # score_map을 클로저로 캡처하는 Opik 메트릭 팩토리
    def make_opik_metric(metric_name: str, scores: Dict[int, Dict[str, tuple]]):
        class _Metric(BaseMetric):
            name = metric_name

            def score(self, output: str, **kwargs) -> ScoreResult:
                for idx, data in scores.items():
                    if agent_outputs[idx] == output and metric_name in data:
                        s, r = data[metric_name]
                        return ScoreResult(name=self.name, value=s, reason=r or "")
                return ScoreResult(name=self.name, value=0.0, reason="매칭 실패")

        return _Metric()

    opik_metrics = [make_opik_metric(m.name, score_map) for m in metrics]

    def geval_task(item: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": item.get("actual_output", "")}

    opik.evaluate(
        dataset=dataset,
        task=geval_task,
        scoring_metrics=opik_metrics,
        experiment_name_prefix="mlb-geval",
        project_name=settings.OPIK.PROJECT if settings.OPIK else None,
        experiment_config={
            "eval_type": "deepeval-geval",
            "model": settings.OPENAI_MODEL,
            "judge_model": "gpt-4o",
        },
    )

    print("Opik 기록 완료!")
    return results


if __name__ == "__main__":
    run_experiment()
