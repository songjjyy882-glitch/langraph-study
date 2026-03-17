"""
MLB 야구 에이전트 평가용 데이터셋 생성 스크립트.

사용법:
    cd agent && uv run python -m app.evaluation.dataset
"""

import os

from app.core.config import settings

# opik import 전에 환경변수 세팅 (API key 프롬프트 방지)
if settings.OPIK and settings.OPIK.URL_OVERRIDE:
    os.environ.setdefault("OPIK_URL_OVERRIDE", settings.OPIK.URL_OVERRIDE)
if settings.OPIK and settings.OPIK.PROJECT:
    os.environ.setdefault("OPIK_PROJECT_NAME", settings.OPIK.PROJECT)

import opik  # noqa: E402

# 에이전트 평가를 위한 QA 데이터셋
BASEBALL_EVAL_ITEMS = [
    # ── 선수 스탯 조회 (search_player_stats) ──
    {
        "input": "오타니 쇼헤이의 2024시즌 타격 성적 알려줘",
        "expected_tool": "search_player_stats",
        "expected_keywords": ["AVG", "HR", "OPS", "WAR"],
        "category": "player_stats",
    },
    {
        "input": "2024시즌 류현진 투수 성적을 조회해줘",
        "expected_tool": "search_player_stats",
        "expected_keywords": ["ERA", "WHIP", "IP"],
        "category": "player_stats",
    },
    {
        "input": "Aaron Judge 2024 stats",
        "expected_tool": "search_player_stats",
        "expected_keywords": ["HR", "RBI", "WAR"],
        "category": "player_stats",
    },
    # ── 선수 비교 (compare_players) ──
    {
        "input": "오타니와 저지의 2024시즌 성적 비교해줘",
        "expected_tool": "compare_players",
        "expected_keywords": ["HR", "OPS", "WAR"],
        "category": "compare",
    },
    {
        "input": "무키 벳츠와 후안 소토 비교 분석",
        "expected_tool": "compare_players",
        "expected_keywords": ["AVG", "OBP"],
        "category": "compare",
    },
    # ── 팀 경기 결과 (search_game_results) ──
    {
        "input": "다저스 2024시즌 최근 경기 결과 보여줘",
        "expected_tool": "search_game_results",
        "expected_keywords": ["LAD", "경기", "승", "패"],
        "category": "game_results",
    },
    {
        "input": "양키스 시즌 성적 알려줘",
        "expected_tool": "search_game_results",
        "expected_keywords": ["NYY"],
        "category": "game_results",
    },
    # ── Statcast (search_statcast) ──
    {
        "input": "오타니의 Statcast 데이터 분석해줘",
        "expected_tool": "search_statcast",
        "expected_keywords": ["구속", "타구속도", "speed"],
        "category": "statcast",
    },
    {
        "input": "게릿 콜의 구종별 구속과 회전수 분석",
        "expected_tool": "search_statcast",
        "expected_keywords": ["spin", "speed", "구종"],
        "category": "statcast",
    },
    # ── 일반 질문 (도구 불필요) ──
    {
        "input": "WAR이 뭐야?",
        "expected_tool": None,
        "expected_keywords": ["WAR", "대체", "승리"],
        "category": "general",
    },
    {
        "input": "OPS는 어떻게 계산해?",
        "expected_tool": None,
        "expected_keywords": ["OBP", "SLG", "출루율", "장타율"],
        "category": "general",
    },
    {
        "input": "2024 MLB 시즌 홈런왕은 누구야?",
        "expected_tool": "search_player_stats",
        "expected_keywords": ["Aaron Judge", "저지"],
        "category": "player_stats",
    },
]


def create_dataset() -> opik.Dataset:
    """Opik에 MLB 에이전트 평가용 데이터셋을 생성합니다."""
    client = opik.Opik()
    dataset = client.get_or_create_dataset(
        name="mlb-baseball-agent-eval",
        description="MLB 야구 에이전트 평가용 QA 데이터셋 — 도구 호출 정확도 및 응답 품질 평가",
    )

    dataset.insert(BASEBALL_EVAL_ITEMS)
    print(f"데이터셋 생성 완료: {dataset.name} ({len(BASEBALL_EVAL_ITEMS)}개 항목)")
    return dataset


if __name__ == "__main__":
    create_dataset()
