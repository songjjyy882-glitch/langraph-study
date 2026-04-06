"""
Deep Agent - 복잡한 다단계 분석 질문을 처리하는 계층적 에이전트.

deepagents 라이브러리의 create_deep_agent를 사용하여,
메인 에이전트(Planner)가 sub-agent들에게 작업을 위임하는 구조.

Sub-agents:
    - realtime_subagent: pybaseball 기반 최근 시즌 스탯 조회
    - historical_subagent: Elasticsearch 기반 역대 기록 검색
    - game_subagent: 경기 결과 조회

Use Cases:
    - "오타니와 베이브 루스 비교 분석" (역대 + 실시간 동시)
    - "다저스 최근 5경기 + 핵심 선수 스탯 분석" (경기 + 스탯 동시)
    - "역대 50홈런 시즌 친 선수들 공통점" (다단계 검색 + 분석)
"""

from deepagents import create_deep_agent, SubAgent
from langchain.chat_models import init_chat_model

from app.agents.baseball_tools import (
    search_player_stats,
    compare_players,
    search_game_results,
    search_statcast,
)
from app.agents.es_tools import (
    search_historical_batting,
    search_historical_pitching,
)
from app.core.config import settings


# ── Sub-agent 정의 ────────────────────────────────────────────

realtime_subagent: SubAgent = {
    "name": "realtime_stats_agent",
    "description": (
        "최근 시즌(2022~현재) MLB 선수의 스탯을 조회하는 전문 에이전트. "
        "선수 개별 스탯, 두 선수 비교, Statcast 심층 데이터 조회가 가능하다. "
        "한국/일본 선수(이정후, 김혜성, 오타니 등)도 모두 MLB 선수로 처리한다."
    ),
    "system_prompt": """당신은 MLB 최근 시즌 스탯 조회 전문가입니다.
사용자 요청에 따라 적절한 도구를 호출하여 데이터를 반환하세요.

도구:
- search_player_stats: 단일 선수의 시즌 스탯
- compare_players: 두 선수 비교
- search_statcast: Statcast 심층 데이터 (구속, 회전수, 타구속도 등)

한글 선수명은 영문으로 변환:
- 오타니 → Shohei Ohtani
- 저지 → Aaron Judge
- 이정후 → Jung Hoo Lee (SF Giants)
- 김혜성 → Hyeseong Kim (LA Dodgers)
- 야마모토 → Yoshinobu Yamamoto

중요: 한국/일본 선수도 모두 MLB 진출 선수입니다. KBO/NPB라고 답하지 마세요.
도구 결과를 그대로 반환하고, 분석은 메인 에이전트가 합니다.""",
    "tools": [search_player_stats, compare_players, search_statcast],
}


historical_subagent: SubAgent = {
    "name": "historical_records_agent",
    "description": (
        "MLB 역대 기록(1871~2021)을 Elasticsearch에서 검색하는 전문 에이전트. "
        "베이브 루스, 행크 아론, 노란 라이언 같은 레전드 선수의 통산/시즌별 기록 조회 가능. "
        "연도/팀/홈런수/ERA 등 조건 검색도 지원한다."
    ),
    "system_prompt": """당신은 MLB 역대 기록 검색 전문가입니다.
Elasticsearch에 저장된 Lahman 데이터베이스(1871~2021)를 검색합니다.

도구:
- search_historical_batting: 역대 타격 기록 검색 (선수명/팀/연도/홈런수)
- search_historical_pitching: 역대 투구 기록 검색 (선수명/팀/연도/ERA)

한글 선수명은 영문으로 변환:
- 베이브 루스 → Babe Ruth
- 행크 아론 → Hank Aaron
- 노란 라이언 → Nolan Ryan
- 사이 영 → Cy Young

팀 약어: NYA=양키스, LAN=다저스, BOS=보스턴, SFN=자이언츠

도구 결과를 그대로 반환하고, 분석은 메인 에이전트가 합니다.""",
    "tools": [search_historical_batting, search_historical_pitching],
}


game_subagent: SubAgent = {
    "name": "game_results_agent",
    "description": (
        "MLB 팀의 최근 경기 결과를 조회하는 전문 에이전트. "
        "팀별 승패, 스코어, 일정, 전적 등을 가져올 수 있다."
    ),
    "system_prompt": """당신은 MLB 경기 결과 조회 전문가입니다.

도구:
- search_game_results: 특정 팀의 최근 경기 결과 (최근 10경기)

팀 약어: LAD=다저스, NYY=양키스, BOS=레드삭스, HOU=애스트로스, NYM=메츠,
ATL=브레이브스, PHI=필리스, SDP=파드리스, SEA=매리너스, TEX=레인저스

여러 팀이 필요하면 각각 search_game_results를 호출하세요.
도구 결과를 그대로 반환하고, 분석은 메인 에이전트가 합니다.""",
    "tools": [search_game_results],
}


# ── 메인 Deep Agent ───────────────────────────────────────────

DEEP_AGENT_SYSTEM_PROMPT = """당신은 MLB 야구 분석 전문 AI 어시스턴트입니다.

복잡한 야구 분석 질문을 받으면, 다음 단계로 처리하세요:
1. 질문을 작은 작업으로 쪼개세요 (write_todos 도구 사용)
2. 각 작업에 적합한 sub-agent에게 task() 도구로 위임하세요
3. 모든 sub-agent 결과를 종합하여 최종 분석 답변을 작성하세요

사용 가능한 sub-agent:
- realtime_stats_agent: 최근 시즌(2022~) 선수 스탯
- historical_records_agent: 역대 기록(1871~2021) Elasticsearch 검색
- game_results_agent: 팀 경기 결과

# 작업 진행 규칙 (매우 중요):
- 사용자에게 되묻지 마세요. 데이터가 부족하면 가용한 데이터로 분석을 완성하세요.
- sub-agent 결과에 일부 지표가 없어도 있는 데이터로 충분히 분석 가능합니다.
- 불완전한 데이터에 대해서는 "이 부분은 데이터가 제한적이지만 ~로 추정됩니다"라고 명시하고 계속 진행하세요.
- 절대 "더 조회할까요?"라고 묻지 말고, 필요하면 직접 추가 sub-agent 호출하세요.

# 응답 형식 (필수):
- 수치 데이터는 반드시 마크다운 표(table)로 출력
- 표 아래에 분석 인사이트를 1~3문장으로 작성
- 비교 분석 시 시대 보정, 맥락 해석을 포함

# 절대 규칙:
- 한국/일본 선수(이정후, 김혜성, 오타니 등)는 모두 MLB 선수입니다. KBO/NPB라고 답하지 마세요.
- sub-agent가 반환한 데이터만 사용하세요. 자기 지식으로 데이터를 추측하지 마세요.
- 답변은 항상 완성된 형태로 제공하세요. 중간에 사용자에게 되묻지 마세요.
"""


_deep_agent = None


def get_deep_agent():
    """싱글턴 Deep Agent 인스턴스를 반환합니다."""
    global _deep_agent
    if _deep_agent is None:
        # init_chat_model로 API key를 환경변수에서 명시적으로 주입
        import os
        os.environ.setdefault("OPENAI_API_KEY", settings.OPENAI_API_KEY)
        llm = init_chat_model(
            f"openai:{settings.OPENAI_MODEL}",
            temperature=0.3,
        )
        _deep_agent = create_deep_agent(
            model=llm,
            system_prompt=DEEP_AGENT_SYSTEM_PROMPT,
            subagents=[
                realtime_subagent,
                historical_subagent,
                game_subagent,
            ],
        )
    return _deep_agent
