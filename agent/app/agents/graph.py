"""
MLB 야구 분석 에이전트 - LangGraph StateGraph 기반 커스텀 그래프

Graph Structure:
    [User Question]
          ↓
      [router] ─── "realtime_stats" ──→ [realtime_node] ──→ [analysis] ──→ END
               ├── "historical"    ──→ [historical_node] ──→ [analysis] ──→ END
               ├── "game_results"  ──→ [game_node]       ──→ [analysis] ──→ END
               └── "general"       ──→ [analysis]        ──→ END
"""

import json
import re
from typing import TypedDict, Annotated

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from app.agents.baseball_prompts import baseball_system_prompt
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


# ── 1. State 정의 ──────────────────────────────────────────────

class BaseballState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query_type: str
    search_results: str


# ── 2. LLM 헬퍼 ───────────────────────────────────────────────

def _get_llm(temperature: float = 0.0):
    return init_chat_model(settings.OPENAI_MODEL, temperature=temperature)


def _get_last_user_message(state: "BaseballState") -> str:
    """마지막 사용자 메시지를 추출합니다."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _get_conversation_history(state: "BaseballState", max_turns: int = 10) -> list[BaseMessage]:
    """대화 히스토리에서 최근 N턴을 반환합니다 (ToolMessage 제외)."""
    history = []
    for msg in state["messages"]:
        if isinstance(msg, (HumanMessage, AIMessage)):
            history.append(msg)
    return history[-max_turns:]


# ── 3. Router Node ─────────────────────────────────────────────

_HISTORICAL_KEYWORDS = [
    "역대", "올타임", "통산", "레전드", "명예의전당",
    "베이브", "루스", "행크", "아론", "윌리", "메이스",
    "테드", "윌리엄스", "노란", "라이언", "사이영",
    "역사", "올드", "클래식", "과거", "1900", "1800",
    "historical", "all-time", "alltime", "legend",
]

_GAME_KEYWORDS = [
    "경기", "오늘", "어제", "승패", "스코어", "일정",
    "결과", "시리즈", "전적", "순위",
    "game", "score", "schedule", "record", "standing",
]

_STATS_KEYWORDS = [
    "성적", "스탯", "타율", "홈런", "ERA", "era",
    "비교", "WAR", "war", "OPS", "ops", "WHIP", "whip",
    "타격", "투구", "투수", "타자", "안타", "삼진",
    "시즌", "올해", "작년", "금년",
    "stat", "stats", "batting", "pitching", "compare",
    "statcast", "구속", "회전수", "타구속도", "발사각",
]


def _classify_by_keywords(query: str) -> str | None:
    """키워드 기반 빠른 분류. 매칭되면 query_type을, 없으면 None 반환."""
    q = query.lower()

    # 역대/역사 키워드 우선 체크
    for kw in _HISTORICAL_KEYWORDS:
        if kw.lower() in q:
            return "historical"

    # 경기 결과 키워드
    for kw in _GAME_KEYWORDS:
        if kw.lower() in q:
            return "game_results"

    # 실시간 스탯 키워드
    for kw in _STATS_KEYWORDS:
        if kw.lower() in q:
            return "realtime_stats"

    return None


def router(state: BaseballState) -> BaseballState:
    """사용자 질문을 분류하여 query_type을 결정합니다. 대화 컨텍스트를 고려합니다."""
    user_msg = _get_last_user_message(state)
    history = _get_conversation_history(state)

    if not user_msg:
        return {**state, "query_type": "general", "search_results": ""}

    # 1차: 키워드 매칭 (단독 질문일 때만 신뢰)
    if len(history) <= 1:
        query_type = _classify_by_keywords(user_msg)
        if query_type:
            return {**state, "query_type": query_type, "search_results": ""}

    # 2차: LLM 분류 (대화 히스토리 포함)
    llm = _get_llm(temperature=0.0)

    # 이전 대화 요약
    history_text = ""
    if len(history) > 1:
        history_text = "\n\n[이전 대화]\n"
        for msg in history[:-1]:
            role = "사용자" if isinstance(msg, HumanMessage) else "어시스턴트"
            content = msg.content[:200]  # 길이 제한
            history_text += f"{role}: {content}\n"

    classification_prompt = f"""다음 사용자 질문을 분류하세요. 반드시 아래 4가지 중 하나만 답하세요:

- realtime_stats: 최근 시즌(2022~현재) 선수 성적, 스탯 조회, 선수 비교, Statcast 데이터
- historical: 역대 기록, 과거(2021년 이전) 통계, 레전드 선수, 통산 기록
- game_results: 경기 결과, 스코어, 일정, 팀 전적, 승패
- general: 야구 규칙, 용어 설명, 일반 상식 등 데이터 조회가 불필요한 질문

이전 대화 맥락이 있다면 그것까지 고려하세요. 예를 들어 이전에 "베이브 루스" 얘기를 하다가 "그럼 통산 타율은?"이라고 물으면 historical입니다.
{history_text}
[현재 질문]
{user_msg}

분류 결과(한 단어만):"""

    response = llm.invoke([HumanMessage(content=classification_prompt)])
    result = response.content.strip().lower()

    for qt in ["realtime_stats", "historical", "game_results", "general"]:
        if qt in result:
            return {**state, "query_type": qt, "search_results": ""}

    return {**state, "query_type": "general", "search_results": ""}


# ── 4. Search Nodes ────────────────────────────────────────────

def _execute_tool_calls(ai_message: AIMessage, tool_map: dict) -> str:
    """AIMessage의 tool_calls를 실행하고 결과를 모아 반환합니다."""
    results = []
    for tool_call in ai_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        if tool_name in tool_map:
            tool_fn = tool_map[tool_name]
            try:
                result = tool_fn.invoke(tool_args)
                results.append(f"[{tool_name}] {result}")
            except Exception as e:
                results.append(f"[{tool_name}] 오류: {str(e)}")
    return "\n\n".join(results) if results else "검색 결과 없음"


def realtime_node(state: BaseballState) -> BaseballState:
    """실시간 선수 스탯을 조회합니다 (pybaseball 기반). 대화 히스토리 고려."""
    user_msg = _get_last_user_message(state)
    history = _get_conversation_history(state)

    tools = [search_player_stats, compare_players, search_statcast]
    tool_map = {t.name: t for t in tools}

    llm = _get_llm(temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)

    system_msg = SystemMessage(content="""당신은 MLB 야구 분석 도구 호출자입니다.
사용자의 최근 시즌 스탯 질문에 답하기 위해 적절한 도구를 호출하세요.
이전 대화 맥락을 반드시 고려하세요. 예: 이전에 "오타니" 얘기 후 "작년은?"이라고 하면 오타니의 작년 스탯을 조회.
한글 선수명은 영문으로 변환하세요. 예: "오타니" → "Shohei Ohtani", "저지" → "Aaron Judge".""")

    messages = [system_msg] + history
    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        search_results = _execute_tool_calls(response, tool_map)
    else:
        search_results = "도구 호출 없음 - LLM이 직접 응답을 생성합니다."

    return {**state, "search_results": search_results}


def historical_node(state: BaseballState) -> BaseballState:
    """역대 기록을 조회합니다 (Elasticsearch 기반). 대화 히스토리 고려."""
    history = _get_conversation_history(state)

    tools = [search_historical_batting, search_historical_pitching]
    tool_map = {t.name: t for t in tools}

    llm = _get_llm(temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)

    system_msg = SystemMessage(content="""당신은 MLB 역대 기록 검색 도구 호출자입니다.
사용자의 역대 기록 질문에 답하기 위해 적절한 도구를 호출하세요.
이전 대화 맥락을 반드시 고려하세요. 예: 이전에 "베이브 루스" 얘기 후 "그럼 통산 타점은?"이라고 하면 베이브 루스의 통산 타점을 조회.
한글 선수명은 영문으로 변환하세요. 예: "베이브 루스" → "Babe Ruth", "행크 아론" → "Hank Aaron".
팀 약어 참고: NYA=양키스, LAN=다저스, BOS=보스턴, SFN=자이언츠, SLN=카디널스.""")

    messages = [system_msg] + history
    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        search_results = _execute_tool_calls(response, tool_map)
    else:
        search_results = "도구 호출 없음 - LLM이 직접 응답을 생성합니다."

    return {**state, "search_results": search_results}


def game_node(state: BaseballState) -> BaseballState:
    """경기 결과를 조회합니다. 대화 히스토리 고려."""
    history = _get_conversation_history(state)

    tools = [search_game_results]
    tool_map = {t.name: t for t in tools}

    llm = _get_llm(temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)

    system_msg = SystemMessage(content="""당신은 MLB 경기 결과 검색 도구 호출자입니다.
사용자의 경기 결과 질문에 답하기 위해 적절한 도구를 호출하세요.
이전 대화 맥락을 반드시 고려하세요. 예: 이전에 "다저스" 얘기 후 "어제 결과는?"이라고 하면 다저스의 어제 경기를 조회.

중요: 사용자가 특정 팀을 언급하지 않고 "오늘 경기", "경기 결과" 등 전체 경기를 물어보면,
주요 인기 팀 5개(LAD, NYY, BOS, HOU, NYM)에 대해 각각 search_game_results를 호출하세요.

팀 약어 참고: LAD=다저스, NYY=양키스, BOS=레드삭스, HOU=애스트로스, NYM=메츠,
ATL=브레이브스, PHI=필리스, SDP=파드리스, SEA=매리너스, TEX=레인저스.""")

    messages = [system_msg] + history
    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        search_results = _execute_tool_calls(response, tool_map)
    else:
        search_results = "도구 호출 없음 - LLM이 직접 응답을 생성합니다."

    return {**state, "search_results": search_results}


# ── 5. Analysis Node ───────────────────────────────────────────

def analysis(state: BaseballState) -> BaseballState:
    """검색 결과를 바탕으로 최종 분석 답변을 생성합니다. 대화 히스토리 포함."""
    user_msg = _get_last_user_message(state)
    history = _get_conversation_history(state)

    llm = _get_llm(temperature=0.7)

    query_type = state.get("query_type", "general")
    search_results = state.get("search_results", "")

    # 시스템 프롬프트 + 이전 대화 히스토리(현재 질문 제외)
    messages: list[BaseMessage] = [SystemMessage(content=baseball_system_prompt)]
    if len(history) > 1:
        messages.extend(history[:-1])

    if query_type == "general" or not search_results:
        # 일반 질문: 검색 결과 없이 답변
        messages.append(HumanMessage(content=user_msg))
    else:
        # 데이터 기반 답변 (현재 질문 + 검색 결과)
        messages.append(HumanMessage(content=f"""{user_msg}

[참고 데이터 - 도구 조회 결과]
{search_results}"""))

    response = llm.invoke(messages)

    return {
        **state,
        "messages": [response],
    }


# ── 6. Routing 함수 ───────────────────────────────────────────

def route_by_query_type(state: BaseballState) -> str:
    """query_type에 따라 다음 노드를 결정합니다."""
    query_type = state.get("query_type", "general")
    if query_type == "realtime_stats":
        return "realtime_node"
    elif query_type == "historical":
        return "historical_node"
    elif query_type == "game_results":
        return "game_node"
    else:
        return "analysis"


# ── 7. Graph 조립 ─────────────────────────────────────────────

def build_baseball_graph():
    """MLB 야구 분석 StateGraph를 빌드하고 컴파일합니다."""
    graph = StateGraph(BaseballState)

    # 노드 추가
    graph.add_node("router", router)
    graph.add_node("realtime_node", realtime_node)
    graph.add_node("historical_node", historical_node)
    graph.add_node("game_node", game_node)
    graph.add_node("analysis", analysis)

    # 엔트리 포인트
    graph.set_entry_point("router")

    # 조건부 엣지: router → 각 노드
    graph.add_conditional_edges(
        "router",
        route_by_query_type,
        {
            "realtime_node": "realtime_node",
            "historical_node": "historical_node",
            "game_node": "game_node",
            "analysis": "analysis",
        },
    )

    # 검색 노드 → analysis
    graph.add_edge("realtime_node", "analysis")
    graph.add_edge("historical_node", "analysis")
    graph.add_edge("game_node", "analysis")

    # analysis → END
    graph.add_edge("analysis", END)

    # 컴파일 (MemorySaver 체크포인터)
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    return compiled


# ── 8. 싱글턴 ─────────────────────────────────────────────────

_graph = None


def get_baseball_graph():
    """싱글턴 야구 분석 그래프 인스턴스를 반환합니다."""
    global _graph
    if _graph is None:
        _graph = build_baseball_graph()
    return _graph
