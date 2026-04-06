# LangGraph 커스텀 그래프 도입 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 현재 LangChain `create_agent` 기반의 암묵적 ReAct 에이전트를 LangGraph `StateGraph`로 교체하여, 라우터 → 검색 → 분석 노드로 분기하는 명시적 그래프를 구축한다.

**Architecture:** 사용자 질문이 들어오면 라우터 노드가 질문 유형(실시간 스탯 / 역대 기록 / 경기 결과 / 일반 대화)을 판단하고, 해당 검색 노드로 분기하여 데이터를 가져온 뒤, 분석 노드에서 LLM이 최종 답변을 생성한다. 기존 `baseball_tools.py`와 `es_tools.py`의 tool 함수들은 그대로 재사용하되, LangChain tool이 아닌 일반 함수로 호출한다.

**Tech Stack:** LangGraph 0.4.5, StateGraph, langchain-openai, MemorySaver

---

## 그래프 구조

```
[사용자 질문]
      ↓
  [router] ─── "realtime_stats" ──→ [realtime_node] ──→ [analysis] ──→ END
           ├── "historical"    ──→ [historical_node] ──→ [analysis] ──→ END
           ├── "game_results"  ──→ [game_node]       ──→ [analysis] ──→ END
           └── "general"       ──→ [analysis]        ──→ END
```

---

### Task 1: State 정의 및 그래프 뼈대

**Files:**
- Create: `agent/app/agents/graph.py`

**Step 1: State 타입과 빈 그래프 작성**

```python
# agent/app/agents/graph.py
from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage


class BaseballState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query_type: str  # "realtime_stats" | "historical" | "game_results" | "general"
    search_results: str  # 검색 노드에서 가져온 데이터
```

**Step 2: 실행 확인**

Run: `cd agent && python3 -c "from app.agents.graph import BaseballState; print('OK')"`
Expected: `OK`

---

### Task 2: 라우터 노드 구현

**Files:**
- Modify: `agent/app/agents/graph.py`

**Step 1: 라우터 노드 추가**

LLM에게 질문 유형을 분류하게 한다. 키워드 기반으로 빠르게 판단하되, 애매한 경우 LLM이 판단한다.

```python
from langchain.chat_models import init_chat_model
from app.core.config import settings


def create_router_node(llm):
    """질문 유형을 분류하는 라우터 노드"""
    def router(state: BaseballState) -> BaseballState:
        last_message = state["messages"][-1].content

        # 키워드 기반 빠른 분류
        historical_keywords = ["역대", "올타임", "통산", "1900", "1800", "레전드", "베이브", "행크", "노란 라이언"]
        game_keywords = ["경기", "오늘", "어제", "승패", "스코어", "일정"]
        stats_keywords = ["성적", "스탯", "타율", "홈런", "ERA", "비교", "시즌"]

        lower = last_message.lower()

        if any(k in last_message for k in historical_keywords):
            query_type = "historical"
        elif any(k in last_message for k in game_keywords):
            query_type = "game_results"
        elif any(k in last_message for k in stats_keywords):
            query_type = "realtime_stats"
        else:
            # LLM으로 분류
            response = llm.invoke(
                f"""사용자 질문을 아래 4가지 중 하나로 분류하세요. 카테고리명만 답하세요.
- realtime_stats: 최근 시즌 선수 성적, 스탯 비교, Statcast 데이터
- historical: 역대 기록, 과거(2021년 이전) 통계, 레전드 선수
- game_results: 경기 결과, 일정, 스코어
- general: 인사, 야구 용어 설명, 일반 대화

질문: {last_message}
카테고리:"""
            )
            query_type = response.content.strip().lower()
            if query_type not in ["realtime_stats", "historical", "game_results", "general"]:
                query_type = "general"

        return {**state, "query_type": query_type, "search_results": ""}

    return router
```

**Step 2: 실행 확인**

Run: `cd agent && python3 -c "from app.agents.graph import create_router_node; print('OK')"`
Expected: `OK`

---

### Task 3: 검색 노드 구현 (3개)

**Files:**
- Modify: `agent/app/agents/graph.py`

**Step 1: 실시간 스탯 검색 노드**

기존 `baseball_tools.py`의 함수들을 직접 호출한다.

```python
from app.agents.baseball_tools import (
    search_player_stats, compare_players,
    search_game_results, search_statcast,
)
from app.agents.es_tools import search_historical_batting, search_historical_pitching


def create_realtime_node(llm):
    """최근 시즌 스탯을 조회하는 노드. LLM이 적절한 tool을 선택하여 호출한다."""
    tools = [search_player_stats, compare_players, search_statcast]

    def realtime_node(state: BaseballState) -> BaseballState:
        last_message = state["messages"][-1].content
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(
            f"사용자 질문에 답하기 위해 적절한 도구를 호출하세요.\n\n질문: {last_message}"
        )

        # tool call 실행
        results = []
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_fn = {t.name: t for t in tools}[tool_call["name"]]
                result = tool_fn.invoke(tool_call["args"])
                results.append(result)

        search_results = "\n".join(results) if results else "검색 결과가 없습니다."
        return {**state, "search_results": search_results}

    return realtime_node
```

**Step 2: 역대 기록 검색 노드**

```python
def create_historical_node(llm):
    """ES에서 역대 기록을 검색하는 노드"""
    tools = [search_historical_batting, search_historical_pitching]

    def historical_node(state: BaseballState) -> BaseballState:
        last_message = state["messages"][-1].content
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(
            f"사용자 질문에 답하기 위해 적절한 도구를 호출하세요.\n\n질문: {last_message}"
        )

        results = []
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_fn = {t.name: t for t in tools}[tool_call["name"]]
                result = tool_fn.invoke(tool_call["args"])
                results.append(result)

        search_results = "\n".join(results) if results else "검색 결과가 없습니다."
        return {**state, "search_results": search_results}

    return historical_node
```

**Step 3: 경기 결과 검색 노드**

```python
def create_game_node(llm):
    """팀 경기 결과를 조회하는 노드"""
    tools = [search_game_results]

    def game_node(state: BaseballState) -> BaseballState:
        last_message = state["messages"][-1].content
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(
            f"""사용자 질문에 답하기 위해 적절한 도구를 호출하세요.
특정 팀이 없으면 LAD, NYY, BOS, HOU, NYM 5개 팀 모두 조회하세요.

질문: {last_message}"""
        )

        results = []
        if response.tool_calls:
            for tool_call in response.tool_calls:
                result = search_game_results.invoke(tool_call["args"])
                results.append(result)

        search_results = "\n".join(results) if results else "검색 결과가 없습니다."
        return {**state, "search_results": search_results}

    return game_node
```

**Step 4: 실행 확인**

Run: `cd agent && python3 -c "from app.agents.graph import create_realtime_node, create_historical_node, create_game_node; print('OK')"`
Expected: `OK`

---

### Task 4: 분석 노드 구현

**Files:**
- Modify: `agent/app/agents/graph.py`

**Step 1: LLM 분석/답변 생성 노드**

```python
from langchain_core.messages import AIMessage
from app.agents.baseball_prompts import baseball_system_prompt


def create_analysis_node(llm):
    """검색 결과를 바탕으로 최종 답변을 생성하는 노드"""
    def analysis_node(state: BaseballState) -> BaseballState:
        last_message = state["messages"][-1].content
        search_results = state.get("search_results", "")
        query_type = state.get("query_type", "general")

        if query_type == "general":
            # 일반 대화는 검색 없이 바로 답변
            prompt = f"""{baseball_system_prompt}

사용자 질문: {last_message}"""
        else:
            prompt = f"""{baseball_system_prompt}

사용자 질문: {last_message}

아래는 검색된 데이터입니다. 이 데이터를 바탕으로 답변하세요:
{search_results}"""

        response = llm.invoke(prompt)
        return {
            **state,
            "messages": [AIMessage(content=response.content)],
        }

    return analysis_node
```

**Step 2: 실행 확인**

Run: `cd agent && python3 -c "from app.agents.graph import create_analysis_node; print('OK')"`
Expected: `OK`

---

### Task 5: 그래프 조립 및 빌드

**Files:**
- Modify: `agent/app/agents/graph.py`

**Step 1: 그래프 조립 함수 추가**

```python
def build_baseball_graph(model_name=None, temperature=0.7):
    """LangGraph StateGraph 기반 야구 분석 에이전트를 빌드한다."""
    llm = init_chat_model(
        model_name or settings.OPENAI_MODEL,
        temperature=temperature,
    )

    # 노드 생성
    router = create_router_node(llm)
    realtime_node = create_realtime_node(llm)
    historical_node = create_historical_node(llm)
    game_node = create_game_node(llm)
    analysis_node = create_analysis_node(llm)

    # 그래프 정의
    graph = StateGraph(BaseballState)

    # 노드 추가
    graph.add_node("router", router)
    graph.add_node("realtime", realtime_node)
    graph.add_node("historical", historical_node)
    graph.add_node("game", game_node)
    graph.add_node("analysis", analysis_node)

    # 엣지 정의
    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        lambda state: state["query_type"],
        {
            "realtime_stats": "realtime",
            "historical": "historical",
            "game_results": "game",
            "general": "analysis",
        },
    )

    graph.add_edge("realtime", "analysis")
    graph.add_edge("historical", "analysis")
    graph.add_edge("game", "analysis")
    graph.add_edge("analysis", END)

    # 컴파일
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# 싱글턴
_graph = None

def get_baseball_graph():
    global _graph
    if _graph is None:
        _graph = build_baseball_graph()
    return _graph
```

**Step 2: 실행 확인**

Run: `cd agent && python3 -c "from app.agents.graph import get_baseball_graph; g = get_baseball_graph(); print(g.get_graph().nodes)"`
Expected: 노드 목록 출력 (`router`, `realtime`, `historical`, `game`, `analysis`, `__end__`)

---

### Task 6: Service 레이어 연결

**Files:**
- Modify: `agent/app/services/baseball_agent_service.py`

**Step 1: `get_baseball_agent()` → `get_baseball_graph()`로 교체**

Service의 import를 변경하고, `agent.astream()` 호출 방식은 동일하게 유지한다. StateGraph의 `astream()`도 같은 인터페이스를 제공하므로 스트리밍 로직은 크게 바꿀 필요가 없다.

다만, 커스텀 그래프에서는 `structured_response`가 아닌 일반 `messages`로 응답이 오므로, `analysis` 노드의 마지막 AIMessage를 최종 응답으로 사용하도록 스트리밍 처리를 수정해야 한다.

```python
# 변경: import
from app.agents.graph import get_baseball_graph

# 변경: agent 가져오기
agent = get_baseball_graph()

# 변경: 스트리밍 chunk 처리에서 analysis 노드의 messages를 최종 응답으로 처리
# 기존 structured_response 대신 step == "analysis" 일 때 messages[-1].content를 사용
```

**Step 2: 실행 확인**

Run: `cd agent && python3 -m uvicorn app.main:app --port 8000` 후 curl 테스트
Expected: 정상 응답

**Step 3: Commit**

```bash
git add agent/app/agents/graph.py agent/app/services/baseball_agent_service.py
git commit -m "feat: LangGraph StateGraph 기반 커스텀 그래프 도입"
```

---

### Task 7: 통합 테스트

**Step 1: 각 라우팅 경로 테스트**

```bash
# 실시간 스탯
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "오타니 2025시즌 성적 알려줘", "thread_id": "550e8400-e29b-41d4-a716-446655440010"}'

# 역대 기록 (ES)
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "베이브 루스 역대 홈런 기록", "thread_id": "550e8400-e29b-41d4-a716-446655440011"}'

# 경기 결과
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "다저스 어제 경기 결과", "thread_id": "550e8400-e29b-41d4-a716-446655440012"}'

# 일반 대화
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "WAR이 뭐야?", "thread_id": "550e8400-e29b-41d4-a716-446655440013"}'
```

Expected: 각 질문이 올바른 노드로 라우팅되어 답변 생성

**Step 2: Commit**

```bash
git commit -m "test: LangGraph 커스텀 그래프 통합 테스트 완료"
```
