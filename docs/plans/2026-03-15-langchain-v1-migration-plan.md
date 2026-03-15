# LangChain v1.0 마이그레이션 및 개선 계획

## 개요
LangChain 최신 문서(Core Components + Middleware) 기준으로 현재 프로젝트를 개선한다.
import 경로 변경, 아키텍처 개선, 미들웨어 적용까지 포함.

---

## Phase 1: Import 경로 v1.0 마이그레이션 (난이도: 낮음)

### 1-1. `langchain_core` → `langchain` import 변경

| 파일 | 현재 | 변경 |
|------|------|------|
| `agents/agent.py` | `from langchain_core.tools import tool` | `from langchain.tools import tool` |
| `agents/baseball_agent.py` | `from langchain_core.tools import tool` | `from langchain.tools import tool` |
| `agents/baseball_tools.py` | `from langchain_core.tools import tool` | `from langchain.tools import tool` |
| `services/agent_service.py` | `from langchain_core.messages import HumanMessage` | dict 형태 사용 또는 `from langchain.messages import HumanMessage` |
| `services/baseball_agent_service.py` | `from langchain_core.messages import HumanMessage` | 동일 |

### 1-2. 모델 초기화 방식 변경

```python
# 현재
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model=settings.OPENAI_MODEL, api_key=settings.OPENAI_API_KEY)

# 변경
from langchain.chat_models import init_chat_model
llm = init_chat_model(settings.OPENAI_MODEL, temperature=0)
```

- `init_chat_model`은 프로바이더 무관 통합 인터페이스
- 나중에 Anthropic/Gemini 등으로 교체 시 모델명만 변경하면 됨
- `langchain-openai` 의존성은 유지 (provider 패키지로 필요)

---

## Phase 2: 아키텍처 개선 (난이도: 중간)

### 2-1. 에이전트 싱글턴 패턴 적용

**문제**: 현재 매 요청마다 `build_agent()`를 호출하여 에이전트를 재생성
**해결**: 모듈 레벨에서 한 번만 생성하고 재사용

```python
# agents/agent.py
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent
```

- checkpointer가 thread_id로 대화를 분리하므로 에이전트 인스턴스 공유 가능
- 적용 파일: `services/agent_service.py`, `services/baseball_agent_service.py`

### 2-2. Structured Output 적용 (ChatResponse 도구 제거)

**문제**: `ChatResponse`라는 커스텀 도구로 응답 포맷을 강제하고 있음
**해결**: `response_format`에 Pydantic 모델을 넘겨 자동 구조화

```python
from pydantic import BaseModel, Field

class ChatResponseFormat(BaseModel):
    message_id: str = Field(description="응답 메시지 UUID")
    content: str = Field(description="답변 내용")
    metadata: dict = Field(default_factory=dict, description="추가 메타데이터")

agent = create_agent(
    model=llm,
    tools=baseball_tools,        # ChatResponse 도구 제거
    response_format=ChatResponseFormat,
    checkpointer=_checkpointer,
)

# 결과 접근: result["structured_response"]
```

- `agent_service.py`의 스트리밍 로직에서 ChatResponse 파싱 부분 수정 필요
- 프론트엔드 응답 포맷은 유지

### 2-3. 도구 입력 스키마 Pydantic 적용

**문제**: 함수 시그니처 + docstring으로만 스키마 정의
**해결**: `args_schema`로 명확한 입력 스키마

```python
from pydantic import BaseModel, Field

class PlayerStatsInput(BaseModel):
    player_name: str = Field(description="선수 영문 이름 (예: Shohei Ohtani)")
    season: int | None = Field(default=None, description="시즌 연도 (예: 2024)")

@tool(args_schema=PlayerStatsInput)
def search_player_stats(player_name: str, season: int = None) -> str:
    ...
```

- 적용 대상: `search_player_stats`, `compare_players`, `search_game_results`, `search_statcast`

### 2-4. 메시지 입력 dict 형태로 변경

```python
# 현재
from langchain_core.messages import HumanMessage
agent.astream({"messages": [HumanMessage(content=user_messages)]}, ...)

# 변경
agent.astream({"messages": [{"role": "user", "content": user_messages}]}, ...)
```

---

## Phase 3: 미들웨어 적용 (난이도: 중간)

### 3-1. SummarizationMiddleware

- 대화가 길어지면 컨텍스트 윈도우 초과 방지
- 오래된 메시지를 자동 요약

```python
from langchain.agents.middleware import SummarizationMiddleware

SummarizationMiddleware(
    model="gpt-4o-mini",
    trigger=("tokens", 4000),
    keep=("messages", 10),
)
```

### 3-2. ToolRetryMiddleware

- pybaseball API 호출 실패 시 자동 재시도
- 외부 API 의존성이 있으므로 필수

```python
from langchain.agents.middleware import ToolRetryMiddleware

ToolRetryMiddleware(
    max_retries=2,
    backoff_factor=2.0,
)
```

### 3-3. ModelCallLimitMiddleware

- 무한루프/과도한 API 호출 방지
- 현재 `DEEPAGENT_RECURSION_LIMIT=20` 설정을 대체

```python
from langchain.agents.middleware import ModelCallLimitMiddleware

ModelCallLimitMiddleware(
    run_limit=20,
    exit_behavior="end",
)
```

### 3-4. 최종 에이전트 생성 코드

```python
agent = create_agent(
    model=init_chat_model(settings.OPENAI_MODEL, temperature=0),
    tools=baseball_tools,
    system_prompt=system_prompt,
    response_format=ChatResponseFormat,
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 10),
        ),
        ToolRetryMiddleware(max_retries=2),
        ModelCallLimitMiddleware(run_limit=20, exit_behavior="end"),
    ],
    checkpointer=_checkpointer,
)
```

---

## Phase 4: 스트리밍 개선 (난이도: 중간)

### 4-1. v2 스트리밍 포맷 적용

```python
# 현재
agent.astream(..., stream_mode="updates")

# 변경
agent.astream(..., stream_mode=["updates", "messages"], version="v2")
```

- `updates`: 도구 호출/결과 단위 업데이트 (기존 유지)
- `messages`: 토큰 단위 스트리밍 추가 (글자가 하나씩 나타남)
- `version="v2"`: 통일된 chunk 포맷 (`type`, `data` 키)

### 4-2. ToolRuntime으로 진행상황 스트리밍

```python
from langchain.tools import tool, ToolRuntime

@tool
def search_player_stats(player_name: str, season: int = None, runtime: ToolRuntime = None) -> str:
    if runtime and runtime.stream_writer:
        runtime.stream_writer(f"{player_name} 데이터 조회 중...")
    # 기존 로직
```

### 4-3. agent_service.py 스트리밍 로직 수정

- v2 포맷에 맞게 chunk 파싱 로직 변경
- `structured_response` 키에서 최종 응답 추출
- 토큰 스트리밍 이벤트를 프론트에 전달

---

## Phase 5: 향후 확장 (난이도: 높음, ES 연동 시)

### 5-1. Checkpointer 프로덕션 전환

```python
# 현재 (인메모리 - 서버 재시작 시 소실)
from langgraph.checkpoint.memory import MemorySaver

# 프로덕션
from langgraph.checkpoint.postgres import PostgresSaver
```

### 5-2. ModelFallbackMiddleware (선택)

```python
ModelFallbackMiddleware("gpt-4o-mini", "claude-sonnet-4-6")
```

---

## 작업 순서 및 체크리스트

### Phase 1 (즉시 적용) ✅
- [x] `langchain_core` → `langchain` import 변경 (5개 파일)
- [x] `ChatOpenAI` → `init_chat_model` 변경
- [x] `HumanMessage` → dict 형태 변경
- [ ] 동작 테스트

### Phase 2 (구조 개선) ✅ (싱글턴 적용 완료)
- [x] 에이전트 싱글턴 패턴 적용
- [ ] Structured Output 적용 (ChatResponse 도구 제거)
- [ ] 도구 `args_schema` Pydantic 모델 추가
- [ ] `agent_service.py` 스트리밍 로직 수정
- [ ] 동작 테스트

### Phase 3 (미들웨어) ✅
- [x] SummarizationMiddleware 적용
- [x] ToolRetryMiddleware 적용
- [x] ModelCallLimitMiddleware 적용
- [ ] 동작 테스트

### Phase 4 (스트리밍)
- [ ] v2 스트리밍 포맷 적용
- [ ] ToolRuntime 진행상황 스트리밍
- [ ] 프론트엔드 스트리밍 파싱 수정
- [ ] 동작 테스트

### Phase 5 (ES 연동 시)
- [ ] PostgresSaver checkpointer 전환
- [ ] ModelFallbackMiddleware 적용 (선택)

---

## 변경 영향 파일 목록

| 파일 | Phase | 변경 내용 |
|------|-------|----------|
| `agents/agent.py` | 1,2,3 | import 변경, init_chat_model, structured output, 미들웨어, 싱글턴 |
| `agents/baseball_agent.py` | 1,2,3 | 동일 |
| `agents/baseball_tools.py` | 1,2,4 | import 변경, args_schema, ToolRuntime |
| `agents/prompts.py` | 2 | Response Format 섹션 제거 (structured output로 대체) |
| `services/agent_service.py` | 1,2,4 | import 변경, 싱글턴, 스트리밍 v2 |
| `services/baseball_agent_service.py` | 1,2,4 | 동일 |
| `ui/src/hooks/useChat.ts` | 4 | 토큰 스트리밍 처리 추가 |
| `ui/src/services/common.ts` | 4 | v2 SSE 파싱 |
