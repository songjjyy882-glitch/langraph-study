# LangGraph Study - MLB 야구 분석 에이전트

LangGraph + FastAPI 기반의 AI 에이전트 교육 프로젝트.
MLB 야구 데이터를 활용한 선수 스탯 조회, 비교 분석, Statcast 심층 분석 기능을 제공합니다.

## 프로젝트 구조

```
langraph-study/
├── agent/                  # Backend (FastAPI + LangGraph)
│   ├── app/
│   │   ├── agents/         # 에이전트 정의
│   │   │   ├── agent.py            # 메인 에이전트 (ReAct)
│   │   │   ├── prompts.py          # 시스템 프롬프트
│   │   │   ├── baseball_agent.py   # 야구 전용 에이전트
│   │   │   ├── baseball_tools.py   # MLB 데이터 조회 도구 (pybaseball)
│   │   │   └── baseball_prompts.py # 야구 전용 프롬프트
│   │   ├── api/routes/     # API 엔드포인트
│   │   │   ├── chat.py             # POST /api/v1/chat
│   │   │   └── baseball_chat.py    # POST /api/v1/baseball/chat
│   │   ├── services/       # 비즈니스 로직
│   │   ├── models/         # Pydantic 모델
│   │   ├── core/           # 설정 (config)
│   │   └── utils/          # 유틸리티
│   └── pyproject.toml
├── ui/                     # Frontend (React + TypeScript)
│   ├── src/
│   │   ├── pages/          # 페이지 컴포넌트
│   │   ├── components/     # UI 컴포넌트
│   │   ├── hooks/          # 커스텀 훅
│   │   ├── services/       # API 서비스
│   │   └── store/          # Jotai 상태 관리
│   └── package.json
└── docs/                   # 설계 문서
```

## 주요 기능

### MLB 야구 분석 도구

| 도구 | 기능 | 예시 질문 |
|------|------|----------|
| `search_player_stats` | 선수 시즌 타격/투수 스탯 조회 | "오타니 2024시즌 성적 알려줘" |
| `compare_players` | 두 선수 스탯 비교 | "오타니 vs 저지 비교해줘" |
| `search_game_results` | 팀 경기 결과 조회 | "다저스 최근 경기 결과" |
| `search_statcast` | Statcast 심층 데이터 분석 | "오타니 구종별 분석해줘" |

### 기술 스택

**Backend**
- FastAPI + Uvicorn
- LangChain + LangGraph (ReAct Agent)
- OpenAI GPT
- pybaseball (MLB 데이터)

**Frontend**
- React + TypeScript + Vite
- Jotai (상태 관리)
- MUI (UI 컴포넌트)
- Highcharts (차트/테이블)

## 시작하기

### 사전 요구사항

- Python 3.11+
- Node.js 18+
- OpenAI API Key

### Backend 설정

```bash
cd agent

# .env 파일 생성
cat > .env << EOF
API_V1_PREFIX=/api/v1
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4o
EOF

# 의존성 설치
uv sync

# 서버 실행
uv run uvicorn app.main:app --reload --port 8000
```

### Frontend 설정

```bash
cd ui

# .env 파일 생성
cp env.sample .env

# 의존성 설치
pnpm install

# 개발 서버 실행
pnpm dev
```

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| POST | `/api/v1/chat` | 메인 채팅 (야구 도구 포함) |
| POST | `/api/v1/baseball/chat` | 야구 전용 채팅 |
| GET | `/api/v1/threads` | 대화 목록 조회 |
| GET | `/api/v1/threads/{id}` | 대화 상세 조회 |
| GET | `/health` | 헬스체크 |

### 요청 예시

```bash
curl -N -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "message": "오타니 2024시즌 성적 분석해줘"
  }'
```

### 응답 형식 (SSE)

```
data: {"step": "model", "tool_calls": ["search_player_stats"]}
data: {"step": "tools", "name": "search_player_stats", "content": {...}}
data: {"step": "done", "message_id": "...", "content": "분석 결과...", "metadata": {}}
```

## 향후 계획

- Elasticsearch 연동을 통한 데이터 적재 및 검색 최적화
