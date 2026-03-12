# MLB 야구 분석 에이전트 설계

## 개요
pybaseball 라이브러리를 활용한 MLB 야구 경기 분석 에이전트.
기존 범용 에이전트와 별도로 분리된 전용 에이전트로 구현.
추후 Elasticsearch 연동을 고려한 구조.

## 아키텍처

```
사용자 질문
    ↓
FastAPI (POST /v1/baseball/chat)
    ↓
BaseballAgentService (스트리밍)
    ↓
LangGraph ReAct Agent
    ├── search_player_stats    (pybaseball)
    ├── compare_players        (pybaseball)
    ├── search_game_results    (pybaseball)
    ├── search_statcast        (pybaseball)
    └── ChatResponse           (최종 응답)
```

## 도구 상세

### 1. search_player_stats
- 입력: 선수명, 시즌연도(선택)
- pybaseball: batting_stats() / pitching_stats()
- 타자: AVG, OBP, SLG, OPS, HR, RBI, WAR
- 투수: ERA, WHIP, K/9, W, L, SV, WAR

### 2. compare_players
- 입력: 선수명 2명, 시즌연도(선택)
- 두 선수의 스탯을 나란히 비교

### 3. search_game_results
- 입력: 팀명, 기간(시작일~종료일, 선택)
- pybaseball: schedule_and_record()
- 경기 날짜, 상대팀, 스코어, 승패 반환

### 4. search_statcast
- 입력: 선수명, 기간(선택)
- pybaseball: statcast_batter() / statcast_pitcher()
- 구속, 회전수, 타구속도, 발사각도 등

## 파일 구조

| 파일 | 역할 |
|------|------|
| agents/baseball_tools.py | 4개 pybaseball 도구 정의 |
| agents/baseball_prompts.py | 야구 분석 전용 시스템 프롬프트 |
| agents/baseball_agent.py | 야구 에이전트 그래프 생성 |
| services/baseball_agent_service.py | 에이전트 실행/스트리밍 |
| api/routes/baseball_chat.py | /v1/baseball/chat 엔드포인트 |
| main.py | 라우터 등록 추가 |
