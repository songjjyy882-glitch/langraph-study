from typing import Optional

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent as _create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    ToolRetryMiddleware,
    ModelCallLimitMiddleware,
)
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from app.agents.prompts import system_prompt
from app.agents.baseball_tools import baseball_tools
from app.core.config import settings


class ChatResponseFormat(BaseModel):
    """에이전트의 구조화된 최종 응답 포맷"""
    message_id: str = Field(description="응답 메시지의 고유 UUID")
    content: str = Field(description="사용자 질문에 대한 답변 내용")
    metadata: dict = Field(default_factory=dict, description="추가 메타데이터")


# 싱글턴 checkpointer (서버 수명 동안 대화 기록 유지)
_checkpointer = MemorySaver()


_agent = None


def build_agent(
    model_name: Optional[str] = None,
    temperature: float = 0,
):
    """LangGraph ReAct 에이전트를 생성합니다."""
    llm = init_chat_model(
        model_name or settings.OPENAI_MODEL,
        temperature=temperature,
    )

    agent = _create_agent(
        model=llm,
        tools=baseball_tools,
        system_prompt=system_prompt,
        response_format=ToolStrategy(ChatResponseFormat),
        middleware=[
            SummarizationMiddleware(
                model="gpt-4o-mini",
                trigger=("tokens", 4000),
                keep=("messages", 10),
            ),
            ToolRetryMiddleware(max_retries=2, backoff_factor=2.0),
            ModelCallLimitMiddleware(
                run_limit=min(settings.DEEPAGENT_RECURSION_LIMIT, 5),
                exit_behavior="end",
            ),
        ],
        checkpointer=_checkpointer,
    )

    return agent


def get_agent():
    """싱글턴 에이전트 인스턴스를 반환합니다."""
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent
