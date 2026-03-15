from typing import Optional

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent as _create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool

from app.agents.baseball_prompts import baseball_system_prompt
from app.agents.baseball_tools import baseball_tools
from app.core.config import settings


@tool
def ChatResponse(message_id: str, content: str, metadata: dict) -> str:
    """최종 응답을 사용자에게 반환할 때 반드시 이 도구를 사용합니다.

    Args:
        message_id: 응답 메시지의 고유 UUID
        content: 사용자 질문에 대한 답변 내용
        metadata: 추가 메타데이터 (빈 dict 가능)
    """
    return "Response delivered."


# 싱글턴 checkpointer (서버 수명 동안 대화 기록 유지)
_checkpointer = MemorySaver()


_agent = None


def build_baseball_agent(
    model_name: Optional[str] = None,
    temperature: float = 0.7,
):
    """MLB 야구 분석 전용 LangGraph ReAct 에이전트를 생성합니다."""
    llm = init_chat_model(
        model_name or settings.OPENAI_MODEL,
        temperature=temperature,
    )

    agent = _create_agent(
        model=llm,
        tools=[*baseball_tools, ChatResponse],
        system_prompt=baseball_system_prompt,
        checkpointer=_checkpointer,
    )

    return agent


def get_baseball_agent():
    """싱글턴 야구 에이전트 인스턴스를 반환합니다."""
    global _agent
    if _agent is None:
        _agent = build_baseball_agent()
    return _agent
