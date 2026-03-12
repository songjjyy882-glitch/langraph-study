from typing import Optional

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent as _create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool

from app.agents.prompts import system_prompt
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


def build_agent(
    model_name: Optional[str] = None,
    temperature: float = 0.7,
):
    """LangGraph ReAct 에이전트를 생성합니다."""
    llm = ChatOpenAI(
        model=model_name or settings.OPENAI_MODEL,
        temperature=temperature,
        api_key=settings.OPENAI_API_KEY,
    )

    agent = _create_agent(
        model=llm,
        tools=[ChatResponse],
        system_prompt=system_prompt,
        checkpointer=_checkpointer,
    )

    return agent
