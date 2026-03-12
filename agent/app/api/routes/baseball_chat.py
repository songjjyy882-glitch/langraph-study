import uuid

from app.utils.logger import custom_logger
from fastapi import APIRouter, HTTPException
from app.models.chat import ChatRequest
from app.services.baseball_agent_service import BaseballAgentService
from fastapi.responses import StreamingResponse

baseball_chat_router = APIRouter()


@baseball_chat_router.post("/baseball/chat")
async def post_baseball_chat(request: ChatRequest):
    """
    MLB 야구 분석 에이전트에게 질문합니다.

    ## Request json
    ```json
    {
        "thread_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "message": "오타니 2024시즌 성적 분석해줘"
    }
    ```
    """
    custom_logger.info(f"[Baseball] API Request: {request}")
    try:
        thread_id = getattr(request, "thread_id", uuid.uuid4())

        async def event_generator():
            try:
                yield f'data: {{"step": "model", "tool_calls": ["Planning"]}}\n\n'
                agent_service = BaseballAgentService()
                async for chunk in agent_service.process_query(
                    user_messages=request.message,
                    thread_id=thread_id
                ):
                    yield f"data: {chunk}\n\n"
            except Exception as e:
                import json
                from datetime import datetime
                error_response = {
                    "step": "done",
                    "message_id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": "요청 처리 중 오류가 발생했습니다. 다시 시도해주세요.",
                    "metadata": {},
                    "created_at": datetime.utcnow().isoformat(),
                    "error": str(e)
                }
                yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n"
                custom_logger.error(f"[Baseball] Error in event_generator: {e}")
                import traceback
                custom_logger.error(traceback.format_exc())

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )

    except Exception as e:
        custom_logger.error(f"[Baseball] Error in /baseball/chat (before streaming): {e}")
        import traceback
        custom_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
