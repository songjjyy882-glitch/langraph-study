import asyncio
import contextlib
from datetime import datetime
import json
from typing import Optional
import uuid

from app.utils.logger import log_execution, custom_logger
from app.agents.baseball_agent import build_baseball_agent

from langgraph.errors import GraphRecursionError


class BaseballAgentService:
    def __init__(self):
        self.agent = None
        self.progress_queue: asyncio.Queue = asyncio.Queue()

    def _create_agent(self, thread_id: uuid.UUID = None):
        """MLB 야구 분석 에이전트 생성"""
        self.agent = build_baseball_agent()

    @log_execution
    async def process_query(self, user_messages: str, thread_id: uuid.UUID):
        """사용자 쿼리를 처리하고 스트리밍으로 반환합니다."""
        try:
            self._create_agent(thread_id=thread_id)

            custom_logger.info(f"[Baseball] 사용자 메시지: {user_messages}")

            agent_stream = self.agent.astream(
                {"messages": [{"role": "user", "content": user_messages}]},
                config={"configurable": {"thread_id": str(thread_id)}},
                stream_mode="updates",
            )

            agent_iterator = agent_stream.__aiter__()
            agent_task = asyncio.create_task(agent_iterator.__anext__())
            progress_task = asyncio.create_task(self.progress_queue.get())

            while True:
                pending = {agent_task}
                if progress_task is not None:
                    pending.add(progress_task)

                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                if progress_task in done:
                    try:
                        progress_event = progress_task.result()
                        yield json.dumps(progress_event, ensure_ascii=False)
                        progress_task = asyncio.create_task(self.progress_queue.get())
                    except asyncio.CancelledError:
                        progress_task = None
                    except Exception as e:
                        custom_logger.error(f"[Baseball] Error in progress_task: {e}")
                        progress_task = None

                if agent_task in done:
                    try:
                        chunk = agent_task.result()
                    except StopAsyncIteration:
                        agent_task = None
                        break
                    except Exception as e:
                        custom_logger.error(f"[Baseball] Error in agent_task: {e}")
                        import traceback
                        custom_logger.error(traceback.format_exc())
                        agent_task = None
                        error_response = {
                            "step": "done",
                            "message_id": str(uuid.uuid4()),
                            "role": "assistant",
                            "content": "처리 중 오류가 발생했습니다. 다시 시도해주세요.",
                            "metadata": {},
                            "created_at": datetime.utcnow().isoformat(),
                            "error": str(e)
                        }
                        yield json.dumps(error_response, ensure_ascii=False)
                        break

                    custom_logger.info(f"[Baseball] 에이전트 청크: {chunk}")
                    try:
                        for step, event in chunk.items():
                            if not event or not (step in ["model", "agent", "tools"]):
                                continue
                            messages = event.get("messages", [])
                            if len(messages) == 0:
                                continue
                            message = messages[0]
                            if step in ("model", "agent"):
                                tool_calls = message.tool_calls
                                if not tool_calls:
                                    continue
                                tool = tool_calls[0]
                                if tool.get("name") == "ChatResponse":
                                    args = tool.get("args", {})
                                    metadata = args.get("metadata")
                                    custom_logger.info("========================================")
                                    custom_logger.info(args)
                                    yield f'{{"step": "done", "message_id": {json.dumps(args.get("message_id"))}, "role": "assistant", "content": {json.dumps(args.get("content"), ensure_ascii=False)}, "metadata": {json.dumps(self._handle_metadata(metadata), ensure_ascii=False)}, "created_at": "{datetime.utcnow().isoformat()}"}}'
                                    async for _ in agent_iterator:
                                        pass
                                    if progress_task is not None:
                                        progress_task.cancel()
                                        with contextlib.suppress(asyncio.CancelledError):
                                            await progress_task
                                    return
                                else:
                                    yield f'{{"step": "model", "tool_calls": {json.dumps([tool["name"] for tool in tool_calls])}}}'
                            if step == "tools":
                                yield f'{{"step": "tools", "name": {json.dumps(message.name)}, "content": {message.content}}}'
                    except Exception as e:
                        custom_logger.error(f"[Baseball] Error processing chunk: {e}")
                        import traceback
                        custom_logger.error(traceback.format_exc())
                        error_response = {
                            "step": "done",
                            "message_id": str(uuid.uuid4()),
                            "role": "assistant",
                            "content": "데이터 처리 중 오류가 발생했습니다.",
                            "metadata": {},
                            "created_at": datetime.utcnow().isoformat(),
                            "error": str(e)
                        }
                        yield json.dumps(error_response, ensure_ascii=False)
                        break

                    agent_task = asyncio.create_task(agent_iterator.__anext__())

            if progress_task is not None:
                progress_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await progress_task

            while not self.progress_queue.empty():
                try:
                    remaining = self.progress_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                yield json.dumps(remaining, ensure_ascii=False)

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            custom_logger.error(f"[Baseball] Error in process_query: {e}")
            custom_logger.error(error_trace)

            error_response = {
                "step": "done",
                "message_id": str(uuid.uuid4()),
                "role": "assistant",
                "content": "처리 중 오류가 발생했습니다. 다시 시도해주세요.",
                "metadata": {},
                "created_at": datetime.utcnow().isoformat(),
                "error": str(e) if not isinstance(e, GraphRecursionError) else None
            }
            yield json.dumps(error_response, ensure_ascii=False)

    @log_execution
    def _handle_metadata(self, metadata) -> dict:
        result = {}
        if metadata:
            for k, v in metadata.items():
                result[k] = v
        return result
