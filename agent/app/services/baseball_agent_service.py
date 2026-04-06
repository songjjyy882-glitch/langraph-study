import asyncio
import contextlib
from datetime import datetime
import json
import uuid

from app.utils.logger import log_execution, custom_logger
from app.agents.graph import get_baseball_graph
from app.core.opik_tracer import opik_tracer

from langgraph.errors import GraphRecursionError


class BaseballAgentService:
    def __init__(self):
        self.progress_queue: asyncio.Queue = asyncio.Queue()

    @log_execution
    async def process_query(self, user_messages: str, thread_id: uuid.UUID):
        """사용자 쿼리를 LangGraph 커스텀 그래프로 처리하고 스트리밍으로 반환합니다."""
        try:
            graph = get_baseball_graph()

            custom_logger.info(f"[Baseball] 사용자 메시지: {user_messages}")

            agent_stream = graph.astream(
                {"messages": [{"role": "user", "content": user_messages}]},
                config={
                    "configurable": {"thread_id": str(thread_id)},
                    "callbacks": [opik_tracer],
                },
                stream_mode="updates",
            )

            agent_iterator = agent_stream.__aiter__()
            agent_task = asyncio.create_task(agent_iterator.__anext__())
            progress_task = asyncio.create_task(self.progress_queue.get())
            got_final_response = False

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

                    custom_logger.info(f"[Baseball] 그래프 청크: {chunk}")
                    try:
                        for node_name, event in chunk.items():
                            if not event:
                                continue

                            # 라우터 노드: 질문 유형 분류 결과 스트리밍
                            if node_name == "router":
                                query_type = event.get("query_type", "")
                                node_label = {
                                    "realtime_stats": "⚾ pybaseball 실시간 조회",
                                    "historical": "📊 Elasticsearch 역대 기록 검색",
                                    "game_results": "🏟️ pybaseball 경기 결과 조회",
                                    "general": "💬 일반 답변 (검색 없음)",
                                    "complex": "🧠 Deep Agent 다단계 분석",
                                }.get(query_type, query_type)
                                yield f'{{"step": "model", "tool_calls": ["{node_label}"]}}'

                            # 검색 노드: tool 호출 결과 스트리밍
                            elif node_name in ("realtime_node", "historical_node", "game_node"):
                                search_results = event.get("search_results", "")
                                if search_results and search_results != "검색 결과 없음":
                                    # search_results를 파싱해서 tool name 추출
                                    tool_names = []
                                    for line in search_results.split("\n\n"):
                                        if line.startswith("[") and "]" in line:
                                            tool_name = line[1:line.index("]")]
                                            tool_names.append(tool_name)
                                    if tool_names:
                                        yield f'{{"step": "model", "tool_calls": {json.dumps(tool_names)}}}'

                                    # tool 결과를 개별 스트리밍
                                    for line in search_results.split("\n\n"):
                                        if line.startswith("[") and "]" in line:
                                            tool_name = line[1:line.index("]")]
                                            tool_content = line[line.index("]") + 2:]
                                            try:
                                                parsed = json.loads(tool_content)
                                                yield f'{{"step": "tools", "name": {json.dumps(tool_name)}, "content": {json.dumps(parsed, ensure_ascii=False)}}}'
                                            except json.JSONDecodeError:
                                                yield f'{{"step": "tools", "name": {json.dumps(tool_name)}, "content": {json.dumps(tool_content, ensure_ascii=False)}}}'

                            # Deep Agent 노드: 최종 응답 스트리밍
                            elif node_name == "deep_agent_node":
                                messages = event.get("messages", [])
                                if messages:
                                    last_msg = messages[-1]
                                    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                                    got_final_response = True
                                    yield f'{{"step": "done", "message_id": {json.dumps(str(uuid.uuid4()))}, "role": "assistant", "content": {json.dumps(content, ensure_ascii=False)}, "metadata": {{}}, "created_at": "{datetime.utcnow().isoformat()}"}}'

                                    async for _ in agent_iterator:
                                        pass
                                    if progress_task is not None:
                                        progress_task.cancel()
                                        with contextlib.suppress(asyncio.CancelledError):
                                            await progress_task
                                    return

                            # 분석 노드: 최종 응답 스트리밍
                            elif node_name == "analysis":
                                messages = event.get("messages", [])
                                if messages:
                                    last_msg = messages[-1]
                                    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                                    got_final_response = True
                                    yield f'{{"step": "done", "message_id": {json.dumps(str(uuid.uuid4()))}, "role": "assistant", "content": {json.dumps(content, ensure_ascii=False)}, "metadata": {{}}, "created_at": "{datetime.utcnow().isoformat()}"}}'

                                    # 나머지 chunk 소비
                                    async for _ in agent_iterator:
                                        pass
                                    if progress_task is not None:
                                        progress_task.cancel()
                                        with contextlib.suppress(asyncio.CancelledError):
                                            await progress_task
                                    return

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

            if not got_final_response:
                custom_logger.warning("[Baseball] 그래프가 최종 응답 없이 종료됨")
                fallback_response = {
                    "step": "done",
                    "message_id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": "처리 한도에 도달하여 응답을 완료하지 못했습니다. 질문을 더 구체적으로 해주시거나 다시 시도해주세요.",
                    "metadata": {},
                    "created_at": datetime.utcnow().isoformat(),
                }
                yield json.dumps(fallback_response, ensure_ascii=False)

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
