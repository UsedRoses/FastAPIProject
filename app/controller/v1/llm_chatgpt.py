from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import json

from langchain_core.messages import HumanMessage, AIMessageChunk
from langsmith import traceable
from pydantic import BaseModel

from service.chatbot_service.chat_bot_service import chat_graph
from service.llm_service.chatgpt_service import gpt_sse_stream

router = APIRouter(prefix="/api/v1/chatgpt", tags=["llm"])


@router.post("/prompt")
async def prompt(request: Request):
    data = await request.json()
    return StreamingResponse(
        gpt_sse_stream(data.get("prompt")),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 如果你前面有 Nginx
        }
    )


@router.post("/tarot")
def tarot_agent(query):
    config = {"configurable": {"user_id": query.user_id}}
    # 假设 graph.nodes, graph.edges 是你编译后获得的结构
    nodes = [{"id": n.name, "label": n.name} for n in chat_graph.compiled_nodes]
    edges = [{"source": e.src, "target": e.dst} for e in chat_graph.compiled_edges]
    return {"nodes": nodes, "edges": edges}

class PictureBookDTO(BaseModel):
    user_id: str
    description: str
    style: str
    chapter: int
    aspect_ratio: str

@router.post("/chat-story")
@traceable
def chat_story(query: PictureBookDTO):
    config = {"configurable": {"thread_id": query.user_id, "user_id": query.user_id, "run_name": "chat_story", "tags": ["story"]}}

    async def event_generator():
        async for step in chat_graph.astream(
                query.model_dump(),
                config=config,
                stream_mode=["updates", "debug", "messages", "custom"], # 可以试试 values / updates / debug 等模式
                debug=False
        ):
            # 示例：只提取 token 内容
            print("step 内容", step)
            # —— 更新事件 —— #
            if isinstance(step, dict) and "type" not in step:
                node, out = next(iter(step.items()))
                event = {"type": "update", "node": node, "payload": out}

            # —— tuple 事件 —— #
            elif isinstance(step, tuple) and len(step) == 2:
                tag, body = step

                # 1) debug 任务级别信息，body 是 dict
                if isinstance(body, dict) and body.get("type") == "task":
                    ev = body
                    event = {
                        "type": "debug",
                        "node": ev["payload"]["name"],
                        "step": ev["step"],
                        "timestamp": ev["timestamp"],
                    }

                # 2) LLM token 流，body 是 (AIMessageChunk, metadata_dict)
                elif (
                        isinstance(body, tuple)
                        and isinstance(body[0], AIMessageChunk)
                        and isinstance(body[1], dict)
                ):
                    chunk, meta = body
                    event = {
                        "type": "message",
                        "node": meta.get("langgraph_node"),
                        "token": chunk.content,
                        "step": meta.get("langgraph_step"),
                    }

                # 3) 自定义流，body 也是 (dict, metadata_dict)
                elif (
                        isinstance(body, tuple)
                        and isinstance(body[0], dict)
                        and isinstance(body[1], dict)
                        # 排除 message 的元数据：ls_provider 字段
                        and "ls_provider" not in body[1]
                ):
                    data, meta = body
                    event = {
                        "type": "custom",
                        "node": meta.get("langgraph_node"),
                        "payload": data,
                    }

                else:
                    # checkpoint、其他 debug 类型等都跳过
                    continue

            else:
                # 其它类型都跳过
                continue

            # 发送 SSE
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")