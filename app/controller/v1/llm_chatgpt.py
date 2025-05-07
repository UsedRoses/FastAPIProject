from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import json

from langchain_core.messages import HumanMessage

from service.llm_service.chatgpt_service import gpt_sse_stream
from service.tarot_deck_service.tarot_deck import TarotQuery, graph

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
def tarot_agent(query: TarotQuery):
    config = {"configurable": {"user_id": query.user_id}}

    async def event_generator():
        async for step in graph.astream(
                {"messages": [HumanMessage(content=query.question)]},
                config=config,
                stream_mode="messages"  # 可以试试 values / updates / debug 等模式
        ):
            # 示例：只提取 token 内容
            if isinstance(step, tuple):
                chunk, metadata = step
                yield f"data: {chunk.content}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")