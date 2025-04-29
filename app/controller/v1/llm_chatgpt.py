from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import json

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
