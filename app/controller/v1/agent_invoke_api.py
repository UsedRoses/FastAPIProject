from typing import Any
from fastapi import APIRouter
from langchain_core.messages import AIMessage
from starlette.responses import StreamingResponse
from agents.agents import DEFAULT_AGENT, get_agent, AgentGraph
from models.exception import ServiceException
from utils.agent_util import handle_input, langchain_to_chat_message, message_generator
from models.agent_schema import UserInput, ChatMessage, StreamInput

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


@router.post("/{agent_id}/invoke")
@router.post("/invoke")
async def invoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    """
    调用 agent 并使用用户输入获取最终响应。

    如果未提供 agent_id，将使用默认 agent。
    使用 thread_id 可以在多轮对话中保持上下文。
    run_id 参数会附加到消息中，用于记录反馈。
    使用 user_id 可以在多个线程间保持会话连续性。
    """
    # 注意：目前此方法只返回最后一条消息或中断。
    # 如果 agent 输出了多条 AIMessages（例如在 interrupt-agent 的后台步骤，
    # 或 research-assistant 的工具步骤），这些消息会被忽略。
    # 理论上，你可能希望包含所有消息，这时可以更新 API 返回 ChatMessages 列表。
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id = await handle_input(user_input, agent)

    try:
        response_events: list[tuple[str, Any]] = await agent.ainvoke(**kwargs, stream_mode=["updates", "values"])  # type: ignore # fmt: skip
        response_type, response = response_events[-1]
        if response_type == "values":
            # Normal response, the agent completed successfully
            output = langchain_to_chat_message(response["messages"][-1])
        elif response_type == "updates" and "__interrupt__" in response:
            # The last thing to occur was an interrupt
            # Return the value of the first interrupt as an AIMessage
            output = langchain_to_chat_message(
                AIMessage(content=response["__interrupt__"][0].value)
            )
        else:
            raise ServiceException(f"Unexpected response type: {response_type}")

        output.run_id = str(run_id)
        return output
    except Exception as e:
        raise ServiceException("Unexpected error")


@router.post(
    "/{agent_id}/stream",
    response_class=StreamingResponse,
)
@router.post("/stream", response_class=StreamingResponse)
async def stream(user_input: StreamInput, agent_id: str = DEFAULT_AGENT) -> StreamingResponse:
    """
    对用户输入流式返回 agent 的响应，包括中间消息和生成的 token。

    如果未提供 agent_id，将使用默认 agent。
    使用 thread_id 可以在多轮对话中保持上下文。
    run_id 参数会附加到所有消息中，用于记录反馈。
    使用 user_id 可以在多个线程间保持会话连续性。

    设置 `stream_tokens=False` 可仅返回中间消息，而不按 token 逐步输出。
    """
    return StreamingResponse(
        message_generator(user_input, agent_id),
        media_type="text/event-stream",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )