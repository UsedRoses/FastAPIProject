import inspect
import json
from typing import Any, AsyncGenerator
from uuid import UUID, uuid4

from fastapi import HTTPException
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Interrupt

from agents.agents import AgentGraph, DEFAULT_AGENT, get_agent
from models.agent_schema import UserInput, ChatMessage, StreamInput


async def handle_input(user_input: UserInput, agent: AgentGraph) -> tuple[dict[str, Any], UUID]:
    """
    Parse user input and handle any required interrupt resumption.
    Returns kwargs for agent invocation and the run_id.
    """
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    user_id = user_input.user_id or str(uuid4())

    configurable = {"thread_id": thread_id, "model": user_input.model, "user_id": user_id}

    callbacks = []

    if user_input.agent_config:
        if overlap := configurable.keys() & user_input.agent_config.keys():
            raise HTTPException(
                status_code=422,
                detail=f"agent_config contains reserved keys: {overlap}",
            )
        configurable.update(user_input.agent_config)

    config = RunnableConfig(
        configurable=configurable,
        run_id=run_id,
        callbacks=callbacks,
    )

    # 检查需要回复的中断
    if agent.checkpointer is not None:
        state = await agent.aget_state(config=config)
        interrupted_tasks = [
            task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts
        ]
    else:
        interrupted_tasks = []

    input: Command | dict[str, Any]
    if interrupted_tasks:
        # assume user input is response to resume agent execution from interrupt
        input = Command(resume=user_input.message)
    else:
        input = {"messages": [HumanMessage(content=user_input.message)]}
        params = user_input.params or {}
        input.update(params)

    kwargs = {
        "input": input,
        "config": config,
    }


    return kwargs, run_id


def convert_message_content_to_string(content: str | list[str | dict]) -> str:
    if isinstance(content, str):
        return content
    text: list[str] = []
    for content_item in content:
        if isinstance(content_item, str):
            text.append(content_item)
            continue
        if content_item["type"] == "text":
            text.append(content_item["text"])
    return "".join(text)

def langchain_to_chat_message(message: BaseMessage, LangchainChatMessage=None) -> ChatMessage:
    """Create a ChatMessage from a LangChain message."""
    match message:
        case HumanMessage():
            human_message = ChatMessage(
                type="human",
                content=convert_message_content_to_string(message.content),
            )
            return human_message
        case AIMessage():
            ai_message = ChatMessage(
                type="ai",
                content=convert_message_content_to_string(message.content),
            )
            if message.tool_calls:
                ai_message.tool_calls = message.tool_calls
            if message.response_metadata:
                ai_message.response_metadata = message.response_metadata
            return ai_message
        case ToolMessage():
            tool_message = ChatMessage(
                type="tool",
                content=convert_message_content_to_string(message.content),
                tool_call_id=message.tool_call_id,
            )
            return tool_message
        case LangchainChatMessage():
            if message.role == "custom":
                custom_message = ChatMessage(
                    type="custom",
                    content="",
                    custom_data=message.content[0],
                )
                return custom_message
            else:
                raise ValueError(f"Unsupported chat message role: {message.role}")
        case _:
            raise ValueError(f"Unsupported message type: {message.__class__.__name__}")


def remove_tool_calls(content: str | list[str | dict]) -> str | list[str | dict]:
    """Remove tool calls from content."""
    if isinstance(content, str):
        return content
    # Currently only Anthropic models stream tool calls, using content item type tool_use.
    return [
        content_item
        for content_item in content
        if isinstance(content_item, str) or content_item["type"] != "tool_use"
    ]


async def message_generator(
        user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    agent = get_agent(agent_id)
    kwargs, run_id = await handle_input(user_input, agent)

    try:
        # 处理来自图的流式事件，并通过 SSE 流逐步输出消息。
        async for stream_event in agent.astream(
                **kwargs, stream_mode=["updates", "messages", "custom"], subgraphs=True
        ):
            if not isinstance(stream_event, tuple):
                continue

            # 遇到不同子图产生的事件时，用不同的方式去处理它们的结构。
            if len(stream_event) == 3:
                # 当 subgraphs=True 时：会返回 (node_path, stream_mode, event)
                _, stream_mode, event = stream_event
            else:
                # 当 subgraphs=False 时：会返回 (stream_mode, event)
                stream_mode, event = stream_event

            new_messages = []

            if stream_mode == "updates":
                for node, updates in event.items():
                    # 1. 处理人机交互 / Agent 中断
                    if node == "__interrupt__":
                        for interrupt in updates:
                            # 兼容 interrupt.value
                            new_messages.append(AIMessage(content=interrupt.value))
                        continue

                    updates = updates or {}
                    raw_messages = updates.get("messages", [])

                    # 兼容 LangGraph 的 Overwrite, RemoveMessage 等修饰器对象
                    if hasattr(raw_messages, "value"):
                        parsed_messages = raw_messages.value
                    else:
                        parsed_messages = raw_messages

                    # 2. 确保 update_messages 绝对是一个 list，防止 extend 和 [-1] 报错
                    if not isinstance(parsed_messages, list):
                        if parsed_messages is not None and parsed_messages != "":
                            update_messages = [parsed_messages]
                        else:
                            update_messages = []
                    else:
                        # 浅拷贝，防止修改原始数据
                        update_messages = list(parsed_messages)

                    print(f"[{node}] 的消息: {update_messages}")

                    # 使用 langgraph-supervisor 库的特殊情况
                    if node == "supervisor":
                        # 只获取最后一条 ToolMessage
                        if update_messages and isinstance(update_messages[-1], ToolMessage):
                            update_messages = [update_messages[-1]]
                        else:
                            update_messages = []

                    if node in ("research_expert", "math_expert"):
                        update_messages = []

                    new_messages.extend(update_messages)

            if stream_mode == "custom":
                new_messages = [event]

            # LangGraph 的流式输出可能会产生元组: (field_name, field_value)
            processed_messages = []
            current_message: dict[str, Any] = {}
            for message in new_messages:
                if isinstance(message, tuple):
                    key, value = message
                    current_message[key] = value
                else:
                    if current_message:
                        processed_messages.append(_create_ai_message(current_message))
                        current_message = {}
                    processed_messages.append(message)

            if current_message:
                processed_messages.append(_create_ai_message(current_message))

            # 推送完整的结构化消息
            for message in processed_messages:
                try:
                    chat_message = langchain_to_chat_message(message)
                    chat_message.run_id = str(run_id)
                except Exception as e:
                    # 遇到无法解析的数据结构时，静默跳过或报警，避免整个流中断
                    print(f"结构化消息解析失败跳过: {e}")
                    continue

                # LangGraph 会重新发送输入的消息, 这部分内容忽略
                if chat_message.type == "human" and chat_message.content == user_input.message:
                    continue

                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

            # 推送 Token 碎片（打字机效果）
            if stream_mode == "messages":
                if not user_input.stream_tokens:
                    continue
                msg, metadata = event
                if "skip_stream" in metadata.get("tags", []):
                    continue

                # 由于某些原因，astream("messages") 会导致非 LLM 节点发送额外的消息。忽略掉
                if not isinstance(msg, AIMessageChunk):
                    continue

                content = remove_tool_calls(msg.content)
                if content:
                    # 在 OpenAI 的上下文中，内容为空通常表示模型在请求调用某个工具。只打印非空内容。
                    yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"

    except Exception as e:
        import traceback
        traceback.print_exc()  # 打印完整堆栈，方便日后排查类似错误
        yield f"data: {json.dumps({'type': 'error', 'content': f'Internal server error: {str(e)}'})}\n\n"
    finally:
        yield "data: [DONE]\n\n"


def _create_ai_message(parts: dict) -> AIMessage:
    sig = inspect.signature(AIMessage)
    valid_keys = set(sig.parameters)
    filtered = {k: v for k, v in parts.items() if k in valid_keys}
    return AIMessage(**filtered)

