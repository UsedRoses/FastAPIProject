import asyncio
import os

from langchain_deepseek import ChatDeepSeek
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.constants import START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent

model = ChatDeepSeek(model="deepseek-chat", api_key="sk-******")

API_TOKEN = "apify_api_cal5qEfaYuiQc6CINlxBUA0PEg4Tk40AYQah"
ACTOR_NAME = "scrape-creators/best-tiktok-transcripts-scraper"
SSE_URL = f"https://actors-mcp-server.apify.actor/sse?token={API_TOKEN}&actors={ACTOR_NAME}"

async def build_graph():
    async with MultiServerMCPClient({
        "actors-mcp-server": {
                "transport": "sse",
                "url": f"https://actors-mcp-server.apify.actor/sse?token={API_TOKEN}&actors=scrape-creators/best-tiktok-transcripts-scraper",
                "message_endpoint": f"/message?token={API_TOKEN}",
                "headers": {
                    "Authorization": f"Bearer {API_TOKEN}"
                },
        }
    }) as client:
        tools = client.get_tools()
        def call_model(state: MessagesState):
            response = model.bind_tools(tools).invoke(state["messages"])
            return {"messages": response}

        # builder = StateGraph(MessagesState)
        # builder.add_node(call_model)
        # builder.add_node(ToolNode(tools))
        # builder.add_edge(START, "call_model")
        # builder.add_conditional_edges(
        #     "call_model",
        #     tools_condition,
        # )
        # builder.add_edge("tools", "call_model")
        # graph = builder.compile()
        graph = create_react_agent(model, client.get_tools())
        return await graph.ainvoke({
            "messages": "帮我获取这个tiktok视频的字幕, https://www.tiktok.com/@aomen111/video/7473605616871738654"
        })


async def main():
    result = await build_graph()

    # 2. 调用 ainvoke，传入一个 messages 字段

    # 3. 处理或打印结果
    print("字幕结果：")
    print(result)

if __name__ == '__main__':
    asyncio.run(main())