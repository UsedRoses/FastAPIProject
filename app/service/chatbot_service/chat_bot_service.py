import operator
from typing import TypedDict, Annotated

from langchain_community.llms.openai import OpenAIChat
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel

model_chat = ChatDeepSeek(model="deepseek-chat")


tavily_search = TavilySearchResults(max_results=2)
# # 绑定工具
# llm_with_tools = model_chat.bind_tools([tavily_search])

class PictureBook(TypedDict):
    description: str
    style: str
    chapter: int
    aspect_ratio: str
    story_detail: dict
    context: Annotated[list, operator.add]


class StoryDetail(BaseModel):
    """

    """
    title: str
    synopsis: str
    summary: str


class SearchQuery(BaseModel):
    search_query: str


async def create_story_synopsis(state: PictureBook, config: RunnableConfig, store: BaseStore):
    """
    生成故事大纲
    :return:
    """
    user_id = config["configurable"].get("user_id")

    # 调用结构化输出 LLM
    structured_llm = model_chat.with_structured_output(StoryDetail)

    sys_msg = SystemMessage(
        content="You are a story master. Please generate a story outline, title and story summary based on the story description entered by the user.")

    message = HumanMessage(content=f"description: {state.get('description')}")
    print("开始查询输出", message)
    return {"story_detail": await structured_llm.ainvoke([sys_msg] + [message])}

async def search_web(node_state: PictureBook, config: RunnableConfig, store: BaseStore):
    """ 使用 Tavily 搜索资料 """

    # 结构化输出 search_query
    structured_llm = model_chat.with_structured_output(SearchQuery)

    # 提示
    system = SystemMessage(
        content="Your task is to create a concise and effective web search query based on the story title and story synopsis.")
    human = HumanMessage(content=f"Story Title: {node_state['story_detail'].title}. Story Summary: {node_state['story_detail'].synopsis}. Return only the query string.")

    print("故事查询", human)
    # 调用
    result = await structured_llm.ainvoke([system, human])

    if result is None:
        result = await structured_llm.ainvoke([system, human])

    # 用 Tavily 搜索
    tavily_search = TavilySearchResults(max_results=3)
    docs = await tavily_search.ainvoke(result.search_query)

    # 格式化
    formatted = [
        {
            "title": doc["url"],
            "content": doc["content"],
            "url": doc["url"]
        }
        for doc in docs
    ]

    return {"context": formatted}


async def write_story(node_state: PictureBook, config: RunnableConfig, store: BaseStore):
    """
    写故事
    :return:
    """
    print(node_state)
    return node_state

builder = StateGraph(PictureBook)
builder.add_node("create_story_synopsis", create_story_synopsis)
builder.add_node("search_web", search_web)
builder.add_node("write_story", write_story)
# builder.add_node("tools", ToolNode([tavily_search], messages_key="description"))

builder.set_entry_point("create_story_synopsis")
# builder.add_conditional_edges(
#     "create_story_synopsis",
#     tools_condition,
#
# )
# builder.add_edge("tools", "create_story_synopsis")
# builder.add_edge("create_story_synopsis", END)
# builder.add_edge("create_story_synopsis", "search_web")
# builder.add_edge("search_web", "write_story")
# builder.add_edge("write_story", END)

builder.add_edge("create_story_synopsis", "write_story")
builder.add_edge("write_story", END)


# Store for long-term (across-thread) memory
across_thread_memory = InMemoryStore()
# Checkpointer for short-term (within-thread) memory
within_thread_memory = MemorySaver()

chat_graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)