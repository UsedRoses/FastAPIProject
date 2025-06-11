from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langchain_community.tools.tavily_search import TavilySearchResults
# from IPython.display import Image, display
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


"""
实现AI 的记忆功能
"""
llm = ChatDeepSeek(model="deepseek-chat")


# 工具1 乘法
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# 工具2 网页数据查询
tavily_search = TavilySearchResults(max_results=2)

# 绑定工具
llm_with_tools = llm.bind_tools([multiply, tavily_search])


# Node
def tool_calling_llm(state: MessagesState):
    # 添加系统提示词
    sys_msg = SystemMessage(content="You are a helpful assistant tasked with writing performing arithmetic on a set of inputs.")
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# 图
builder = StateGraph(MessagesState)

# 节点
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply, tavily_search]))

# 线
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition,
)
builder.add_edge("tools", END)

# 记忆管理
memory = MemorySaver()
# Compile graph
graph = builder.compile(checkpointer=memory)

# View
# display(Image(graph.get_graph().draw_mermaid_png()))

messages = [HumanMessage(content="你好，你知道有关特朗普的生平吗!请只回答我是否知道即可")]

config = {"configurable": {"thread_id": "1"}}

for chunk in graph.stream(
    {"messages": messages},
    config,
    stream_mode="updates"
):
    print(chunk)
    print("\n")



messages = [HumanMessage(content="那么请你帮我在网络上搜索与他相关的信息")]

for chunk in graph.stream(
    {"messages": messages},
    config,
    stream_mode="updates",
):
    print(chunk)
    print("\n")


