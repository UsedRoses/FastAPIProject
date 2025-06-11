from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from langchain_community.tools.tavily_search import TavilySearchResults
# from IPython.display import Image, display
from dotenv import load_dotenv
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


"""
一个简单的Agent
并进行工具绑定
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
tavily_search = TavilySearchResults(max_results=4)

# 绑定工具
llm_with_tools = llm.bind_tools([multiply, tavily_search])


# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply, tavily_search]))

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition,
)
builder.add_edge("tools", END)

# Compile graph
graph = builder.compile()

# View
# display(Image(graph.get_graph().draw_mermaid_png()))

messages = [HumanMessage(content="请帮我搜索Denote的5个竞品")]
messages = graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()