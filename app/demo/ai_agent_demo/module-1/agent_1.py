from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from langgraph.errors import NodeInterrupt
from langgraph.graph import START, END, StateGraph

"""
失败后从检查点重试
"""

attempts = 0


class State(TypedDict):
    input: str


def step_1(state: State) -> State:
    print("---Step 1---")
    return state


def step_2(state: State) -> State:
    # Let's optionally raise a NodeInterrupt if the length of the input is longer than 5 characters
    global attempts
    attempts += 1

    if attempts < 2:
        raise ValueError("Failure")

    print("---Step 2---")
    return state


def step_3(state: State) -> State:
    print("---Step 3---")
    return state


builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

checkpointer = MemorySaver()

graph = builder.compile(checkpointer=checkpointer)

config = {
    "configurable": {
        "thread_id": "1"  # Unique identifier to track workflow execution
    }
}

try:
    # First invocation will raise an exception due to the `get_info` task failing
    graph.invoke({'any_input': 'foobar'}, config=config)
except ValueError:
    print("ValueError")

print("retry")
graph.invoke(None, config=config)