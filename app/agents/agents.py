from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel

from agents.deep_agent_demo import custom_deep_graph
from models.agent_schema import AgentInfo

DEFAULT_AGENT = "faceless-agent"

# 类型别名，用于处理 LangGraph 中不同的 agent 模式
# - @entrypoint 装饰的函数返回 Pregel
# - StateGraph().compile() 返回 CompiledStateGraph
AgentGraph = CompiledStateGraph | Pregel


@dataclass
class Agent:
    description: str
    graph: AgentGraph


agents: dict[str, Agent] = {
    "extension-deep-agent": Agent(
        description="搭载官方 Deep Agents 框架的超级智能体，基于 oapi.extensiondock.com 定制模型，支持 SKILL.md 动态加载与渐进式披露。",
        graph=custom_deep_graph
    )
}


def get_agent(agent_id: str) -> AgentGraph:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
