from deepagents import create_deep_agent
from langgraph.graph.state import CompiledStateGraph
from models.llm_models import CustomModelName
import os
from utils.llm_utils import get_model
from deepagents.backends.filesystem import FilesystemBackend
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool


@tool
def query_extensiondock_plugin(plugin_id: str) -> str:
    """
    核心业务 API：必须使用此工具来查询 ExtensionDock 平台上的插件最新状态。
    传入参数为插件的 ID (如 "9527")。
    """
    # 这里我们打印一下，方便你在终端看到真实的 Python 代码被触发了！
    print(f"\n[🚀 触发真实动作] 正在向 ExtensionDock 内部数据库请求 ID: {plugin_id} 的数据...\n")

    # 模拟 API 接口返回的 JSON 数据
    return f'{{"id": "{plugin_id}", "name": "AI 提效助手", "status": "运行中", "daily_active_users": 12500, "version": "v2.1.0"}}'

def build_custom_deep_agent() -> CompiledStateGraph:
    """
    构建基于自定义大模型底座的 Deep Agent
    """
    llm = get_model(CustomModelName.CUSTOM)

    # 2. 获取绝对路径 (对应官方的 /Users/user/{project})
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前是 app/agents
    app_root_dir = os.path.abspath(os.path.join(current_dir, ".."))  # 退回 app 目录

    # 3. 拼接 skills 的绝对路径，并严格按照官方习惯加上结尾斜杠 "/"
    skills_dir = os.path.join(app_root_dir, "skills") + "/"

    print(f"📁 挂载 Root 目录: {app_root_dir}")
    print(f"🎯 加载 Skills 目录: {skills_dir}")

    # 4. 实例化官方后端和 Checkpointer
    backend = FilesystemBackend(root_dir=app_root_dir)
    checkpointer = MemorySaver()

    agent_graph = create_deep_agent(
        model=llm,
        backend=backend,
        skills=[skills_dir],
        checkpointer=checkpointer,
        tools=[query_extensiondock_plugin],
    )

    return agent_graph

custom_deep_graph = build_custom_deep_agent()


# curl -N -X POST "http://localhost:8000/api/v1/agent/extension-deep-agent/stream" \
#      -H "Content-Type: application/json" \
#      -d '{
#            "message": "what is langraph? Use the langgraph-docs skill if available.",
#            "params": {}
#          }'