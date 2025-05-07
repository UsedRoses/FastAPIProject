# tarot_agent.py
import json
import faiss
import uuid
from datetime import datetime

from langchain_deepseek import ChatDeepSeek
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.checkpoint.memory import MemorySaver
from dataclasses import dataclass, field, fields
from typing import Optional, Any

# ---- 配置类：支持用户记忆隔离 ----
@dataclass(kw_only=True)
class Configuration:
    user_id: str = "default-user"

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: configurable.get(f.name)
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})

# 初始化嵌入模型与 GPT 模型
model_embed = SentenceTransformer("all-MiniLM-L6-v2")
# model_chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
model_chat = ChatDeepSeek(model="deepseek-chat")

# 加载塔罗牌数据与索引
with open("app/service/tarot_deck_service/tarot-images.json", "r", encoding="utf-8") as f:
    card_refs = json.load(f)
    card_refs = card_refs["cards"]
index = faiss.read_index("app/service/tarot_deck_service/tarot.index")

class TarotQuery(BaseModel):
    user_id: str
    question: str


def generate_tarot_prompt(card):
    prompt = f"Please analyze the tarot card '{card['name']}' from the {card['arcana']} deck."

    # Handle optional fields with .get() and check if the value exists before appending
    archetype = card.get('Archetype', None)
    if archetype:
        prompt += f" It represents the archetype of {archetype}."

    keywords = card.get('keywords', [])
    if keywords:
        prompt += f" The card has the following keywords: {', '.join(keywords)}."

    meanings_light = card.get('meanings', {}).get('light', [])
    if meanings_light:
        prompt += f" Its meaning includes the light aspects: {', '.join(meanings_light)}."

    meanings_shadow = card.get('meanings', {}).get('shadow', [])
    if meanings_shadow:
        prompt += f" And the shadow aspects: {', '.join(meanings_shadow)}."

    hebrew_alphabet = card.get('Hebrew Alphabet', None)
    numerology = card.get('Numerology', None)
    elemental = card.get('Elemental', None)
    if hebrew_alphabet or numerology or elemental:
        prompt += f" It is associated with the Hebrew alphabet: {hebrew_alphabet or 'N/A'}, the number: {numerology or 'N/A'}, and the elemental influence of {elemental or 'N/A'}."

    mythical_spiritual = card.get('Mythical/Spiritual', None)
    if mythical_spiritual:
        prompt += f" Myths and spiritual connections include: {mythical_spiritual}."

    fortune_telling = card.get('fortune_telling', [])
    if fortune_telling:
        prompt += f" Additionally, the fortune telling advice for this card includes: {', '.join(fortune_telling)}."

    return prompt


# 节点 1：向量检索相关塔罗牌并交给 GPT 解读
def tarot_reasoner(state: MessagesState, config: RunnableConfig, store: BaseStore):
    cfg = Configuration.from_runnable_config(config)
    user_id = cfg.user_id
    question = state["messages"][-1].content

    # 读取用户历史记忆
    namespace = ("tarot_history", user_id)
    memories = store.get(namespace, "tarot")
    if memories:
        history = memories.value
    else:
        history = None

    print(history)

    # 使用向量索引查找相关牌义
    vector = model_embed.encode([question]).astype("float32")
    D, I = index.search(vector, k=3)
    top_cards = [card_refs[i] for i in I[0]]

    print(top_cards)

    card_texts = "\n\n".join([
        f"Card {i + 1}:\n{generate_tarot_prompt(card)}"
        for i, card in enumerate(top_cards)
    ])

    print(card_texts)

    system_prompt = f"""
你是一位智慧的塔罗牌解读师。
结合用户过往的占卜记录, 记录可能为空:
{history}

这次用户的问题是：{question}
以下是我们从牌义库中检索出的 3 张相关塔罗牌：
{card_texts}
请基于这些牌义为用户提供详细的占卜解读。
"""

    messages = [SystemMessage(content=system_prompt)]
    response = model_chat.invoke(messages)
    return {"messages": [HumanMessage(content=question), response]}

# 节点 2：记录问答到用户记忆
def record_tarot_result(state: MessagesState, config: RunnableConfig, store: BaseStore):
    cfg = Configuration.from_runnable_config(config)
    user_id = cfg.user_id
    question = state["messages"][-2].content
    answer = state["messages"][-1].content

    # 1. 让 AI 对答案进行总结
    summary_prompt = f"""
    请总结以下问答内容，并用适合 AI 后续问答使用的格式表达：
    ---
    问题: {question}
    回答: {answer}
    ---
    请输出一个简洁结构化的总结，包含关键点、推理过程或结论，适合被 AI 进一步分析或回答。
    """

    messages = [SystemMessage(content=summary_prompt)]
    response = model_chat.invoke(messages)
    summary = response.content
    store.put(("tarot_history", user_id), "tarot", {"summary": summary})
    return {"messages": state["messages"]}

# 构建 LangGraph
builder = StateGraph(MessagesState, config_schema=Configuration)
builder.add_node("tarot_reasoner", tarot_reasoner)
builder.add_node("record_result", record_tarot_result)
builder.set_entry_point("tarot_reasoner")
builder.add_edge("tarot_reasoner", "record_result")
builder.add_edge("record_result", END)

# Store for long-term (across-thread) memory
across_thread_memory = InMemoryStore()

# Checkpointer for short-term (within-thread) memory
within_thread_memory = MemorySaver()

graph = builder.compile(store=across_thread_memory)


