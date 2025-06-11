import uuid
import asyncio
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.store.redis.aio import AsyncRedisStore
from langgraph.store.base import BaseStore

load_dotenv()

model = ChatDeepSeek(model="deepseek-chat")

DB_URI = "redis://default:1q2w3e@localhost:6379/0"

async def main():
    async with (
        AsyncRedisStore.from_conn_string(DB_URI) as store,
        AsyncRedisSaver.from_conn_string(DB_URI) as checkpointer,
    ):
        # await store.setup()
        # await checkpointer.asetup()

        async def call_model(
            state: MessagesState,
            config: RunnableConfig,
            *,
            store: BaseStore,
        ):
            user_id = config["configurable"]["user_id"]
            namespace = ("memories", user_id)
            memories = await store.asearch(namespace, query=str(state["messages"][-1].content))
            info = "\n".join([d.value["data"] for d in memories])
            system_msg = f"You are a helpful assistant talking to the user. User info: {info}"

            # Store new memories if the user asks the model to remember
            last_message = state["messages"][-1]
            if "remember" in last_message.content.lower():
                memory = "User name is Bob"
                await store.aput(namespace, str(uuid.uuid4()), {"data": memory})

            response = await model.ainvoke(
                [{"role": "system", "content": system_msg}] + state["messages"]
            )
            return {"messages": response}

        builder = StateGraph(MessagesState)
        builder.add_node(call_model)
        builder.add_edge(START, "call_model")

        graph = builder.compile(
            checkpointer=checkpointer,
            store=store,
        )

        config = {
            "configurable": {
                "thread_id": "1",
                "user_id": "1",
            }
        }
        async for chunk in graph.astream(
            {"messages": [{"role": "user", "content": "Hi!"}]},
            config,
            stream_mode="values",
        ):
            chunk["messages"][-1].pretty_print()

        config = {
            "configurable": {
                "thread_id": "2",
                "user_id": "1",
            }
        }

        async for chunk in graph.astream(
            {"messages": [{"role": "user", "content": "what is my name?"}]},
            config,
            stream_mode="values",
        ):
            chunk["messages"][-1].pretty_print()

        graph.checkpointer.delete_thread("1")

if __name__ == "__main__":
    asyncio.run(main())