import os
import asyncio
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
# Load environment variables
load_dotenv('../../.env.test')

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from solana.rpc.api import Client as SolanaClient
from solders.keypair import Keypair

from goat_adapters.langchain import get_on_chain_tools
from goat_wallets.solana import solana
from goat_plugins.jupiter import jupiter, JupiterPluginOptions
from goat_plugins.spl_token import spl_token, SplTokenPluginOptions
from goat_plugins.spl_token.tokens import SPL_TOKENS

# Initialize LLM
llm = ChatDeepSeek(model="deepseek-chat")


def main():
    # Initialize Solana client
    client = SolanaClient(os.getenv("SOLANA_RPC_ENDPOINT"))

    # Initialize regular Solana wallet
    keypair = Keypair.from_base58_string(os.getenv("SOLANA_WALLET_SEED") or "")
    wallet = solana(client, keypair)
    # Get the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Initialize tools with Solana wallet
    tools = get_on_chain_tools(
        wallet=wallet,
        plugins=[
            jupiter(JupiterPluginOptions()),  # No options needed for Jupiter v6
            spl_token(SplTokenPluginOptions(
                network="mainnet",  # Using devnet as specified in .env
                tokens=SPL_TOKENS
            )),
        ],
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, handle_parsing_errors=True, verbose=True
    )

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        try:
            response = agent_executor.invoke(
                {
                    "input": user_input,
                }
            )

            print("\nAssistant:", response["output"])
        except Exception as e:
            print("\nError:", str(e))


if __name__ == "__main__":
    main()
    # from solders.keypair import Keypair
    # import base58
    # import json
    #
    # with open("wallet-keypair.json", "r") as f:
    #     secret = json.load(f)
    #
    # secret_bytes = bytes(secret)
    # base58_key = base58.b58encode(secret_bytes).decode()
    #
    # print("Set this as your SOLANA_WALLET_SEED env:")
    # print(base58_key)