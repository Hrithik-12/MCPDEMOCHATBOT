import asyncio  # Missing import
from dotenv import load_dotenv
from langchain_core.language_models import LLM
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient
import os

async def run_memory_chat():
    """ Run a memory chat with the MCP agent and client. """
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    config_file = "browser_mcp.json"

    print("Initializing MCP Chat...")  # Fixed typo

    client = MCPClient.from_config_file(config_file)
    llm = ChatGroq(model="qwen-qwq-32b")

    agent = MCPAgent(
        llm=llm,
        client=client,  # Fixed: was client==client (comparison instead of assignment)
        max_steps=15,
        memory_enabled=True  # Fixed: was true (lowercase), should be True
    )

    print("<<<<<-----Interactive Chat------->>>>")  # Fixed typo

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting chat...")
                break
            if user_input.lower() == "clear":
                agent.clear_conversation_history()
                print("Conversation cleared.")
                continue
            print("Assistant:", end=" ", flush=True)

            try:
                response = await agent.run(user_input)
                print(response)
            except Exception as e:
                print(f"Error: {e}")
    finally:
        if client and client.sessions:
            await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(run_memory_chat())