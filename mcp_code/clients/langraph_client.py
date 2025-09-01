import os
import asyncio
from config import get_env_var
from langchain_mcp_adapters.client import MultiServerMCPClient, load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain import hub


MCP_BASE = get_env_var("MCP_BASE", "http://127.0.0.1:8080/mcp/")
AZ_DEPLOYMENT = get_env_var("AZURE_OPENAI_DEPLOYMENT")

async def main():
    client = MultiServerMCPClient({
        "tourism": {
            "url": MCP_BASE,
            "transport": "streamable_http"
        }
    })

    # Open a session ONLY for the desired server
    async with client.session("tourism") as session:
        # Load available MCP tools for this session
        tools = await load_mcp_tools(session)
        tourism_tools = [t for t in tools if t.name.startswith("tourism_")]

        # Build the ReAct agent using the loaded tools
        # Load the base ReAct prompt from LangChain Hub
        # prompt = hub.pull("hwchase17/react")

        # Initialize the Azure LLM
        llm = AzureChatOpenAI(
            azure_deployment=AZ_DEPLOYMENT,
            openai_api_version=get_env_var("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=get_env_var("AZURE_OPENAI_ENDPOINT"),
            api_key=get_env_var("AZURE_OPENAI_API_KEY")
        )

        # Create the agent with the LLM and MCP tools
        agent = create_react_agent(llm, tourism_tools)

        # Call the agent with a sample question
        response = await agent.ainvoke({
            "messages": "How many tourists are expected in Cataluña in August 2025?"
        })

        print("Agent response:", response)

if __name__ == "__main__":
    asyncio.run(main())
