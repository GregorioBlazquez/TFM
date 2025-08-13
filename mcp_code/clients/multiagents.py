from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient, load_mcp_tools
import asyncio, os
from dotenv import load_dotenv

load_dotenv()

MCP_BASE = os.getenv("MCP_BASE", "http://127.0.0.1:8080/mcp/")

async def build_agents(session):
    tools = await load_mcp_tools(session)

    # Filtrar por prefijo
    tourism_tools = [t for t in tools if t.name.startswith("tourism_")]
    rag_tools = [t for t in tools if t.name.startswith("rag_")]

    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )

    predictor_agent = create_react_agent(llm, tourism_tools)
    rag_agent = create_react_agent(llm, rag_tools)

    return predictor_agent, rag_agent

# Funciones de los nodos
async def run_predictor(state, predictor):
    query = state["query"]
    result = await predictor.ainvoke({"messages": query})
    return {"result": str(result)}

async def run_rag(state, rag):
    query = state["query"]
    result = await rag.ainvoke({"messages": query})
    return {"result": str(result)}

async def supervisor(state):
    query = state["query"].lower()
    if "turistas" in query:
        return "predictor"
    elif "documento" in query or "informe" in query:
        return "rag"
    else:
        return END

async def main():
    client = MultiServerMCPClient({
        "main": {"url": MCP_BASE, "transport": "streamable_http"}
    })

    async with client.session("main") as session:
        predictor, rag = await build_agents(session)

        workflow = StateGraph(state_schema=dict)

        async def predictor_node(s):
            return await run_predictor(s, predictor)

        async def rag_node(s):
            return await run_rag(s, rag)

        workflow.add_node("predictor", predictor_node)
        workflow.add_node("rag", rag_node)

        workflow.add_conditional_edges(START, supervisor, {
            "predictor": "predictor",
            "rag": "rag",
            END: END
        })

        workflow.add_edge("predictor", END)
        workflow.add_edge("rag", END)

        app = workflow.compile()

        while True:
            user_input = input("Pregunta: ")
            if user_input.lower() in ["salir", "exit"]:
                break
            result_state = await app.ainvoke({"query": user_input})
            print("Respuesta:", result_state["result"])


if __name__ == "__main__":
    asyncio.run(main())
