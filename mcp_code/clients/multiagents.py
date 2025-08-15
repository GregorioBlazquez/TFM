from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient, load_mcp_tools
import asyncio, os
from dotenv import load_dotenv

load_dotenv()

MCP_BASE = os.getenv("MCP_BASE", "http://127.0.0.1:8080/mcp/")

# --- LLM for routing ---
llm_router = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_ROUTER_DEPLOYMENT"),
    openai_api_version=os.getenv("AZURE_OPENAI_ROUTER_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ROUTER_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_ROUTER_API_KEY")
)

# --- LLM for main agents ---
llm_agents = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)


async def build_agents(session):
    tools = await load_mcp_tools(session)

    # Filter by prefix
    api_tools = [t for t in tools if t.name.startswith("api_")]
    rag_tools = [t for t in tools if t.name.startswith("rag_")]

    predictor_agent = create_react_agent(llm_agents, api_tools)
    rag_agent = create_react_agent(llm_agents, rag_tools)

    return predictor_agent, rag_agent


# --- Nodes ---
async def run_predictor(state, predictor):
    query = state["query"]
    result = await predictor.ainvoke({"messages": query})
    return {"result": str(result)}

async def run_rag(state, rag):
    query = state["query"]
    result = await rag.ainvoke({"messages": query})
    return {"result": str(result)}


# --- Supervisor ---
async def supervisor(state):
    query = state["query"].lower()

    # Step 1: quick rules (save tokens in clear cases)
    """
    if any(k in query for k in ["turistas", "viajeros", "visitantes", "predicción"]):
        return "predictor"
    if any(k in query for k in ["documento", "informe", "pdf", "reporte"]):
        return "rag"
    """

    # Step 2: LLM for intent classification
    system_prompt = """You are an intent classifier.
    Classify the user's query into one of these categories:
    - predictor: if the question requires prediction of the number of tourists or another numerical model.
    - rag: if the question is about documents, reports, or textual information.
    - other: if it does not fit the above.
    
    Respond ONLY with one word: predictor, rag or other."""
    msg = [{"role": "system", "content": system_prompt},
           {"role": "user", "content": state["query"]}]

    classification = (await llm_router.ainvoke(msg)).content.strip().lower()

    if classification not in ["predictor", "rag"]:
        return END
    return classification


# --- Main loop ---
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
            user_input = input("Question: ")
            if user_input.lower() in ["salir", "exit"]:
                break
            result_state = await app.ainvoke({"query": user_input})
            print("Answer:", result_state["result"])


if __name__ == "__main__":
    asyncio.run(main())
