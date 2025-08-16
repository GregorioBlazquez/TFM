from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient, load_mcp_tools
import asyncio, os
from dotenv import load_dotenv
import logging

########## ENV VARS ##########
# Load environment variables
load_dotenv()
MCP_BASE = os.getenv("MCP_BASE", "http://127.0.0.1:8080/mcp/")

########## LOGGING ##########
# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("multiagent-client")

########## LANGGRAPH AGENTS ##########
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

# --- Build agents ---
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
    logger.info(f"[Predictor] Input query: {query}")

    # Build messages from history
    messages = []
    for i, text in enumerate(state.get("history", [])):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": text})

    # Add the current query as the last user message
    messages.append({"role": "user", "content": query})

    result = await predictor.ainvoke({"messages": messages})
    answer = extract_answer(result)
    logger.info(f"[Predictor] Raw result: {result}")

    # Save the answer in history
    state.setdefault("history", []).append(answer)

    return {"result": answer}

async def run_rag(state, rag):
    query = state["query"]
    logger.info(f"[RAG] Input query: {query}")

    # Build messages from history
    messages = []
    for i, text in enumerate(state.get("history", [])):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": text})

    # Add the current query as the last user message
    messages.append({"role": "user", "content": query})

    result = await rag.ainvoke({"messages": messages})
    answer = extract_answer(result)
    logger.info(f"[RAG] Raw result: {result}")

    # Save the answer in history
    state.setdefault("history", []).append(answer)

    return {"result": answer}

async def handle_other(state):
    msg = "No agent can handle this query."
    state.setdefault("history", []).append(msg)
    return {"result": msg}

# --- Supervisor ---
async def supervisor(state):
    query = state["query"].lower()
    logger.info(f"[Supervisor] Routing query: {query}")

    # Step 1: quick rules (save tokens in clear cases)
    """
    if any(k in query for k in ["turistas", "viajeros", "visitantes", "predicción"]):
        return "predictor"
    if any(k in query for k in ["documento", "informe", "pdf", "reporte"]):
        return "rag"
    """

    # Initialize history if it doesn't exist
    state["history"] = state.get("history", [])

    # Build prompt with history
    history_text = "\n".join(state["history"])

    # Step 2: LLM for intent classification
    system_prompt = """You are an intent classifier.
    Classify the user's query into one of these categories:
    - predictor: if the question requires prediction of the number of tourists or another numerical model.
    - rag: if the question is about documents, reports, or textual information.
    - other: if it does not fit the above.
    
    Respond ONLY with one word: predictor, rag or other."""

    msg_content = f"Conversation so far:\n{history_text}\nCurrent query:\n{query}" if history_text else query
    msg = [{"role": "system", "content": system_prompt},
           {"role": "user", "content": msg_content}]

    classification = (await llm_router.ainvoke(msg)).content.strip().lower()
    logger.info(f"[Supervisor] Classification result: {classification}")

    # Save query in history
    state["history"].append(query)

    if classification not in ["predictor", "rag"]:
        return "other"

    return classification

########## MAIN CLIENT ##########
def extract_answer(result) -> str:
    """
    Extracts the useful text from the LangGraph agent's response.
    - Looks in result['messages'] for the last AIMessage with content.
    - If not found, returns the result itself as a string.
    """
    try:
        msgs = result.get("messages", [])
        for msg in reversed(msgs):
            if getattr(msg, "type", None) == "ai" and msg.content:
                return msg.content
        # Fallback if there is no AIMessage
        return str(result)
    except Exception as e:
        logger.warning(f"Failed to extract answer: {e}")
        return str(result)


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
        workflow.add_node("other", handle_other)

        workflow.add_conditional_edges(START, supervisor, {
            "predictor": "predictor",
            "rag": "rag",
            "other": "other",
            END: END
        })

        workflow.add_edge("predictor", END)
        workflow.add_edge("rag", END)

        app = workflow.compile()

        state = {"history": []}
        while True:
            user_input = input("Question: ")
            if user_input.lower() in ["salir", "exit"]:
                break
            result_state = await app.ainvoke({**state, "query": user_input})
            state.update(result_state)
            print("Answer:", result_state["result"])


if __name__ == "__main__":
    asyncio.run(main())
