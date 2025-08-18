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
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("multiagent-client")

########## LANGGRAPH AGENTS ##########
# --- LLM for routing ---
llm_router = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_ROUTER_DEPLOYMENT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# --- LLM for main agents ---
llm_agents = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# --- LLM for reasoning agent ---
llm_reasoning = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_REASONING_DEPLOYMENT"),
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
    rag_agent = create_react_agent(
        llm_agents,
        rag_tools,
        prompt="You are a RAG assistant. Always use the available tools (rag_rag_search, rag_rag_upsert) to answer questions. Do not answer directly, call the tools first."
    )

    reports_agent = create_react_agent(
        llm_reasoning,
        [],
        prompt="""You are a Reasoning assistant.
            Your role is to:
            - Interpret outputs from other agents (predictions, RAG results).
            - Generate clear explanations, clarifications, and analytical narratives.
            - If the user asks things like "what does it mean?", "summarize", "explain", 
            or requests an interpretation, this is your responsibility.
            Do not call external tools. Only reason and explain."""
        )

    return predictor_agent, rag_agent, reports_agent


# --- Nodes ---
async def run_predictor(state, predictor):
    query = state["query"]
    logger.info(f"[Predictor] Input query: {query}")

    # Initialize user/assistant history
    state.setdefault("user_history", [])
    state.setdefault("assistant_history", [])

    # Append current query to user_history
    state["user_history"].append(query)

    # Build messages alternating user/assistant
    messages = []
    for u, a in zip(state["user_history"], state["assistant_history"]):
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    if len(state["user_history"]) > len(state["assistant_history"]):
        messages.append({"role": "user", "content": query})

    # 🔹 Log exact messages sent to the LLM
    logger.debug(f"[Predictor] Messages sent to LLM:\n{messages}")

    result = await predictor.ainvoke({"messages": messages})
    answer = extract_answer(result)
    logger.info(f"[Predictor] Raw result: {result}")

    # Save answer in assistant_history and last_result
    state["assistant_history"].append(answer)
    state["last_result"] = answer

    return {"result": answer}

async def run_rag(state, rag):
    query = state["query"]
    logger.info(f"[RAG] Input query: {query}")

    state.setdefault("user_history", [])
    state.setdefault("assistant_history", [])
    state["user_history"].append(query)

    messages = []
    for u, a in zip(state["user_history"], state["assistant_history"]):
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    if len(state["user_history"]) > len(state["assistant_history"]):
        messages.append({"role": "user", "content": query})

    # 🔹 Log exact messages sent to the LLM
    logger.info(f"[RAG] Messages sent to LLM:\n{messages}")

    result = await rag.ainvoke({"messages": messages})
    answer = extract_answer(result)
    logger.info(f"[RAG] Raw result: {result}")

    state["assistant_history"].append(answer)
    state["last_result"] = answer

    return {"result": answer}

async def run_reports(state, reports):
    query = state["query"]
    logger.info(f"[Reports] Input query: {query}")

    state["user_history"].append(query)

    messages = []
    for u, a in zip(state["user_history"], state["assistant_history"]):
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    if len(state["user_history"]) > len(state["assistant_history"]):
        messages.append({"role": "user", "content": query})

    # 🔹 Log exact messages sent to the LLM
    logger.info(f"[Reports] Messages sent to LLM:\n{messages}")

    result = await reports.ainvoke({"messages": messages})
    answer = extract_answer(result)
    logger.info(f"[Reports] Raw result: {result}")

    state["assistant_history"].append(answer)
    state["last_result"] = answer

    return {"result": answer}

async def handle_other(state):
    msg = "No agent can handle this query."
    state.setdefault("user_history", [])
    state.setdefault("assistant_history", [])
    state["assistant_history"].append(msg)
    state["last_result"] = msg
    return {"result": msg}

# --- Supervisor ---
async def supervisor(state):
    query = state["query"].lower()
    logger.info(f"[Supervisor] Routing query: {query}")

    state.setdefault("user_history", [])
    state.setdefault("assistant_history", [])

    # Build messages as alternating user/assistant pairs
    messages_text = ""
    for u, a in zip(state["user_history"], state["assistant_history"]):
        messages_text += f"User: {u}\nAssistant: {a}\n"
    # Append current query without duplicar
    messages_text += f"User: {query}\n"

    system_prompt = """
        You are an intent classifier.
        Classify the user's query into one of these categories:
        - predictor: if the question requires prediction of the number of tourists or another numerical model.
        - rag: if the question is about documents, data, this project information, EGATUR, FRONTUR or textual information.
        - reports: if the question requires explanation, reasoning, interpretation, or summaries.
        - other: if it does not fit the above.

        Respond ONLY with one word: predictor, rag, reports or other.
        """

    msg = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": messages_text}
    ]

    # 🔹 Log exact messages sent to the LLM
    logger.debug(f"[Supervisor] Messages sent to LLM:\n{msg}")

    classification = (await llm_router.ainvoke(msg)).content.strip().lower()
    logger.info(f"[Supervisor] Classification result: {classification}")

    # Append current query to user_history
    state["user_history"].append(query)

    if classification not in ["predictor", "rag", "reports"]:
        return "other"
    return classification

########## HELPER ##########
def extract_answer(result) -> str:
    """
    Extracts useful text from LangGraph agent's response.
    Looks in result['messages'] for the last AIMessage with content.
    Fallback: returns str(result)
    """
    try:
        msgs = result.get("messages", [])
        for msg in reversed(msgs):
            if getattr(msg, "type", None) == "ai" and msg.content:
                return msg.content
        return str(result)
    except Exception as e:
        logger.warning(f"Failed to extract answer: {e}")
        return str(result)

########## MAIN LOOP ##########
async def main():
    client = MultiServerMCPClient({
        "main": {"url": MCP_BASE, "transport": "streamable_http"}
    })

    async with client.session("main") as session:
        predictor, rag, reports = await build_agents(session)

        workflow = StateGraph(state_schema=dict)

        async def predictor_node(s):
            return await run_predictor(s, predictor)

        async def rag_node(s):
            return await run_rag(s, rag)
        
        async def reports_node(s):
            return await run_reports(s, reports)

        workflow.add_node("predictor", predictor_node)
        workflow.add_node("rag", rag_node)
        workflow.add_node("reports", reports_node)
        workflow.add_node("other", handle_other)

        workflow.add_conditional_edges(START, supervisor, {
            "predictor": "predictor",
            "rag": "rag",
            "reports": "reports",
            "other": "other",
            END: END
        })

        workflow.add_edge("predictor", END)
        workflow.add_edge("rag", END)
        workflow.add_edge("reports", END)

        app = workflow.compile()

        state = {"user_history": [], "assistant_history": []}
        while True:
            user_input = input("Question: ")
            if user_input.lower() in ["salir", "exit"]:
                break
            result_state = await app.ainvoke({**state, "query": user_input})
            state.update(result_state)
            print("Answer:", result_state["result"])


if __name__ == "__main__":
    asyncio.run(main())
