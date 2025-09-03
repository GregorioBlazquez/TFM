import asyncio
import os
import json
import logging
from typing import TypedDict, List, Optional, Dict, Any
import uuid

from fastapi import Depends
from fastapi import FastAPI, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import get_env_var, load_environment
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient, load_mcp_tools
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

########## LOGGING CONFIGURATION ##########
logging.basicConfig(
    level=logging.INFO,   # root level: INFO (no DEBUG noise)
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

# Keep debug for our app
logging.getLogger("multiagent-client").setLevel(logging.DEBUG)

# Quiet noisy libraries
for noisy in (
    "httpx",
    "httpcore",
    #"mcp.client",                  # MCP client core
    #"mcp.client.streamable_http",  # SSE/stream client
    "uvicorn.access",
    "uvicorn.error",
    "asyncio"
):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("multiagent-client")

########## ENVIRONMENT VARIABLES ##########
load_environment()

# Check loaded enviroment variables
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT", 
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_ROUTER_DEPLOYMENT",
    "AZURE_OPENAI_REASONING_DEPLOYMENT",
    "MCP_BASE"
]

missing_vars = [var for var in required_vars if not get_env_var(var)]
if missing_vars:
    logger.error(f"❌ Missing required environment variables: {missing_vars}")
    raise ValueError(f"Missing environment variables: {missing_vars}")
logger.info("✓ All required environment variables are present")

# Load environment variables
MCP_BASE = get_env_var("MCP_BASE", "http://127.0.0.1:8080/mcp/")
AZURE_OPENAI_API_KEY = get_env_var("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = get_env_var("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = get_env_var("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = get_env_var("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_ROUTER_DEPLOYMENT = get_env_var("AZURE_OPENAI_ROUTER_DEPLOYMENT")
AZURE_OPENAI_REASONING_DEPLOYMENT = get_env_var("AZURE_OPENAI_REASONING_DEPLOYMENT")

########## LLMs and conversation state ##########
class AgentState(TypedDict):
    messages: List[BaseMessage]
    last_query: str
    last_agent_output: Optional[dict]
    last_intent: str
    report_agent_input: Optional[str]
    agents_to_call: Optional[List[str]]
    collected_outputs: Optional[Dict[str, object]]
    next_node: str
    result: str

llm_router = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_ROUTER_DEPLOYMENT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY
)
llm_agents = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY
)
llm_reasoning = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_REASONING_DEPLOYMENT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY
)

SUPERVISOR_PROMPT_FALLBACK = """You are an intent classifier. Return JSON ONLY, example: {"intent":"reports","agents":["predictor","rag"]}"""

########## AGENTS ##########
# --- Build agent instances ---
async def build_agents(session):
    tools = await load_mcp_tools(session)

    # Get prompts from MCP server
    logger.info("🔄 Loading prompts from MCP server...")
    rag_prompt = await session.get_prompt("rag_agent_prompt")
    report_prompt = await session.get_prompt("report_agent_prompt")
    supervisor_prompt = await session.get_prompt("supervisor_prompt")
    predictor_prompt = await session.get_prompt("predictor_prompt")

    # Use loaded supervisor prompt if present, otherwise fallback
    global SUPERVISOR_PROMPT_FALLBACK
    if supervisor_prompt and supervisor_prompt.messages:
        SUPERVISOR_PROMPT_FALLBACK = supervisor_prompt.messages[0].content.text

    # Filter tools by prefix
    api_tools = [t for t in tools if t.name.startswith("api_")]
    rag_tools = [t for t in tools if t.name.startswith("rag_")]

    # Create react agents with tools and prompts
    predictor_agent = create_react_agent(llm_agents, api_tools,
                                    prompt=predictor_prompt.messages[0].content.text if predictor_prompt and predictor_prompt.messages else "")
    rag_agent = create_react_agent(llm_agents, rag_tools,
                                  prompt=rag_prompt.messages[0].content.text if rag_prompt and rag_prompt.messages else "")
    reports_agent = create_react_agent(llm_reasoning, [],
                                      prompt=report_prompt.messages[0].content.text if report_prompt and report_prompt.messages else "")

    return predictor_agent, rag_agent, reports_agent

# --- Agent Nodes ---
async def run_predictor(state: AgentState, predictor):
    query = state["last_query"]
    logger.info(f"[Predictor] Processing query: {query}")

    # Initialize messages list if not exists
    state.setdefault("messages", [])
    
    # Add user query to conversation history
    state["messages"].append(HumanMessage(content=query))
    
    # Invoke predictor agent with complete message history
    result = await predictor.ainvoke({"messages": state["messages"]})
    answer = extract_answer(result)
    logger.info(f"[Predictor] Raw result: {result}")

    # Add agent response to conversation history
    state["messages"].append(AIMessage(content=answer))
    
    # Store raw output for other agents
    state["last_agent_output"] = result
    
    return {
        "result": answer,
        "last_agent_output": result,
        "messages": state["messages"]
    }

async def run_rag(state: AgentState, rag):
    query = state["last_query"]
    logger.info(f"[RAG] Processing query: {query}")

    # Initialize messages list if not exists
    state.setdefault("messages", [])
    
    # Add user query to conversation history
    state["messages"].append(HumanMessage(content=query))
    
    # Invoke RAG agent with complete message history
    result = await rag.ainvoke({"messages": state["messages"]})
    answer = extract_answer(result)
    logger.info(f"[RAG] Raw result: {result}")

    # Add agent response to conversation history
    state["messages"].append(AIMessage(content=answer))
    
    # Store raw output for other agents
    state["last_agent_output"] = result
    
    return {
        "result": answer,
        "last_agent_output": result,
        "messages": state["messages"]
    }

async def run_reports(state: AgentState, reports):
    logger.info(f"[Reports] Starting report generation...")

    # Initialize messages list if not exists
    state.setdefault("messages", [])
    if state.get("report_agent_input"):
        logger.info("[Reports] Using prepared context")
        messages = [{"role": "system", "content": state["report_agent_input"]}]
    else:
        query = state["last_query"]
        logger.info(f"[Reports] Using standard query: {query}")
        state["messages"].append(HumanMessage(content=query))
        messages = state["messages"]
    logger.info(f"[Reports] Messages sent to LLM: {len(messages)}")
    result = await reports.ainvoke({"messages": messages})
    answer = extract_answer(result)
    logger.info(f"[Reports] Raw result: {result}")
    state["messages"].append(AIMessage(content=answer))
    state["last_agent_output"] = result
    if "report_agent_input" in state:
        del state["report_agent_input"]
    return {
        "result": answer,
        "last_agent_output": result,
        "messages": state["messages"]
    }

async def prepare_for_reports(state: AgentState):
    logger.info(f"[Prep Reports] Preparing context for report agent...")

    user_question = state["last_query"]
    raw_data = state.get("last_agent_output", {})  # now will be a dict of outputs

    # Create comprehensive context for report agent
    system_message = f"""
    You are a Tourism Analytics Assistant. Your task is to EXPLAIN and INTERPRET the data provided.

    USER'S QUESTION:
    "{user_question}"

    RAW DATA TO ANALYZE:
    {json.dumps(raw_data, default=str, indent=2)}

    Provide a clear, concise explanation in English. Focus on:
    1. Interpreting the numerical results or cluster assignments
    2. Explaining patterns or anomalies
    3. Providing context-aware insights
    4. Relating findings to tourism industry context

    Do not call any tools - use only reasoning and analysis.
    """
    return {
        "report_agent_input": system_message,
        "result": state.get("result") or "[prepared context]"
    }

async def handle_other(state: AgentState):
    logger.info("[Other] Unsupported query")
    msg = "I'm sorry, I cannot handle this type of query with my current capabilities."
    state.setdefault("messages", [])
    state["messages"].append(AIMessage(content=msg))
    state["last_agent_output"] = {"content": msg}
    return {
        "result": msg,
        "last_agent_output": {"content": msg},
        "messages": state["messages"]
    }

# --- Supervisor: returns intent and agents_to_call (writes into state) ---
async def supervisor(state: AgentState):
    query = state["last_query"]
    logger.info(f"[Supervisor] Routing query: {query}")

    # Build conversation history from messages
    conversation_history = ""
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            conversation_history += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            conversation_history += f"Assistant: {msg.content}\n"

    # Prepare classification prompt
    classification_prompt = [
        {"role": "system", "content": SUPERVISOR_PROMPT_FALLBACK},
        {"role": "user", "content": conversation_history + f"User: {query}\n"}
    ]

    raw_response = await llm_router.ainvoke(classification_prompt)
    resp_text = ""
    try:
        # try to access content robustly
        resp_text = raw_response.content.strip() if hasattr(raw_response, "content") else str(raw_response)
    except Exception:
        resp_text = str(raw_response)

    logger.debug(f"[Supervisor] raw LLM response: {resp_text}")

    # Attempt to parse JSON. Fallback to heuristics.
    intent = "other"
    agents: List[str] = []

    try:
        parsed = json.loads(resp_text)
        intent = parsed.get("intent", "other").lower()
        agents = [a.lower() for a in parsed.get("agents", [])]
    except Exception:
        # fallback heuristic
        t = resp_text.lower()
        if "reports" in t or "explain" in t or "why" in t or "interpret" in t:
            intent = "reports"
            # conservative: ask both if ambiguous
            agents = ["predictor", "rag"]
        elif "egatur" in t or "frontur" in t or "pdf" in t or "document" in t:
            intent = "rag"
            agents = ["rag"]
        else:
            intent = "predictor"
            agents = ["predictor"]

    # Normalize agents (only allow predictor/rag)
    agents = [a for a in agents if a in ("predictor", "rag")]

    state["agents_to_call"] = agents
    state["last_intent"] = intent

    logger.info(f"[Supervisor] intent={intent}, agents_to_call={agents}")

    # For routing we set next_node to the intent name (StateGraph maps those to dispatch)
    return intent

# --- Supervisor node wrapper (sets next_node) ---
async def supervisor_node(state: AgentState):
    intent = await supervisor(state)
    return {
        "next_node": intent,
        "last_intent": intent,
        "result": state.get("result"),
        "agents_to_call": state.get("agents_to_call")
    }


# --- Prepare reports node ---
# Prepares the context for the report agent by consolidating outputs from predictor/rag
async def prepare_reports_node(s: AgentState):
    return await prepare_for_reports(s)

# Dispatcher: executes the requested agents in parallel and consolidates outputs
# This node runs the agents requested by the supervisor (predictor/rag), collects their outputs,
# and prepares the state for the report agent. It always routes to prepare_reports.
import asyncio

async def dispatch_node(s: AgentState, predictor, rag):
    logger.info("[Dispatch] Running requested agents (parallel)...")
    agents = s.get("agents_to_call") or []
    collected: Dict[str, object] = {}
    s.setdefault("messages", [])

    tasks = {}
    if "predictor" in agents:
        tasks["predictor"] = asyncio.create_task(run_predictor(s, predictor))
    if "rag" in agents:
        tasks["rag"] = asyncio.create_task(run_rag(s, rag))

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    last_result = None
    for agent_name, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            logger.error(f"[Dispatch] Error running {agent_name}: {result}", exc_info=True)
            continue
        collected[agent_name] = result.get("last_agent_output") or result.get("result")
        if result.get("result"):
            last_result = result["result"]

    # Persist collected outputs
    s["collected_outputs"] = collected
    s["last_agent_output"] = collected
    s["agents_to_call"] = None
    s["next_node"] = "prepare_reports"
    if last_result:
        s["result"] = last_result

    logger.info(f"[Dispatch] collected outputs keys: {list(collected.keys())}, next_node={s['next_node']}")
    return {
        "result": s.get("result") or "[dispatched]",
        "last_agent_output": s["last_agent_output"],
        "messages": s.get("messages"),
        "next_node": s["next_node"],
        "collected_outputs": s["collected_outputs"],
        "agents_to_call": s.get("agents_to_call")
    }

async def done_node(s):
    # Terminal node that simply returns the accumulated result (for predictor-only or rag-only queries)
    return {
        "result": s.get("result") or "[done]",
        "last_agent_output": s.get("last_agent_output"),
        "messages": s.get("messages"),
        "collected_outputs": s.get("collected_outputs")
    }

########## HELPER: extract text answer from agent result ##########
def extract_answer(result) -> str:
    try:
        messages = result.get("messages", [])
        for msg in reversed(messages):
            # adapt to possible shapes
            if hasattr(msg, "type") and msg.type == "ai" and hasattr(msg, "content"):
                return msg.content
            # fallback if msg is mapping-like
            if isinstance(msg, dict) and msg.get("role") in ("assistant", "ai") and msg.get("content"):
                return msg["content"]
        return str(result)
    except Exception as e:
        logger.warning(f"Failed to extract answer: {e}")
        return str(result)
    
def serialize_value(v: Any) -> Any:
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, list):
        return [serialize_value(i) for i in v]
    if isinstance(v, dict):
        return {k: serialize_value(val) for k, val in v.items()}
    if isinstance(v, (HumanMessage, AIMessage, BaseMessage)):
        return {"role": getattr(v, "type", "message"), "content": getattr(v, "content", str(v))}
    try:
        return json.loads(json.dumps(v, default=str))
    except Exception:
        return str(v)

########## FASTAPI APP ##########
backend_app = FastAPI(title="Multi-agent API")

# Frontend HTML and static files
backend_app.mount("/static", StaticFiles(directory=os.path.dirname(__file__)), name="static")
logger.info("Static files mounted at /static/chat.html")

@backend_app.get("/chat")
async def root():
    return FileResponse("api/client/chat.html")

# Health check
@backend_app.get("/health")
def health_check():
    return {"status": "ok"}

# Pydantic query
class Query(BaseModel):
    query: str

# Session storage
sessions: Dict[str, AgentState] = {}
sessions_lock = asyncio.Lock()

########## STARTUP ##########
@backend_app.on_event("startup")
async def startup_event():
    logger.info("Starting app: initializing MCP client and agents...")
    client = MultiServerMCPClient({"main": {"url": MCP_BASE, "transport": "streamable_http"}})
    session_ctx = client.session("main")
    session = await session_ctx.__aenter__()

    backend_app.state.mcp_client = client
    backend_app.state.mcp_session_ctx = session_ctx
    backend_app.state.mcp_session = session

    predictor, rag, reports = await build_agents(session)
    backend_app.state.predictor = predictor
    backend_app.state.rag = rag
    backend_app.state.reports = reports

    workflow = StateGraph(state_schema=AgentState)

    # Define async node wrappers
    async def predictor_node(s):
        return await run_predictor(s, predictor)

    async def rag_node(s):
        return await run_rag(s, rag)

    async def reports_node(s):
        return await run_reports(s, reports)

    async def prepare_reports_node(s):
        return await prepare_for_reports(s)
    
    async def dispatch_wrapper(s):
        return await dispatch_node(s, predictor, rag)

    # Add workflow nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("dispatch", dispatch_wrapper)
    workflow.add_node("predictor", predictor_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("prepare_reports", prepare_reports_node)
    workflow.add_node("reports", reports_node)
    workflow.add_node("done", done_node)
    workflow.add_node("other", handle_other)

    # Edges as in bash_multiagents.py
    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next_node", "other"),
        {"predictor": "dispatch", "rag": "dispatch", "reports": "dispatch", "other": "other"}
    )
    workflow.add_edge("dispatch", "prepare_reports")
    workflow.add_edge("prepare_reports", "reports")
    workflow.add_edge("predictor", END)
    workflow.add_edge("rag", END)
    workflow.add_edge("reports", END)
    workflow.add_edge("done", END)
    workflow.add_edge("other", END)

    backend_app.state.workflow_app = workflow.compile()
    logger.info("Startup complete")

########## LOGIN ##########
USERS = {"demo": "password123"}

class LoginRequest(BaseModel):
    username: str
    password: str

@backend_app.post("/login")
async def login(req: LoginRequest):
    if USERS.get(req.username) != req.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    session_id = str(uuid.uuid4())
    async with sessions_lock:
        sessions[session_id] = AgentState(
            messages=[],
            last_query="",
            last_agent_output=None,
            last_intent="",
            report_agent_input=None,
            agents_to_call=None,
            collected_outputs=None,
            next_node="",
            result=""
        )
    return {"session_id": session_id}


# ########## ASK ENDPOINT (with session persistence) ##########
@backend_app.post("/ask")
async def ask(query: Query, x_session_id: str = Header(default=None)):
    session_id = x_session_id
    logger.info(f"[ASK] Received request: query='{query.query}' session_id='{session_id}'")

    if not query.query.strip():
        logger.warning("[ASK] Empty query received")
        raise HTTPException(status_code=400, detail="Empty query")
    if not session_id:
        logger.warning("[ASK] Missing session_id header")
        raise HTTPException(status_code=400, detail="Missing session_id header")

    async with sessions_lock:
        state = sessions.get(session_id)
        if not state:
            logger.info(f"[ASK] Creating new session for session_id={session_id}")
            state = AgentState(
                messages=[],
                last_query="",
                last_agent_output=None,
                last_intent="",
                report_agent_input=None,
                agents_to_call=None,
                collected_outputs=None,
                next_node="",
                result=""
            )
            sessions[session_id] = state
        else:
            logger.info(f"[ASK] Found existing session for session_id={session_id}")

    # Update the current query
    state["last_query"] = query.query

    compiled_app = getattr(backend_app.state, "workflow_app", None)
    if compiled_app is None:
        logger.error("[ASK] Workflow not initialized")
        raise HTTPException(status_code=500, detail="Workflow not initialized")

    try:
        logger.info(f"[ASK] Invoking workflow for session_id={session_id}")
        result_state = await compiled_app.ainvoke(state)
        logger.info(f"[ASK] Workflow completed for session_id={session_id}")
    except Exception as e:
        logger.error(f"[ASK] Error in workflow for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # Save updated state in the session
    async with sessions_lock:
        sessions[session_id] = result_state
        logger.info(f"[ASK] Updated session state for session_id={session_id}")

    # Serialize before returning
    serialized = {k: serialize_value(v) for k, v in result_state.items()}
    logger.info(f"[ASK] Returning response for session_id={session_id}: keys={list(serialized.keys())}")
    return serialized


# If you want to run with "python api_app.py" for dev:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_app:backend_app", host="0.0.0.0", port=8000, reload=True)
