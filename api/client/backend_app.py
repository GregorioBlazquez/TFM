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
    level=get_env_var("LOGGIN_LEVEL", logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

# Quiet noisy libraries
for noisy in (
    "httpx",
    "httpcore",
    "mcp.client",                  # MCP client core
    "mcp.client.streamable_http",  # SSE/stream client
    "uvicorn.access",
    "uvicorn.error",
    "asyncio"
):
    # Different logging level noisy libraries
    logging.getLogger(noisy).setLevel(get_env_var("LOGGIN_LEVEL_NOISY", logging.WARNING))

logger = logging.getLogger("multiagent-client")

def safe_log_message(msg: str, max_len=200):
    if len(msg) > max_len:
        return msg[:max_len] + "...[truncated]"
    return msg

# Use custom log function for session-aware logging
def log(session_id: Optional[str], msg: str, level=logging.INFO):
    prefix = f"[session_id={session_id}]" if session_id else ""
    logging.log(level, f"{prefix} {msg}")

########## ENVIRONMENT VARIABLES ##########
load_environment()

# Check loaded enviroment variables
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT", 
    "AZURE_OPENAI_PREDICTOR_DEPLOYMENT",
    "AZURE_OPENAI_RAG_DEPLOYMENT",
    "AZURE_OPENAI_ROUTER_DEPLOYMENT",
    "AZURE_OPENAI_REASONING_DEPLOYMENT",
    "MCP_BASE"
]

missing_vars = [var for var in required_vars if not get_env_var(var)]
if missing_vars:
    log(None, f"❌ Missing required environment variables: {missing_vars}", level=logging.ERROR)
    raise ValueError(f"Missing environment variables: {missing_vars}")
log(None, "✓ All required environment variables are present", level=logging.INFO)

# Load environment variables
MCP_BASE = get_env_var("MCP_BASE", "http://127.0.0.1:8080/mcp/")
AZURE_OPENAI_API_KEY = get_env_var("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = get_env_var("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = get_env_var("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_PREDICTOR_DEPLOYMENT = get_env_var("AZURE_OPENAI_PREDICTOR_DEPLOYMENT")
AZURE_OPENAI_RAG_DEPLOYMENT = get_env_var("AZURE_OPENAI_RAG_DEPLOYMENT")
AZURE_OPENAI_ROUTER_DEPLOYMENT = get_env_var("AZURE_OPENAI_ROUTER_DEPLOYMENT")
AZURE_OPENAI_SUMMARY_DEPLOYMENT = get_env_var("AZURE_OPENAI_SUMMARY_DEPLOYMENT")
AZURE_OPENAI_REASONING_DEPLOYMENT = get_env_var("AZURE_OPENAI_REASONING_DEPLOYMENT")

log(None, f"Loaded env var MCP_BASE={MCP_BASE}", level=logging.DEBUG)
log(None, f"Loaded Azure deployments: predictor={AZURE_OPENAI_PREDICTOR_DEPLOYMENT}, rag={AZURE_OPENAI_RAG_DEPLOYMENT}, router={AZURE_OPENAI_ROUTER_DEPLOYMENT}, reasoning={AZURE_OPENAI_REASONING_DEPLOYMENT}", level=logging.DEBUG)

########## LLMs and conversation state ##########
class AgentState(TypedDict):
    messages: List[BaseMessage]
    context_summary: Optional[str]
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
    api_key=AZURE_OPENAI_API_KEY,
    temperature=0
    #max_tokens=200
)
llm_summary = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_SUMMARY_DEPLOYMENT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    temperature=0
    #max_tokens=200
)
llm_predictor = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_PREDICTOR_DEPLOYMENT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY
)
llm_rag = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_RAG_DEPLOYMENT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    temperature=0
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
    # Filter tools by prefix
    api_tools = [t for t in tools if t.name.startswith("api_")]
    rag_tools = [t for t in tools if t.name.startswith("rag_")]

    log(None, f"Tools loaded from MCP server: {[t.name for t in tools]}", level=logging.DEBUG)
    log(None, f"[Agents] api_tools={len(api_tools)}, rag_tools={len(rag_tools)}", level=logging.DEBUG)

    # Get prompts from MCP server
    log(None, "🔄 Loading prompts from MCP server...", level=logging.INFO)
    rag_prompt = await session.get_prompt("rag_agent_prompt")
    report_prompt = await session.get_prompt("report_agent_prompt")
    supervisor_prompt = await session.get_prompt("supervisor_prompt")
    predictor_prompt = await session.get_prompt("predictor_prompt")

    # Use loaded supervisor prompt if present, otherwise fallback
    global SUPERVISOR_PROMPT_FALLBACK
    if supervisor_prompt and supervisor_prompt.messages:
        SUPERVISOR_PROMPT_FALLBACK = supervisor_prompt.messages[0].content.text

    # Create react agents with tools and prompts
    predictor_agent = create_react_agent(llm_predictor, api_tools,
                                    prompt=predictor_prompt.messages[0].content.text if predictor_prompt and predictor_prompt.messages else "")
    rag_agent = create_react_agent(llm_rag, rag_tools,
                                  prompt=rag_prompt.messages[0].content.text if rag_prompt and rag_prompt.messages else "")
    reports_agent = create_react_agent(llm_reasoning, [],
                                      prompt=report_prompt.messages[0].content.text if report_prompt and report_prompt.messages else "")

    log(None, f"Predictor agent prompt preview: {safe_log_message(predictor_prompt.messages[0].content.text if predictor_prompt else '', 100)}", level=logging.DEBUG)
    log(None, f"RAG agent prompt preview: {safe_log_message(rag_prompt.messages[0].content.text if rag_prompt else '', 100)}", level=logging.DEBUG)
    log(None, f"Reports agent prompt preview: {safe_log_message(report_prompt.messages[0].content.text if report_prompt and report_prompt.messages else '', 100)}", level=logging.DEBUG)

    return predictor_agent, rag_agent, reports_agent

# --- Agent Nodes ---
async def run_predictor(state: AgentState, predictor):
    query = state["last_query"]
    log(state.get("session_id"), f"[Predictor] Processing query: {query}", level=logging.INFO)

    # Build input messages with summary + current user query
    messages = []
    if state.get("context_summary"):
        messages.append({"role": "system", "content": f"Conversation summary:\n{state['context_summary']}"})
    messages.append(HumanMessage(content=query))
    
    # Invoke predictor agent
    log(state.get("session_id"), f"[Predictor] messages before call: {[safe_log_message(str(m.content), 100) for m in state['messages']]}", level=logging.DEBUG)
    result = await predictor.ainvoke({"messages": messages})
    answer = extract_answer(result)
    log(state.get("session_id"), f"[Predictor] Raw result: {safe_log_message(str(result))}", level=logging.INFO)

    # Add user query to conversation history
    state["messages"].append(HumanMessage(content=query))
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
    log(state.get("session_id"), f"[RAG] Processing query: {query}", level=logging.INFO)

    # Build input messages with summary + current user query
    messages = []
    if state.get("context_summary"):
        messages.append({"role": "system", "content": f"Conversation summary:\n{state['context_summary']}"})
    messages.append(HumanMessage(content=query))
    
    # Invoke RAG agent
    log(state.get("session_id"), f"[RAG] messages before call: {[safe_log_message(str(m.content), 100) for m in state['messages']]}", level=logging.DEBUG)
    result = await rag.ainvoke({"messages": messages})
    answer = extract_answer(result)
    log(state.get("session_id"), f"[RAG] Raw result: {safe_log_message(str(result))}", level=logging.INFO)

    # Add user query to conversation history
    state["messages"].append(HumanMessage(content=query))
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
    log(state.get("session_id"), f"[Reports] Starting report generation...", level=logging.INFO)

    # Initialize messages list if not exists
    state.setdefault("messages", [])
    if state.get("report_agent_input"):
        log(state.get("session_id"), "[Reports] Using prepared context", level=logging.INFO)
        messages = [{"role": "system", "content": state["report_agent_input"]}]
    else:
        query = state["last_query"]
        log(state.get("session_id"), f"[Reports] Using standard query: {query}", level=logging.INFO)
        state["messages"].append(HumanMessage(content=query))
        messages = state["messages"]
    
    log(state.get("session_id"), f"[Reports] Messages sent to LLM: {len(messages)}", level=logging.INFO)
    log(state.get("session_id"), f"[Reports] messages before call: {[safe_log_message(str(m.content), 100) for m in state['messages']]}", level=logging.DEBUG)
    result = await reports.ainvoke({"messages": messages})
    answer = extract_answer(result)
    log(state.get("session_id"), f"[Reports] Raw result: {safe_log_message(str(result))}", level=logging.INFO)
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
    log(state.get("session_id"), f"[Prep Reports] Preparing context for report agent...", level=logging.INFO)

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
    log(state.get("session_id"), "[Other] Unsupported query", level=logging.INFO)
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
    log(state.get("session_id"), f"[Supervisor] Routing query: {query}", level=logging.INFO)

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

    log(state.get("session_id"), f"[Supervisor] raw LLM response: {resp_text}", level=logging.DEBUG)

    # Attempt to parse JSON. Fallback to heuristics.
    intent = "other"
    agents: List[str] = []

    try:
        parsed = json.loads(resp_text)
        intent = parsed.get("intent", "other").lower()
        agents = [a.lower() for a in parsed.get("agents", [])]
    except Exception:
        # fallback heuristic
        log(state.get("session_id"), f"[Supervisor] Using fallback heuristic for intent. Raw resp_text: {resp_text}", level=logging.DEBUG)

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

    log(state.get("session_id"), f"[Supervisor] intent={intent}, agents_to_call={agents}", level=logging.INFO)

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
    log(s.get("session_id"), "[Dispatch] Running requested agents...", level=logging.INFO)

    agents = s.get("agents_to_call") or []
    log(s.get("session_id"), f"[Dispatch] Agents to run: {agents}", level=logging.DEBUG)

    # Always include rag unless the intent was "other"
    if s.get("last_intent") != "other" and "rag" not in agents:
        agents.append("rag")

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
            log(s.get("session_id"), f"[Dispatch] Error running {agent_name}: {safe_log_message(str(result))}", level=logging.ERROR)
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

    log(s.get("session_id"), f"[Dispatch] collected outputs keys: {list(collected.keys())}, next_node={s['next_node']}", level=logging.INFO)
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
        log(None, f"Failed to extract answer: {e}", level=logging.WARNING)
        log(None, f"[extract_answer] Could not extract structured answer, returning str(result)", level=logging.DEBUG)
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

async def update_context_summary(state: AgentState, new_user_query: str, new_agent_answer: str):
    """
    Maintain a running summary of the conversation instead of keeping all assistant messages.
    This summary should provide enough context for the predictor or rag agent.
    """
    previous_summary = state.get("context_summary", "")
    system_prompt = f"""
    You are a summarizer. Update the conversation summary given the last user query and assistant answer.

    Previous summary:
    {previous_summary}

    New user query:
    {new_user_query}

    Assistant answer:
    {new_agent_answer}

    Provide an updated concise summary in English.
    """

    try:
        summary_result = await llm_summary.ainvoke([{"role": "system", "content": system_prompt}])
        updated_summary = summary_result.content.strip()
        state["context_summary"] = updated_summary
        log(state.get("session_id"), f"[Summary] Updated summary: {safe_log_message(updated_summary)}", level=logging.INFO)
    except Exception as e:
        log(state.get("session_id"), f"[Summary] Failed to update summary: {e}", level=logging.ERROR)
        # fallback: append raw text
        state["context_summary"] = f"{previous_summary}\nUser: {new_user_query}\nAssistant: {new_agent_answer}"

########## FASTAPI APP ##########
backend_app = FastAPI(title="Multi-agent API")

# Frontend HTML and static files
backend_app.mount("/static", StaticFiles(directory="api/client"), name="static")
log(None, f"Static files mounted at {MCP_BASE}static/chat.html or {MCP_BASE}chat", level=logging.INFO)

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
    log(None, "Starting app: initializing MCP client and agents...", level=logging.INFO)
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
    log(None, "Startup complete", level=logging.INFO)

########## LOGIN ##########
users_env = get_env_var("USERS", "{}")
try:
    USERS = json.loads(users_env)
except json.JSONDecodeError:
    raise ValueError(f"Invalid USERS environment variable: {users_env}")

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
            context_summary="",
            last_query="",
            last_agent_output=None,
            last_intent="",
            report_agent_input=None,
            agents_to_call=None,
            collected_outputs=None,
            next_node="",
            result=""
        )
    log(session_id, f"Login for session_id='{session_id}'", level=logging.INFO)
    return {"session_id": session_id}

########## NEW CHAT ##########
@backend_app.post("/new_chat")
async def new_chat(x_session_id: str = Header(...)):
    async with sessions_lock:
        if x_session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        sessions[x_session_id] = AgentState(
            messages=[],
            context_summary="",
            last_query="",
            last_agent_output=None,
            last_intent="",
            report_agent_input=None,
            agents_to_call=None,
            collected_outputs=None,
            next_node="",
            result=""
        )

    log(x_session_id, f"New chat for session_id='{x_session_id}'", level=logging.INFO)
    return {"status": "ok", "message": "Session reset"}


# ########## ASK ENDPOINT (with session persistence) ##########
@backend_app.post("/ask")
async def ask(query: Query, x_session_id: str = Header(default=None)):
    session_id = x_session_id
    log(session_id, f"[ASK] Received request: query='{query.query}' session_id='{session_id}'", level=logging.INFO)

    if not query.query.strip():
        log(session_id, "[ASK] Empty query received", level=logging.WARNING)
        raise HTTPException(status_code=400, detail="Empty query")
    if not session_id:
        log(session_id, "[ASK] Missing session_id header", level=logging.WARNING)
        raise HTTPException(status_code=400, detail="Missing session_id header")

    async with sessions_lock:
        state = sessions.get(session_id)
        if not state:
            log(session_id, f"[ASK] Creating new session for session_id={session_id}", level=logging.INFO)
            state = AgentState(
                messages=[],
                context_summary="",
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
            log(session_id, f"[ASK] Found existing session for session_id={session_id}", level=logging.INFO)

    # Update the current query
    state["last_query"] = query.query

    compiled_app = getattr(backend_app.state, "workflow_app", None)
    if compiled_app is None:
        logger.error("[ASK] Workflow not initialized")
        raise HTTPException(status_code=500, detail="Workflow not initialized")

    try:
        log(session_id, f"[ASK] Invoking workflow for session_id={session_id}", level=logging.INFO)
        log(session_id, f"[ASK] Current session state before workflow: {serialize_value(state)}", level=logging.DEBUG)
        result_state = await compiled_app.ainvoke(state)
        log(session_id, f"[ASK] Workflow completed for session_id={session_id}", level=logging.INFO)
        log(session_id, f"[ASK] Workflow result state: {serialize_value(result_state)}", level=logging.DEBUG)
    except Exception as e:
        log(session_id, f"[ASK] Error in workflow for session {session_id}: {e}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

    # Save updated state in the session
    async with sessions_lock:
        sessions[session_id] = result_state
        log(session_id, f"[ASK] Updated session state for session_id={session_id}", level=logging.INFO)

    # Update summary in background (fire-and-forget)
    asyncio.create_task(
        update_context_summary(
            result_state, 
            state["last_query"], 
            result_state.get("result", "")
        )
    )

    # Serialize before returning
    serialized = {k: serialize_value(v) for k, v in result_state.items()}
    log(session_id, f"[ASK] Returning response for session_id={session_id}: keys={list(serialized.keys())}", level=logging.INFO)
    return serialized


# If you want to run with "python api_app.py" for dev:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_app:backend_app", host="0.0.0.0", port=8000, reload=True)
