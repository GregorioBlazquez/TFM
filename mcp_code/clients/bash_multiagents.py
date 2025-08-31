import asyncio
import os
import json
import logging
from typing import TypedDict, List, Optional, Dict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient, load_mcp_tools
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

########## ENVIRONMENT VARIABLES ##########
# Load environment variables
load_dotenv()
MCP_BASE = os.getenv("MCP_BASE", "http://127.0.0.1:8080/mcp/")

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

########## LANGGRAPH AGENTS ##########
# --- State schema for our multi-agent system ---
class AgentState(TypedDict):
    # Complete conversation history using LangChain messages
    messages: List[BaseMessage]
    # Last user question for processing
    last_query: str
    # Raw output from the last executed agent (for context sharing)
    last_agent_output: Optional[dict]
    # Last intent classification from supervisor
    last_intent: str
    # Prepared input for report agent (optional)
    report_agent_input: Optional[str]
    # For routing and results
    agents_to_call: Optional[List[str]]
    collected_outputs: Optional[Dict[str, object]]
    next_node: str
    result: str

# --- LLM for routing/intent classification ---
llm_router = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_ROUTER_DEPLOYMENT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# --- LLM for main agents (predictor, RAG) ---
llm_agents = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# --- LLM for reasoning/report agent ---
llm_reasoning = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_REASONING_DEPLOYMENT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# fallback supervisor prompt (use MCP stored prompt in build_agents when possible)
SUPERVISOR_PROMPT_FALLBACK = """
You are an intent classifier. Return JSON ONLY, example:
{"intent":"reports","agents":["predictor","rag"]}
See the project instructions for mapping user queries to agents.
"""

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

########## MAIN APPLICATION LOOP ##########
async def main():
    client = MultiServerMCPClient({
        "main": {"url": MCP_BASE, "transport": "streamable_http"}
    })

    async with client.session("main") as session:
        # Build agent instances
        predictor, rag, reports = await build_agents(session)

        # Define workflow with StateGraph
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

        # Dispatcher: executes the requested agents in sequence and consolidates outputs
        async def dispatch_node(s: AgentState):
            logger.info("[Dispatch] Running requested agents...")
            agents = s.get("agents_to_call") or []
            collected: Dict[str, object] = {}
            s.setdefault("messages", [])

            last_result = None
            for ag in agents:
                if ag == "predictor":
                    out = await run_predictor(s, predictor)
                elif ag == "rag":
                    out = await run_rag(s, rag)
                else:
                    logger.warning(f"[Dispatch] Unknown agent requested: {ag}")
                    continue

                # out is a dict with keys result,last_agent_output,messages
                collected[ag] = out.get("last_agent_output") or out.get("result")
                last_result = out.get("result") or last_result

            # Persist collected outputs
            s["collected_outputs"] = collected
            s["last_agent_output"] = collected

            # Clear agents_to_call so it doesn't leak into next turn
            s["agents_to_call"] = None

            # Always send to reports (even if only predictor or rag was used)
            s["next_node"] = "prepare_reports"

            # Keep raw result in state (reports agent will reformat)
            if last_result:
                s["result"] = last_result

            """
            intent = s.get("last_intent", "other")
            if intent == "reports":
                s["next_node"] = "prepare_reports"
            else:
                s["next_node"] = "done"

            # If this was predictor-only or rag-only, return the real agent result
            if last_result:
                s["result"] = last_result
            """

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
        
        # Add nodes
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("dispatch", dispatch_node)
        workflow.add_node("predictor", predictor_node)
        workflow.add_node("rag", rag_node)
        workflow.add_node("prepare_reports", prepare_reports_node)
        workflow.add_node("reports", reports_node)
        workflow.add_node("done", done_node)
        workflow.add_node("other", handle_other)

        # Edges
        workflow.add_edge(START, "supervisor")

        # Supervisor -> dispatch for predictor/rag/reports. 'other' goes to other.
        workflow.add_conditional_edges(
            "supervisor",
            lambda state: state.get("next_node", "other"),
            {
                "predictor": "dispatch",
                "rag": "dispatch",
                "reports": "dispatch",
                "other": "other",
            }
        )

        # Dispatch always leads to prepare_reports
        workflow.add_edge("dispatch", "prepare_reports")

        """
        # Dispatch -> either prepare_reports (if reports) or done (terminal)
        workflow.add_conditional_edges(
            "dispatch",
            lambda state: state.get("next_node", "done"),
            {
                "prepare_reports": "prepare_reports",
                "done": "done",
            }
        )
        """

        # prepare_reports -> reports
        workflow.add_edge("prepare_reports", "reports")

        # terminal edges
        workflow.add_edge("predictor", END)
        workflow.add_edge("rag", END)
        workflow.add_edge("reports", END)
        workflow.add_edge("done", END)
        workflow.add_edge("other", END)

        app = workflow.compile()

        initial_state: AgentState = {
            "messages": [],
            "last_query": "",
            "last_agent_output": None,
            "last_intent": "",
            "report_agent_input": None,
            "agents_to_call": None,
            "collected_outputs": None,
            "next_node": "",
            "result": ""
        }

        # Main interaction loop
        while True:
            try:
                user_input = input("Question: ").strip()
                if user_input.lower() in ["exit", "quit", "salir"]:
                    break
                if not user_input:
                    continue

                # Update state with new user input
                initial_state["last_query"] = user_input

                # Process query through workflow
                result_state = await app.ainvoke(initial_state)

                # Update persistent state
                for k, v in result_state.items():
                    if v is not None:
                        initial_state[k] = v
                
                # Display result to user
                print("---")
                print(result_state)
                print("Answer:", result_state["result"])
                print("---")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                print("Sorry, I encountered an error. Please try again.")

if __name__ == "__main__":
    asyncio.run(main())