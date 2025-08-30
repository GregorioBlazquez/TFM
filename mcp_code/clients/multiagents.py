from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient, load_mcp_tools
import asyncio, os
from dotenv import load_dotenv
import logging
from typing import TypedDict, List, Optional
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
    # Needed for routing
    next_node: str
    # Answer to return to user
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

# Global variable for supervisor prompt
SUPERVISOR_PROMPT = ""

# --- Build agent instances ---
async def build_agents(session):
    tools = await load_mcp_tools(session)

    # Get prompts from MCP server
    logger.info("🔄 Loading prompts from MCP server...")
    rag_prompt = await session.get_prompt("rag_agent_prompt")
    report_prompt = await session.get_prompt("report_agent_prompt")
    supervisor_prompt = await session.get_prompt("supervisor_prompt")

    # Save supervisor prompt globally
    global SUPERVISOR_PROMPT
    SUPERVISOR_PROMPT = supervisor_prompt.messages[0].content.text if supervisor_prompt.messages else ""

    # Filter tools by prefix
    api_tools = [t for t in tools if t.name.startswith("api_")]
    rag_tools = [t for t in tools if t.name.startswith("rag_")]

    # Create react agents with tools and prompts
    predictor_agent = create_react_agent(llm_agents, api_tools)
    rag_agent = create_react_agent(llm_agents, rag_tools, 
                                  prompt=rag_prompt.messages[0].content.text if rag_prompt.messages else "")
    reports_agent = create_react_agent(llm_reasoning, [], 
                                      prompt=report_prompt.messages[0].content.text if report_prompt.messages else "")

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
    
    # Prepare messages based on available context
    if "report_agent_input" in state:
        logger.info(f"[Reports] Using prepared context from previous agent")
        # Use prepared system message for context-aware reporting
        messages = [{"role": "system", "content": state["report_agent_input"]}]
    else:
        # Fallback: use standard conversation flow
        query = state["last_query"]
        logger.info(f"[Reports] Using standard query: {query}")
        state["messages"].append(HumanMessage(content=query))
        messages = state["messages"]

    logger.info(f"[Reports] Messages sent to LLM: {len(messages)} messages")

    result = await reports.ainvoke({"messages": messages})
    answer = extract_answer(result)
    logger.info(f"[Reports] Raw result: {result}")

    # Add agent response to conversation history
    state["messages"].append(AIMessage(content=answer))
    
    # Store raw output
    state["last_agent_output"] = result
    
    # Clean up prepared input for next iteration
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
    raw_data = state.get("last_agent_output", {})

    # Create comprehensive context for report agent
    system_message = f"""
    You are a Tourism Analytics Assistant. Your task is to EXPLAIN and INTERPRET the data provided.

    USER'S QUESTION:
    "{user_question}"

    RAW DATA TO ANALYZE:
    {raw_data}

    Provide a clear, concise explanation in English. Focus on:
    1. Interpreting the numerical results
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
    logger.info(f"[Other] Handling unsupported query...")
    
    msg = "I'm sorry, I cannot handle this type of query with my current capabilities."
    
    # Initialize messages list if not exists
    state.setdefault("messages", [])
    
    # Add error response to conversation history
    state["messages"].append(AIMessage(content=msg))
    
    # Store symbolic output
    state["last_agent_output"] = {"content": msg}
    
    return {
        "result": msg, 
        "last_agent_output": {"content": msg},
        "messages": state["messages"]
    }

# --- Supervisor Node ---
async def supervisor(state: AgentState):
    query = state["last_query"].lower()
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
        {"role": "system", "content": SUPERVISOR_PROMPT},
        {"role": "user", "content": conversation_history + f"User: {query}\n"}
    ]

    logger.debug(f"[Supervisor] Classification prompt: {classification_prompt}")

    # Get classification from router LLM
    classification = (await llm_router.ainvoke(classification_prompt)).content.strip().lower()
    logger.info(f"[Supervisor] Classification result: {classification}")

    # Validate and return classification
    if classification not in ["predictor", "rag", "reports"]:
        return "other"
    return classification

# --- Supervisor Node Wrapper ---
async def supervisor_node(state: AgentState):
    intent = await supervisor(state)
    return {
    "next_node": intent,
    "last_intent": intent,
    "result": state.get("result")
}

########## HELPER FUNCTIONS ##########
def extract_answer(result) -> str:
    """
    Extracts the text response from LangGraph agent's result.
    Looks for the last AI message content in result['messages'].
    Fallback: returns string representation of result.
    """
    try:
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == "ai" and hasattr(msg, 'content'):
                return msg.content
        return str(result)
    except Exception as e:
        logger.warning(f"Failed to extract answer: {e}")
        return str(result)

########## MAIN APPLICATION LOOP ##########
async def main():
    # Initialize MCP client
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

        # Add all nodes to workflow
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("predictor", predictor_node)
        workflow.add_node("rag", rag_node)
        workflow.add_node("prepare_reports", prepare_reports_node)
        workflow.add_node("reports", reports_node)
        workflow.add_node("other", handle_other)

        # Define workflow edges
        workflow.add_edge(START, "supervisor")
        
        # Conditional routing based on supervisor's decision
        workflow.add_conditional_edges(
            "supervisor",
            lambda state: state.get("next_node", "other"),
            {
                "predictor": "predictor",
                "rag": "rag", 
                "reports": "prepare_reports",
                "other": "other",
            }
        )

        # Connect preparation to report generation
        workflow.add_edge("prepare_reports", "reports")

        # Define terminal nodes
        workflow.add_edge("predictor", END)
        workflow.add_edge("rag", END)
        workflow.add_edge("reports", END)
        workflow.add_edge("other", END)

        # Compile workflow
        app = workflow.compile()

        # Initialize empty state
        initial_state: AgentState = {
            "messages": [],
            "last_query": "",
            "last_agent_output": None,
            "last_intent": "",
            "report_agent_input": None,
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