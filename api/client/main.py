
from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
import os
import httpx
from pydantic import BaseModel
from typing import Dict
import asyncio
import copy
from config import load_environment, get_env_var
load_environment()

from mcp_code.clients.multiagents import initial_state, get_workflow, logger
from mcp_code.clients.bash_multiagents import main as multiagents_main

# FastAPI app
app_api = FastAPI(title="TFM Multi-Agent API")
workflow_app = None

# Request model
class Query(BaseModel):
    question: str
    user_id: str  # allows multi-user handling

# Multi-user simulation (dict by user_id)
user_states: Dict[str, dict] = {}

# Future dependency for authentication
async def get_current_user(user_id: str):
    # Here will be added validation with JWT or token
    return user_id

@app_api.on_event("startup")
async def startup_event():
    global workflow_app

    # Check critical variables
    required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
    for var in required_vars:
        if not get_env_var(var):
            logger.error(f"❌ Missing {var} in FastAPI context")
    
    logger.info("Starting workflow initialization...")
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        workflow_app = await get_workflow()
    logger.info("Workflow initialized successfully")

@app_api.post("/ask")
async def ask(query: Query):
    try:
        logger.info(f"User {query.user_id} asked: {query.question}")
        
        if query.user_id not in user_states:
            logger.info(f"Creating new state for user {query.user_id}")
            user_states[query.user_id] = copy.deepcopy(initial_state)
        
        state = user_states[query.user_id]
        

        # DETAILED DEBUG OF CURRENT STATE
        logger.info(f"=== DEBUG STATE BEFORE ===")
        logger.info(f"Messages count: {len(state.get('messages', []))}")
        for i, msg in enumerate(state.get('messages', [])):
            logger.info(f"Message {i}: {type(msg).__name__} - '{msg.content}'")
        logger.info(f"Last query: '{state.get('last_query', 'None')}'")
        logger.info(f"=== DEBUG STATE END ===")
        
        # COMPLETE RESET for debugging
        logger.warning("RESETTING STATE FOR DEBUGGING")
        state = copy.deepcopy(initial_state)
        user_states[query.user_id] = state
        
        state["last_query"] = query.question
        from langchain_core.messages import HumanMessage
        
        # Add ONLY the new message
        state["messages"].append(HumanMessage(content=query.question))
        
        logger.info(f"State prepared with {len(state.get('messages', []))} messages")
        logger.info(f"Final message: '{state['messages'][0].content}'")
        
        if workflow_app is None:
            logger.error("Workflow not initialized")
            raise HTTPException(status_code=503, detail="Workflow not ready yet")

        logger.info("Invoking workflow...")
        result_state = await workflow_app.ainvoke(state)
        logger.info(f"Workflow completed successfully")
        
        user_states[query.user_id] = result_state
        
        return {"answer": result_state.get("result"), "state": result_state}
        
    except Exception as e:
        logger.exception(f"Error in /ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# Simple test endpoint
@app_api.get("/health")
def health_check():
    return {"status": "ok"}

@app_api.get("/test-openai")
async def test_openai():
    """Test direct connection to Azure OpenAI"""
    from langchain_openai import AzureChatOpenAI
    from langchain_core.messages import HumanMessage
    
    try:
        llm = AzureChatOpenAI(
            azure_deployment=get_env_var("AZURE_OPENAI_DEPLOYMENT"),
            openai_api_version=get_env_var("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=get_env_var("AZURE_OPENAI_ENDPOINT"),
            api_key=get_env_var("AZURE_OPENAI_API_KEY"),
            timeout=30.0
        )
        
        response = await llm.ainvoke([HumanMessage(content="Hello, test message")])
        return {"status": "success", "response": str(response)}
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Static files (simple frontend)
app_api.mount("/static", StaticFiles(directory=os.path.dirname(__file__)), name="static")
logger.info("API initialized and static files mounted available at: http://localhost:8000/static/chat.html")

@app_api.get("/test-predictor")
async def test_predictor():
    """Test the predictor agent specifically"""
    from langchain_core.messages import HumanMessage
    
    try:

        # Create a test state
        test_state = copy.deepcopy(initial_state)
        test_state["last_query"] = "Test prediction query"
        test_state["messages"].append(HumanMessage(content="Test prediction query"))
        
        # Get the predictor from the workflow
        if workflow_app is None:
            return {"status": "error", "error": "Workflow not initialized"}
        
        # Run only the predictor
        from mcp_code.clients.multiagents import run_predictor
        predictor_agent = None  # You would need a way to access the predictor
        
        # Alternative: invoke the full workflow but with detailed logging
        logger.info("Testing predictor through workflow...")
        result = await workflow_app.ainvoke(test_state)
        
        return {"status": "success", "result": result.get("result", "No result")}
        
    except Exception as e:
        logger.exception(f"Test predictor error: {e}")
        return {"status": "error", "error": str(e)}

@app_api.get("/debug-state/{user_id}")
async def debug_state(user_id: str):
    """Debug the current state for a user"""
    if user_id not in user_states:
        return {"status": "no_state", "user_id": user_id}
    
    state = user_states[user_id]
    return {
        "status": "has_state",
        "user_id": user_id,
        "message_count": len(state.get("messages", [])),
        "messages": [str(msg) for msg in state.get("messages", [])],
        "last_query": state.get("last_query", ""),
        "last_intent": state.get("last_intent", "")
    }

@app_api.post("/reset-state/{user_id}")
async def reset_state(user_id: str):
    """Reset the state for a user"""
    if user_id in user_states:
        del user_states[user_id]
    return {"status": "reset", "user_id": user_id}

@app_api.get("/test-predictor-direct")
async def test_predictor_direct():
    """Test the predictor directly with minimal state"""
    try:
        from langchain_core.messages import HumanMessage
        
        # Minimal and clean state
        test_state = {
            "messages": [HumanMessage(content="Give me the number of tourists in Spain in 2025 08")],
            "last_query": "Give me the number of tourists in Spain in 2025 08",
            "last_agent_output": None,
            "last_intent": "",
            "report_agent_input": None,
            "agents_to_call": ["predictor"],
            "collected_outputs": None,
            "next_node": "",
            "result": ""
        }
        
        logger.info("Testing predictor with clean minimal state...")
        
        # We need to access the predictor directly
        # This requires modifying multiagents.py to export the agents
        from mcp_code.clients.multiagents import run_predictor
        
        # For this test, you would need access to the predictor agent
        # As a workaround, use the workflow but with a clean state
        result = await workflow_app.ainvoke(test_state)
        
        return {"status": "success", "result": result.get("result", "No result")}
        
    except Exception as e:
        logger.exception(f"Direct predictor test error: {e}")
        return {"status": "error", "error": str(e)}