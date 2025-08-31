
from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
import os
from pydantic import BaseModel
from typing import Dict
import asyncio
import copy
from mcp_code.clients.multiagents import initial_state, get_workflow

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
    workflow_app = await get_workflow()

@app_api.post("/ask")
async def ask(query: Query):
    state = copy.deepcopy(user_states.get(query.user_id, initial_state))
    state["last_query"] = query.question

    if workflow_app is None:
        raise HTTPException(status_code=503, detail="Workflow not ready yet")

    try:
        result_state = await workflow_app.ainvoke(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")

    user_states[query.user_id] = result_state
    return {"answer": result_state.get("result"), "state": result_state}


# Simple test endpoint
@app_api.get("/health")
def health_check():
    return {"status": "ok"}

# Static files (simple frontend)
app_api.mount("/static", StaticFiles(directory=os.path.dirname(__file__)), name="static")

