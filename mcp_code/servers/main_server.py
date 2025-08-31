########## IMPORTS ##########
import os
import asyncio
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from dotenv import load_dotenv
from fastmcp import FastMCP
from mcp_code.servers.rag_server import rag_mcp
from mcp_code.servers.report_server import report_mcp
from api.main import app as fastapi_app
import logging
from fastmcp.server.middleware.timing import DetailedTimingMiddleware
from fastmcp.server.middleware.logging import StructuredLoggingMiddleware
from fastmcp.prompts.prompt import Message

import numpy as np
def log_transform(x):
    return np.log1p(x)

########## ENV VARS ##########
# Load environment variables
load_dotenv()
MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", 8080))

########## LOGGING ##########
# Initialize logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

########## MCP SERVER ##########
# Main MCP server
main_mcp = FastMCP(name="main-mcp-server")

# Log timing (tools/resources/prompts)
#main_mcp.add_middleware(DetailedTimingMiddleware())

# Logging structured JSON of MCP requests (include truncated payloads)
#main_mcp.add_middleware(StructuredLoggingMiddleware(include_payloads=True))

########### PROMPTS ##########
@main_mcp.prompt
def supervisor_prompt():
    return Message("""
    You are an intent classifier for a tourism assistant. 
    For each user query, decide the high-level intent and which internal agents should run BEFORE any reporting.

    Always return a valid JSON object with two fields:
    - "intent": one of "predictor", "rag", "reports", "other"
    - "agents": a list of agent names to execute, any subset of ["predictor","rag"]

    CLASSIFICATION RULES:
    1. "predictor": 
       - Use when the user asks for forecasts, predictions, future values, estimates of tourist arrivals or expenditure, or cluster assignment. 
       - Examples: "How many tourists will visit Spain in 2025?", "Predict the average expenditure", "Assign this profile to a cluster".
       - → Always set agents = ["predictor"].

    2. "rag": 
       - Use when the user asks about official statistics, summaries, or definitions from EGATUR, FRONTUR, INE reports, PDFs, or FAQs. 
       - Includes questions about the project itself or its author, since these are documented in the FAQs. 
       - Examples: "Summarize the EGATUR report", "What is FRONTUR?", "Who is the author of this project?", "Are the data official or generated?".
       - → Always set agents = ["rag"].

    3. "reports": 
       - Use when the user requests a comprehensive analysis, report, explanation, recommendations, or benchmarking. 
       - These require both model outputs and document context. 
       - Examples: "Generate a report about tourism in Spain next year", "Explain the forecast compared to EGATUR", "Give me recommendations for regional tourism policy".
       - → Always set agents = ["predictor","rag"].

    4. "other": 
       - Only use when the query is unrelated to tourism, official reports, forecasts, or project FAQs.
       - Examples: "Tell me a joke", "What is the capital of France?".
       - → Always set agents = [].

    PRIORITY RULES:
    - If the query mentions EGATUR, FRONTUR, INE reports, or FAQs → intent must be "rag" (unless it explicitly asks for a full "report", then use "reports").
    - If the query mentions "predict", "forecast", "how many tourists", "how much expenditure", "cluster" → intent = "predictor".
    - If the query mentions "report", "analysis", "recommendations", "explain" → intent = "reports".
    - If unsure, prefer "rag" for factual/document-based queries.

    STRICT OUTPUT FORMAT:
    Respond ONLY with one JSON object. 
    No explanations, no extra text.

    EXAMPLES:
    {"intent":"predictor","agents":["predictor"]}
    {"intent":"rag","agents":["rag"]}
    {"intent":"reports","agents":["predictor","rag"]}
    {"intent":"other","agents":[]}
    """, role="assistant")

########## MCP SERVER ##########
async def setup():
    # Mount existing MCP sub-servers
    main_mcp.mount(rag_mcp, prefix="rag")
    main_mcp.mount(report_mcp, prefix="report")

    # Mount your FastAPI API as an MCP server
    api_mcp = FastMCP.from_fastapi(fastapi_app, name="api-mcp")
    main_mcp.mount(api_mcp, prefix="api")

if __name__ == "__main__":
    asyncio.run(setup())
    print(f"🚀 Main MCP server running on http://{MCP_HOST}:{MCP_PORT}/mcp/")
    main_mcp.run(
        transport="http",
        host=MCP_HOST,
        port=MCP_PORT,
        log_level="DEBUG"
    )

########## HEALTH CHECK ##########
@main_mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")