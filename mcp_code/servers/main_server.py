########## IMPORTS ##########
import os
import asyncio
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from config import get_env_var
from fastmcp import FastMCP
from mcp_code.servers.rag_server import rag_mcp
from mcp_code.servers.report_server import report_mcp
from api.server.main import app as fastapi_app
import logging
from fastmcp.server.middleware.timing import DetailedTimingMiddleware
from fastmcp.server.middleware.logging import StructuredLoggingMiddleware
from fastmcp.prompts.prompt import Message
from pathlib import Path

########## ENV VARS ##########
# Load environment variables
MCP_HOST = get_env_var("MCP_HOST", "127.0.0.1")
MCP_PORT = int(get_env_var("MCP_PORT", 8080))

########## LOGGING ##########
# Module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=get_env_var("MCP_LOGGING_LEVEL", logging.DEBUG),
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

########## MCP SERVER ##########
# Main MCP server
main_mcp = FastMCP(name="main-mcp-server")
logger.info("Initialized main MCP server.")

# Log timing (tools/resources/prompts)
#main_mcp.add_middleware(DetailedTimingMiddleware())

# Logging structured JSON of MCP requests (include truncated payloads)
#main_mcp.add_middleware(StructuredLoggingMiddleware(include_payloads=True))

########### PROMPTS ##########
@main_mcp.prompt
def supervisor_prompt():
    return Message("""
    You are an intent classifier for a tourism assistant.
    Your job is to decide the high-level intent and which internal agents should run BEFORE any reporting.

    Always return a valid JSON object with two fields:
    - "intent": one of "predictor", "rag", "reports", "other"
    - "agents": a list of agent names to execute, subset of ["predictor","rag"]

    CLASSIFICATION RULES:
    1. "predictor":
       - Use when the user asks directly for forecasts, predictions, future values, estimates of tourist arrivals, expenditure, or cluster assignment.
       - Examples: "How many tourists will visit Spain in 2025?", "Predict the average expenditure", "Assign this profile to a cluster".
       - → agents = ["predictor","rag"]

    2. "rag":
       - Use when the user asks about official statistics, summaries, definitions, or project documentation (EGATUR, FRONTUR, INE reports, PDFs, FAQs, project author).
       - Examples: "Summarize the EGATUR report", "What is FRONTUR?", "Who is the author of this project?", "Are the data official or generated?".
       - → agents = ["rag"]

    3. "reports":
       - Use when the user requests a comprehensive analysis, explanation, recommendations, or benchmarking.
       - Distinguish two cases:
         * Predictive reports (contain terms like "predict", "forecast", "future", "estimate", "cluster", "expenditure") → agents = ["predictor","rag"]
         * Descriptive reports (contain terms like "explain", "analysis", "recommendations", "insights", "compare", "why") without forecast/cluster/expenditure keywords → agents = ["rag"]

       - Examples predictive: "Generate a report about tourism in Spain next year", "Explain the forecast compared to EGATUR", "Forecast and analyze expenditure".
       - Examples descriptive: "Explain the EGATUR results for 2024", "Give me recommendations for Andalusia based on past trends", "Analyze tourist clusters".

    4. "other":
       - Only when the query is unrelated to tourism, forecasts, reports, or project documentation.
       - Examples: "Tell me a joke", "What is the capital of France?".
       - → agents = []

    PRIORITY RULES:
    - If query mentions "predict", "forecast", "future", "estimate", "cluster", or "expenditure" → treat as predictor or predictive report (agents must include "predictor").
    - If query mentions EGATUR, FRONTUR, INE, FAQs → treat as rag (unless explicitly predictive, then reports).
    - If query mentions "report", "analysis", "recommendations", "explain", "insights" → treat as reports (decide predictive vs descriptive).
    - If unsure → include "rag".

    STRICT OUTPUT FORMAT:
    Respond ONLY with one JSON object.
    No explanations, no extra text.

    EXAMPLES:
    {"intent":"predictor","agents":["predictor","rag"]}
    {"intent":"rag","agents":["rag"]}
    {"intent":"reports","agents":["predictor","rag"]}
    {"intent":"reports","agents":["rag"]}
    {"intent":"other","agents":[]}
    """, role="assistant")


@main_mcp.prompt
def predictor_prompt():
    return Message("""
    You are the Predictor Agent. 
    Your only role is to answer user queries by calling the available API tools for forecasting, clustering, or expenditure prediction.

    ### Instructions:
    - Always use the provided tools (`api_tools`) to answer.
    - Do NOT generate numbers, explanations, or assumptions yourself.
    - Select the most appropriate tool based on the query:
      * For time-series forecasts (e.g. "How many tourists in Valencia in 2025-08?"), use the ARIMA prediction tool.
      * For clustering (e.g. "What type of tourist is this profile?"), use the clustering tool.
      * For expenditure forecasts (e.g. "What is the average daily expenditure for this profile?"), use the expenditure tool.
    - Return the tool output **as-is**, without extra commentary.

    ### Examples:
    User: "Predict the number of tourists in Spain in August 2025."
    → Call ARIMA tool with {region:"Spain", period:"2025M08"}.

    User: "Assign this tourist profile (7 nights, leisure, UK) to a cluster."
    → Call clustering tool with the given features.

    User: "Estimate the daily average expenditure for a German tourist in Madrid, summer, 5 nights."
    → Call expenditure tool with the given features.

    ### Important:
    - If the query is unrelated to prediction, do NOT guess — just return nothing.
    - You are a backend prediction service, not a report writer or analyst.
    """, role="assistant")

########## MCP SERVER ##########
async def setup():
    try:
        # Mount existing MCP sub-servers
        logger.info("Mounting RAG MCP sub-server...")
        main_mcp.mount(rag_mcp, prefix="rag")
        logger.info("Mounting Report MCP sub-server...")
        main_mcp.mount(report_mcp, prefix="report")

        # Mount FastAPI API as an MCP server (predictor)
        logger.info("Mounting FastAPI API as MCP sub-server...")
        api_mcp = FastMCP.from_fastapi(fastapi_app, name="api-mcp")
        main_mcp.mount(api_mcp, prefix="api")
        logger.info("All sub-servers mounted successfully.")
    except Exception as e:
        logger.exception(f"Error during server setup: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(setup())
    logger.info(f"🚀 Main MCP server running on http://{MCP_HOST}:{MCP_PORT}/mcp/")
    main_mcp.run(
        transport="http",
        host=MCP_HOST,
        port=MCP_PORT,
        log_level=get_env_var("MCP_LOGGING_LEVEL", "DEBUG")
    )

########## HEALTH CHECK ##########
@main_mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    logger.debug("Health check endpoint called.")
    return PlainTextResponse("OK")