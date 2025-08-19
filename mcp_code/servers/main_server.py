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
main_mcp.add_middleware(DetailedTimingMiddleware())

# Logging structured JSON of MCP requests (include truncated payloads)
main_mcp.add_middleware(StructuredLoggingMiddleware(include_payloads=True))

########### PROMPTS ##########
@main_mcp.prompt
def supervisor_prompt():
    return Message("""
    You are an intent classifier.
    Classify the user's query into one of these categories:

    - predictor: if the question requires:
        • prediction of the number of tourists for future periods using a numerical model, or
        • retrieval of historical tourist numbers for specific regions or Spain in specific periods.
    - rag: if the question is about documents, reports, project information, EGATUR, FRONTUR, textual data, or general explanations not related to predictions or historical tourist numbers.
    - reports: if the question requires explanation, reasoning, interpretation, summaries, or clarifications of results.
    - other: if it does not fit the above.

    Respond ONLY with one word: predictor, rag, reports or other.

    Examples:
    - "How many tourists visited Andalucía in 2023-07?" → predictor
    - "What does the ARIMA forecast mean?" → reports
    - "Show me the EGATUR report for last year" → rag
    - "Tell me a joke" → other
    """,
    role="assistant")

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