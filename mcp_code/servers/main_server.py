import os
import asyncio
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from dotenv import load_dotenv
from fastmcp import FastMCP
from mcp_code.servers.rag_server import rag_mcp
from mcp_code.servers.report_server import report_mcp
from api.main import app as fastapi_app

load_dotenv()

MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", 8080))

# Main MCP server
main_mcp = FastMCP(name="main-mcp-server")

async def setup():
    # Mount existing MCP sub-servers
    main_mcp.mount(rag_mcp, prefix="rag")

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


@main_mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")