import os
import asyncio
from dotenv import load_dotenv
from fastmcp import FastMCP
from tourism_server import tourism_mcp
from rag_server import rag_mcp
from report_server import report_mcp

load_dotenv()

MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", 8080))

# Servidor principal
main_mcp = FastMCP(name="main-mcp-server")

async def setup():
    # Montar subservidores con prefijo
    main_mcp.mount(tourism_mcp, prefix="tourism")
    main_mcp.mount(rag_mcp, prefix="rag")
    main_mcp.mount(report_mcp, prefix="report")

if __name__ == "__main__":
    asyncio.run(setup())
    print(f"🚀 Main MCP server running on http://{MCP_HOST}:{MCP_PORT}/mcp/")
    main_mcp.run(
        transport="http",
        host=MCP_HOST,
        port=MCP_PORT,
        log_level="debug"
    )
