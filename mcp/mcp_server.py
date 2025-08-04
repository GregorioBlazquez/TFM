import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load .env variables
load_dotenv()
MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", 8080))

# Create MCP server instance
mcp_server = FastMCP("mcp-tourism-agent", host=MCP_HOST, port=MCP_PORT)

"""
TOOLS
"""
from tools.tourism_predictor import predict_tourism

@mcp_server.tool()
def tourism_tool(comunidad: str, periodo: str) -> dict:
    """Predicts the number of tourists for a given period using the local model.
    
    Args:
        comunidad (str): The community for which to predict tourism.
        periodo (str): The period in the format "YYYYMM".
    
    Returns:
        dict: A dictionary containing the predicted number of tourists.
    """
    return predict_tourism(comunidad, periodo)


# Launch the server
if __name__ == "__main__":
    print("✅ MCP server is running in stdio mode. Enter message...")
    mcp_server.run()