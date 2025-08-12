# mcp_server.py

import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from openai import AzureOpenAI

# Load env vars
load_dotenv()
MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", 8080))

# Create MCP server instance
mcp = FastMCP(name="mcp-tourism-agent", log_level="debug", debug=True)

# Tools
from tools.tourism_predictor import predict_tourism

@mcp.tool()
def tourism_tool(comunidad: str, periodo: str) -> dict:
    """
    Predicts the number of tourists for a given period using the local model.

    Args:
        comunidad (str): The community for which to predict tourism.
        periodo (str): The period in the format "YYYYMM".

    Returns:
        dict: A dictionary containing the predicted number of tourists.
        Ej: {"comunidad": "Andalucia", "periodo": "202509", "turistas": 3000000}
    """
    print(f"Predicting tourism for {comunidad} in {periodo}")
    print("HARD CODED PREDICTION FOR TESTING PURPOSES")
    
    return predict_tourism(comunidad, periodo)

# Run the HTTP server (streamable transport)
if __name__ == "__main__":
    print(f"🚀 MCP server running on http://{MCP_HOST}:{MCP_PORT}/mcp/")
    mcp.run(
        transport="http",
        host=MCP_HOST,
        port=MCP_PORT,
        log_level="debug"
    )
