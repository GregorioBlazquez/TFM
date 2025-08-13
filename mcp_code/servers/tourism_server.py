# mcp_server.py

import os
from dotenv import load_dotenv
from fastmcp import FastMCP

# Load env vars
load_dotenv()

# Create MCP server instance
tourism_mcp = FastMCP(name="mcp-tourism")

# Tools
from tools.tourism_predictor import predict_tourism

@tourism_mcp.tool()
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
    
    return predict_tourism(comunidad, periodo)
