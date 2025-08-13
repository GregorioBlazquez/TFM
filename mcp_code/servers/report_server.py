# servers/report_server.py
from fastmcp import FastMCP

report_mcp = FastMCP(name="mcp-report-agent")

@report_mcp.tool()
def generate_report(data: dict) -> str:
    """
    Genera un informe básico a partir de datos.
    """
    print("Generating report...")
    return f"Informe generado con datos: {data}"
