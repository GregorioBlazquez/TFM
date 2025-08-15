# servers/report_server.py
from fastmcp import FastMCP

report_mcp = FastMCP(name="mcp-report-agent")

@report_mcp.tool(tags={"report"})
def generate(data: dict) -> str:
    """
    Generates a basic report from data.
    """
    print("Generating report...")
    return f"Report generated with data: {data}"
