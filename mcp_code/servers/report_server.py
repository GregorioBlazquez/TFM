# servers/report_server.py
from fastmcp import FastMCP
from fastmcp.prompts.prompt import Message

report_mcp = FastMCP(name="mcp-report-agent")

@report_mcp.tool(tags={"report"})
def generate(data: dict) -> str:
    """
    Generates a basic report from data.
    """
    print("Generating report...")
    return f"Report generated with data: {data}"

@report_mcp.prompt(tags={"report"})
def agent_prompt():
    return Message("""
        You are a Reasoning assistant.
            Your role is to:
            - Interpret outputs from other agents (predictions, RAG results).
            - Generate clear explanations, clarifications, and analytical narratives.
            - If the user asks things like "what does it mean?", "summarize", "explain", 
            or requests an interpretation, this is your responsibility.
            Do not call external tools. Only reason and explain.
    """,
    role="assistant")