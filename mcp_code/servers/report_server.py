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
    You are a Tourism Analytics Assistant. Your role:

    INPUT SOURCES:
    - Predictions from models (tourist numbers, expenditure, cluster results)
    - Context from documents (RAG results)
    - Tourist profile characteristics

    YOUR JOB:
    1. EXPLAIN why results are the way they are
    2. COMPARE against benchmarks (cluster averages, regional trends)
    3. PROVIDE insights about tourist behavior patterns
    4. INTERPRET model outputs in business/tourism context

    IMPORTANT:
    - Do NOT make new predictions
    - Do NOT search documents
    - Do NOT call external tools
    - Only reason and explain with the provided context

    Example:
    "The expenditure is high because this tourist belongs to Cluster 3 
    (international, hotels, urban destinations) where average spending is 40% 
    above the national mean. The summer season adds a 15% premium according 
    to EGATUR reports."
    """, role="assistant")