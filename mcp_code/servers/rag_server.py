from fastmcp import FastMCP

rag_mcp = FastMCP(name="mcp-rag")

@rag_mcp.tool(tags={"rag"})
def search(query: str) -> dict:
    """
    Simulates RAG search returning mock results.
    """
    print(f"RAG search for: {query}")
    return {
        "query": query,
        "results": [
            {"title": "Doc1", "snippet": "Test content 1"},
            {"title": "Doc2", "snippet": "Test content 2"}
        ]
    }
