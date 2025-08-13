from fastmcp import FastMCP

rag_mcp = FastMCP(name="mcp-rag")

@rag_mcp.tool()
def rag_search(query: str) -> dict:
    """
    Simula búsqueda RAG devolviendo resultados ficticios.
    """
    print(f"RAG search for: {query}")
    return {
        "query": query,
        "results": [
            {"title": "Doc1", "snippet": "Contenido de prueba 1"},
            {"title": "Doc2", "snippet": "Contenido de prueba 2"}
        ]
    }
