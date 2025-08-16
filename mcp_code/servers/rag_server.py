from fastmcp import FastMCP
from fastmcp import Context

rag_mcp = FastMCP(name="mcp-rag")

@rag_mcp.tool(tags={"rag"})
async def search(query: str, ctx: Context | None = None) -> dict:
    """
    Simulates RAG search returning mock results.
    """
    if ctx:
        await ctx.info("rag_search:start", extra={
            "request_id": getattr(ctx, "request_id", None),
            "query_preview": (query[:200] + "…") if len(query) > 200 else query
        })
        await ctx.report_progress(1, 100, "Buscando en índice RAG...")

    results = [
        {"title": "Doc1", "snippet": "Test content 1"},
        {"title": "Doc2", "snippet": "Test content 2"}
    ]

    if ctx:
        await ctx.info("rag_search:done", extra={
            "request_id": getattr(ctx, "request_id", None),
            "results_count": len(results)
        })
        await ctx.report_progress(100, 100, "Completado")

    return {"query": query, "results": results}

