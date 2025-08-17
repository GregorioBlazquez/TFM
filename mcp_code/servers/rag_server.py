# mcp_code/servers/rag_server.py
import os, json
from pathlib import Path
from typing import List, Dict

from PyPDF2 import PdfReader
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastmcp import FastMCP, Context

rag_mcp = FastMCP(name="mcp-rag")

# --- Paths ---
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(exist_ok=True)
INDEX_PATH = DATA_DIR / "vector_index.faiss"
DOCS_PATH = DATA_DIR / "docs_meta.json"
DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"

# --- Embeddings + Index ---
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
embedder = SentenceTransformer(MODEL_NAME)
dimension = embedder.get_sentence_embedding_dimension()

# Global state
index = None
_DOCS: List[Dict] = []


def _embed(texts: List[str]) -> np.ndarray:
    return np.array(embedder.encode(texts, normalize_embeddings=True), dtype="float32")


def _save_index():
    faiss.write_index(index, str(INDEX_PATH))
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(_DOCS, f, ensure_ascii=False, indent=2)


def _load_index():
    global index, _DOCS
    if INDEX_PATH.exists() and DOCS_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            _DOCS = json.load(f)
        print(f"✅ Loaded FAISS index with {len(_DOCS)} docs")
        return True
    return False


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    text = []
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            text.append(txt)
    return "\n".join(text)


def _build_index_from_scratch():
    """Build a brand new FAISS index from docs/ contents."""
    global index, _DOCS
    index = faiss.IndexFlatL2(dimension)
    _DOCS = []

    docs = []

    # FAQs
    faqs = DOCS_DIR / "faqs.md"
    if faqs.exists():
        docs.append(("docs://faqs", faqs.read_text(encoding="utf-8")))

    # Destinations
    dests = DOCS_DIR / "destinations"
    if dests.exists():
        for f in dests.glob("*.md"):
            docs.append((f"docs://destinations/{f.stem}", f.read_text(encoding="utf-8")))

    # INE PDFs
    frontur = DOCS_DIR / "FRONTUR0625.pdf"
    if frontur.exists():
        docs.append(("docs://frontur", extract_pdf_text(frontur)))

    egatur = DOCS_DIR / "EGATUR0625.pdf"
    if egatur.exists():
        docs.append(("docs://egatur", extract_pdf_text(egatur)))

    # Encode + add to FAISS
    for uri, text in docs:
        if not text.strip():
            continue
        vec = _embed([text])
        index.add(vec)
        _DOCS.append({"uri": uri, "text": text})

    _save_index()
    print(f"📚 Built new FAISS index with {len(_DOCS)} docs")


# ---------- Tools ----------
@rag_mcp.tool(tags={"rag"})
async def rag_upsert(uri: str, text: str, ctx: Context | None = None) -> str:
    """
    Add a document into the FAISS RAG index and persist it.
    """
    global index, _DOCS
    if ctx:
        await ctx.info("rag_upsert:start", extra={"uri": uri})

    vec = _embed([text])
    index.add(vec)
    _DOCS.append({"uri": uri, "text": text})
    _save_index()

    if ctx:
        await ctx.info("rag_upsert:done", extra={"uri": uri})

    return f"Upserted {uri} (len={len(text)})"


@rag_mcp.tool(tags={"rag"})
async def rag_search(query: str, k: int = 3, ctx: Context | None = None) -> Dict:
    """
    Search top-k snippets from the FAISS RAG index.
    """
    if len(_DOCS) == 0:
        return {"query": query, "results": []}

    qvec = _embed([query])
    scores, ids = index.search(qvec, k)

    results = []
    for i, idx in enumerate(ids[0]):
        if idx < len(_DOCS):
            results.append({
                "score": float(scores[0][i]),
                "uri": _DOCS[idx]["uri"],
                "snippet": _DOCS[idx]["text"][:300]
            })

    return {"query": query, "results": results}


# ---------- Init ----------
#if not _load_index():
_build_index_from_scratch()
print("⚠️ No existing index found, built from scratch.")
for i, doc in enumerate(_DOCS):
    print(i, doc["uri"], len(doc["text"]))

