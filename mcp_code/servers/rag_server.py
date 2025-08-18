# mcp_code/servers/rag_server.py
import os, json, re
from pathlib import Path
from typing import List, Dict, Optional

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

# ---------------- Embeddings ----------------
def _embed(texts: List[str]) -> np.ndarray:
    vecs = np.array(embedder.encode(texts, normalize_embeddings=True), dtype="float32")
    faiss.normalize_L2(vecs)  # Ensure embeddings are normalized for cosine similarity
    return vecs

# ---------------- Save / Load ----------------
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

# ---------------- Text Extraction ----------------
def extract_pdf_text(path: Path) -> str:
    """Extract clean text from PDF."""
    reader = PdfReader(str(path))
    text = []
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            txt = re.sub(r"\s+\n", " ", txt)  # merge broken lines
            txt = re.sub(r"\n{2,}", "\n", txt)  # preserve paragraphs
            text.append(txt)
    return "\n".join(text)

# ---------------- Chunking ----------------

def process_destination(text: str) -> List[str]:
    """
    Destination documents: keep entire text as a single chunk.
    """
    return [text.strip()]

def process_faqs(text: str) -> List[str]:
    """
    FAQs: split by questions (## Question) and include full answer.
    Assumes format: '## Question\nAnswer...'
    """
    chunks = []
    sections = re.split(r"\n## ", text)
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        if not sec.startswith("## "):
            sec = "## " + sec  # first section
        chunks.append(sec)
    return chunks[1:]  # skip the first title split

def process_pdf(text: str, min_len: int = 1000, max_len: int = 2000, overlap: int = 200) -> List[str]:
    """
    PDFs: split by paragraphs into large chunks.
    Combine paragraphs to get 1000–2000 chars per chunk.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 1 <= max_len:
            current += " " + p
        else:
            if current:
                chunks.append(current.strip())
                # start next chunk with overlap
                current = current[-overlap:] + " " + p if overlap < len(current) else current + " " + p
    if current.strip():
        chunks.append(current.strip())
    return chunks

# ---------------- Metadata ----------------
def parse_pdf_metadata(filename: str) -> Dict[str, Optional[int]]:
    m = re.search(r"(\d{2})(\d{2})", filename)
    if not m:
        return {"year": None, "month": None}
    month, year_suffix = int(m.group(1)), int(m.group(2))
    year = 2000 + year_suffix
    return {"year": year, "month": month}

# ---------------- Build Index ----------------
def _build_index_from_scratch():
    global index, _DOCS
    index = faiss.IndexFlatIP(dimension)
    _DOCS = []

    docs = []

    # --- FAQs ---
    faqs = DOCS_DIR / "faqs.md"
    if faqs.exists():
        docs.append({
            "uri": "docs://faqs",
            "text": faqs.read_text(encoding="utf-8"),
            "type": "faq",
            "title": "FAQs",
            "year": None,
            "month": None,
            "source_url": None
        })

    # --- Destinations ---
    dests = DOCS_DIR / "destinations"
    if dests.exists():
        for f in dests.glob("*.md"):
            docs.append({
                "uri": f"docs://destinations/{f.stem}",
                "text": f.read_text(encoding="utf-8"),
                "type": "destination",
                "title": f.stem,
                "year": None,
                "month": None,
                "source_url": None
            })

    # --- PDFs ---
    for pdf_name, title in [("FRONTUR0625.pdf", "FRONTUR 2025-06"),
                            ("EGATUR0625.pdf", "EGATUR 2025-06")]:
        pdf_path = DOCS_DIR / pdf_name
        if pdf_path.exists():
            text = extract_pdf_text(pdf_path)
            meta = parse_pdf_metadata(pdf_name)
            docs.append({
                "uri": f"docs://{pdf_name.lower().replace('.pdf','')}",
                "text": text,
                "type": "pdf",
                "title": title,
                "year": meta["year"],
                "month": meta["month"],
                "source_url": None
            })

    # --- Encode and add to FAISS ---
    for doc in docs:
        if not doc["text"].strip():
            continue
        if doc["type"] == "destination":
            chunks = process_destination(doc["text"])
        elif doc["type"] == "faq":
            chunks = process_faqs(doc["text"])
        else:
            chunks = process_pdf(doc["text"])
        for i, chunk in enumerate(chunks):
            vec = _embed([chunk])
            index.add(vec)
            _DOCS.append({
                "uri": f"{doc['uri']}#chunk{i}",
                "text": chunk,
                "type": doc["type"],
                "title": doc["title"],
                "year": doc["year"],
                "month": doc["month"],
                "source_url": doc.get("source_url")
            })

    _save_index()
    print(f"📚 Built new FAISS index with {len(_DOCS)} chunks")

# ---------------- RAG Tools ----------------
@rag_mcp.tool(tags={"rag"})
async def rag_upsert(uri: str, text: str, ctx: Context | None = None) -> str:
    """
    Add a plain text document into the FAISS RAG index as a single chunk and persist it.
    """
    global index, _DOCS
    if ctx:
        await ctx.info("rag_upsert:start", extra={"uri": uri})

    chunk = text.strip()
    if chunk:
        vec = _embed([chunk])
        index.add(vec)
        _DOCS.append({
            "uri": f"{uri}#chunk0",
            "text": chunk,
            "type": "custom",
            "title": uri,
            "year": None,
            "month": None,
            "source_url": None
        })
        _save_index()

    if ctx:
        await ctx.info("rag_upsert:done", extra={"uri": uri})

    return f"Upserted {uri} as a single chunk"

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
            d = _DOCS[idx]
            results.append({
                "score": float(scores[0][i]),
                "uri": d["uri"],
                "title": d.get("title"),
                "type": d.get("type"),
                "year": d.get("year"),
                "month": d.get("month"),
                "source_url": d.get("source_url"),
                "snippet": d["text"][:600]  # show more context
            })
    return {"query": query, "results": results}

# ---------- Init ----------
# if not _load_index():
_build_index_from_scratch()
print("⚠️ No existing index found, built from scratch.")
#for i, doc in enumerate(_DOCS):
#    print(i, doc["uri"], len(doc["text"]), doc["text"])
