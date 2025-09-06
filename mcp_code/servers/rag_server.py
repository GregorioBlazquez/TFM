# mcp_code/servers/rag_server.py
import os, json, re
from pathlib import Path
from typing import List, Dict, Optional

from pypdf import PdfReader
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from fastmcp import FastMCP, Context
from fastmcp.prompts.prompt import Message
import pandas as pd
from mcp.types import TextResourceContents

from config import get_env_var

rag_mcp = FastMCP(name="mcp-rag")

import logging
logger = logging.getLogger(__name__)

# --- Paths ---
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(exist_ok=True)
INDEX_PATH = DATA_DIR / "vector_index.faiss"
DOCS_PATH = DATA_DIR / "docs_meta.json"
DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"

# --- Embeddings + Index ---
MODEL_NAME = get_env_var("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
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
    logger.info(f"Saved FAISS index and docs metadata. Total docs: {len(_DOCS)}")

def _load_index():
    global index, _DOCS
    if INDEX_PATH.exists() and DOCS_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            _DOCS = json.load(f)
        logger.info(f"✅ Loaded FAISS index with {len(_DOCS)} docs")
        return True
    logger.warning("No existing FAISS index or docs metadata found.")
    return False

# ---------------- PDF Text Extraction ----------------
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
def process_full_markdown(text: str) -> List[str]:
    """
    Not long markdown documents: keep entire text as a single chunk.
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

def process_pdf(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    PDFs: split by paragraphs into large chunks with some overlap
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)

# ---------------- Metadata ----------------
def parse_pdf_metadata(filename: str) -> Dict[str, Optional[int]]:
    m = re.search(r"(\d{2})(\d{2})", filename)
    if not m:
        return {"year": None, "month": None}
    month, year_suffix = int(m.group(1)), int(m.group(2))
    year = 2000 + year_suffix
    return {"year": year, "month": month}

# ---------------- Build Index ----------------
def _scan_docs() -> List[Dict]:
    docs = []

    def add_doc(path: Path, uri: str, title: str, doc_type: str, year=None, month=None):
        docs.append({
            "uri": uri,
            "path": str(path),
            "text": None,   # se llenará después
            "type": doc_type,
            "title": title,
            "year": year,
            "month": month,
            "source_url": None,
            "mtime": path.stat().st_mtime
        })

    # --- FAQs ---
    faqs = DOCS_DIR / "project_faqs.md"
    if faqs.exists():
        add_doc(faqs, "docs://faqs", "Project FAQs", "faq")

    # --- EGATUR ---
    egatur_dir = DOCS_DIR / "egatur"
    if egatur_dir.exists():
        for f in egatur_dir.iterdir():
            if f.suffix.lower() == ".md":
                add_doc(f, f"docs://egatur/{f.stem}", f"EGATUR {f.stem}", "markdown")
            elif f.suffix.lower() == ".pdf":
                meta = parse_pdf_metadata(f.name)
                title = f"EGATUR {meta['year']}-{meta['month']:02d}" if meta["year"] else f.name
                add_doc(f, f"docs://egatur/{f.stem}", title, "pdf", meta["year"], meta["month"])

    # --- FRONTUR ---
    frontur_dir = DOCS_DIR / "frontur"
    if frontur_dir.exists():
        for f in frontur_dir.iterdir():
            if f.suffix.lower() == ".md":
                add_doc(f, f"docs://frontur/{f.stem}", f"FRONTUR {f.stem}", "markdown")
            elif f.suffix.lower() == ".pdf":
                meta = parse_pdf_metadata(f.name)
                title = f"FRONTUR {meta['year']}-{meta['month']:02d}" if meta["year"] else f.name
                add_doc(f, f"docs://frontur/{f.stem}", title, "pdf", meta["year"], meta["month"])

    # --- Destinations ---
    dests = DOCS_DIR / "destinations"
    if dests.exists():
        for f in dests.glob("*.md"):
            add_doc(f, f"docs://destinations/{f.stem}", f"Destination {f.stem.title()}", "destination")

    return docs


def _build_or_update_index():
    global index, _DOCS

    # Cargar índice existente
    fresh = True
    if _load_index():
        fresh = False

    # Escanear docs actuales
    scanned_docs = _scan_docs()

    # Mapear los existentes por URI
    existing = {doc["uri"]: doc for doc in _DOCS}

    new_docs = []
    unchanged = []
    for doc in scanned_docs:
        if doc["uri"] not in existing:
            logger.info(f"🆕 New document: {doc['uri']}")
            new_docs.append(doc)
        else:
            if abs(doc["mtime"] - existing[doc["uri"]].get("mtime", 0)) > 1:
                logger.info(f"♻️ Updated document: {doc['uri']}")
                new_docs.append(doc)
            else:
                unchanged.append(existing[doc["uri"]])

    if fresh or new_docs:
        logger.info(f"📚 Rebuilding index: {len(new_docs)} new/updated docs, {len(unchanged)} unchanged")

        # Reiniciar FAISS si es rebuild completo
        index = faiss.IndexFlatIP(dimension)
        _DOCS = []

        # Volver a indexar todo (unchanged + new_docs)
        for doc in unchanged + new_docs:
            path = Path(doc["path"])
            if doc["type"] == "faq":
                text = path.read_text(encoding="utf-8")
                chunks = process_faqs(text)
            elif doc["type"] in ("destination", "markdown"):
                text = path.read_text(encoding="utf-8")
                chunks = process_full_markdown(text)
            elif doc["type"] == "pdf":
                text = extract_pdf_text(path)
                chunks = process_pdf(text)
            else:
                continue

            for i, chunk in enumerate(chunks):
                vec = _embed([chunk])
                index.add(vec)
                _DOCS.append({
                    **doc,
                    "uri": f"{doc['uri']}#chunk{i}",
                    "text": chunk
                })

        _save_index()
    else:
        logger.info("✅ No document changes detected, using cached index.")

# ---------------- RAG Tools ----------------
@rag_mcp.tool(tags={"rag"})
async def rag_upsert(uri: str, text: str, ctx: Context | None = None) -> str:
    """
    Add a plain text document into the FAISS RAG index as a single chunk and persist it.
    """
    global index, _DOCS
    logger.info(f"Upserting document: {uri}")
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
        logger.info(f"Upserted {uri} as a single chunk.")
    else:
        logger.warning(f"Attempted to upsert empty chunk for {uri}.")

    if ctx:
        await ctx.info("rag_upsert:done", extra={"uri": uri})

    return f"Upserted {uri} as a single chunk"

@rag_mcp.tool(tags={"rag"})
async def rag_search(query: str, k: int = 3, ctx: Context | None = None) -> Dict:
    """
    Search top-k snippets from the FAISS RAG index.
    """
    logger.info(f"RAG search called with query: {query}")
    if len(_DOCS) == 0:
        logger.warning("RAG search attempted with empty document index.")
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
                "snippet": d["text"][:100],
                "content": d["text"]
            })
    logger.info(f"RAG search returned {len(results)} results.")
    return {"query": query, "results": results}

# ---------------- Prompts ----------------
@rag_mcp.prompt(tags={"rag"})
def agent_prompt():
    return Message("""
    You are a Retrieval-Augmented Generation (RAG) assistant. 
    Your only job is to retrieve relevant raw text from the knowledge base (e.g., FRONTUR, EGATUR, FAQs, EDA, clusters).

    Available tools: rag_rag_search, rag_rag_upsert.

    Rules:
    - ALWAYS call rag_rag_search first with the user query.
    - Return only the `content` field from the most relevant chunks (not just snippets).
    - You may filter or drop chunks that are clearly irrelevant.
    - If no chunks are relevant, return an empty result.
    - Do NOT answer the user directly.
    - Do NOT summarize, interpret, or format the content.
    - The downstream Reports agent is responsible for analysis and explanation.

    Output must contain ONLY the retrieved raw text (no extra commentary).
    """,
    role="assistant")

# ---------------- Resources ----------------
from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data/processed/num_tourists.csv"
TOURIST_DATA = pd.read_csv(DATA_PATH, sep=";", parse_dates=["Period"])
TOURIST_DATA.rename(columns={"CCAA": "region", "Total": "tourists"}, inplace=True)

@rag_mcp.resource(
    uri="historical://by-all",
    name="HistoricalAll",
    description="Return the entire historical tourist dataset grouped by period → region → value.",
    mime_type="application/json",
    tags={"tourism", "historical", "all"}
)
def historical_all() -> dict:
    df = TOURIST_DATA.copy()
    print(df)
    output = {}
    for p, group in df.groupby("Period"):
        output[str(p.date())] = {
            r: float(v) for r, v in zip(group["region"], group["tourists"])
        }

    print(output)
    return output

def normalize_region_name(region_input: str) -> str:
    """ Normalize region names to match the dataset """
    region_mapping = {
        'andalucia': '01 Andalucía',
        'andalucía': '01 Andalucía',
        'balears': '04 Balears, Illes',
        'baleares': '04 Balears, Illes',
        'canarias': '05 Canarias',
        'cataluña': '09 Cataluña',
        'cataluna': '09 Cataluña',
        'valenciana': '10 Comunitat Valenciana',
        'valencia': '10 Comunitat Valenciana',
        'comunidad valenciana': '10 Comunitat Valenciana',
        'madrid': '13 Madrid, Comunidad de',
        'comunidad de madrid': '13 Madrid, Comunidad de',
        'total': 'Total',
        'otras': 'Otras Comunidades Autónomas'
    }
    
    normalized = region_mapping.get(region_input.lower(), region_input)
    return normalized

@rag_mcp.resource(
    uri="historical://by-region-year/{year}/{region}",
    name="HistoricalByRegionYear",
    description="Return historical tourist data filtered by region and year.",
    mime_type="application/json",
    tags={"tourism", "historical", "region", "year"}
)
def historical_by_region_year(region: str, year: int) -> dict:
    df = TOURIST_DATA.copy()
    df = df[df["Period"].dt.year == year]

    normalized_region = normalize_region_name(region)

    if normalized_region != "Total":
        df = df[df["region"] == normalized_region]
    elif region == "Total":
        df = df[df["region"] == "Total"]

    if df.empty:
        return {"error": f"No data found for region='{region}' and year='{year}'"}

    output = {}
    for p, group in df.groupby("Period"):
        output[str(p.date())] = {
            r: float(v) for r, v in zip(group["region"], group["tourists"])
        }
    return output

@rag_mcp.resource(
    uri="historical://by-year/{year}",
    name="HistoricalByYear",
    description="Return all historical tourist data for a given year across all regions.",
    mime_type="application/json",
    tags={"tourism", "historical", "year"}
)
def historical_by_year(year: int) -> dict:
    df = TOURIST_DATA.copy()
    df = df[df["Period"].dt.year == year]

    if df.empty:
        return {"error": f"No data found for year {year}"}

    output = {}
    for p, group in df.groupby("Period"):
        output[str(p.date())] = {
            r: float(v) for r, v in zip(group["region"], group["tourists"])
        }
    return output

@rag_mcp.resource(
    uri="historical://by-region/{region}",
    name="HistoricalByRegion",
    description="Return the full historical tourist data for a given region across all years.",
    mime_type="application/json",
    tags={"tourism", "historical", "region"}
)
def historical_by_region(region: str) -> dict:
    df = TOURIST_DATA.copy()

    normalize_region = normalize_region_name(region)

    if normalize_region != "Total":
        df = df[df["region"] == normalize_region]
    else:
        df = df[df["region"] == "Total"]

    if df.empty:
        return {"error": f"No data found for region '{region}'"}

    output = {}
    for p, group in df.groupby("Period"):
        output[str(p.date())] = {
            r: float(v) for r, v in zip(group["region"], group["tourists"])
        }
    return output

# --- Clusters Resource ---
@rag_mcp.resource(
    uri="clusters://profiles",
    name="TouristClusters",
    description="Return descriptive profiles of the 4 tourist clusters identified in the EGATUR analysis.",
    mime_type="application/json",
    tags={"tourism", "clusters", "segmentation"}
)
def cluster_profiles() -> dict:
    return {
        "Cluster 0": {
            "label": "Long-stay visitors in Valencian Community",
            "trip_pattern": "Longer trips, lower-to-medium daily expenditure",
            "accommodation": "Mostly non-market",
            "purpose": "Leisure and personal/other",
            "regions": ["Valencian Community"]
        },
        "Cluster 1": {
            "label": "Standard tourists (Island vacationers)",
            "trip_pattern": "Typical summer leisure trips",
            "accommodation": "Mostly hotels",
            "purpose": "Leisure",
            "regions": ["Balearic Islands", "Canary Islands"]
        },
        "Cluster 2": {
            "label": "Personal/family visits",
            "trip_pattern": "Long stays, spread across regions",
            "accommodation": "Mostly non-market",
            "purpose": "Visiting friends/family",
            "regions": ["Multiple regions", "Winter bias"]
        },
        "Cluster 3": {
            "label": "Urban, high-spending international tourists",
            "trip_pattern": "Shorter trips, high daily expenditure",
            "accommodation": "Almost exclusively hotels",
            "purpose": "Leisure + business",
            "regions": ["Catalonia", "Madrid"],
            "notable_markets": ["Russia + Rest of the world"]
        }
    }

# --- EDA Resource ---
@rag_mcp.resource(
    uri="eda://summary",
    name="EDAFindings",
    description="Summary of exploratory data analysis (EDA) on EGATUR tourist dataset.",
    mime_type="application/json",
    tags={"tourism", "eda", "analysis"}
)
def eda_summary() -> dict:
    return {
        "univariate": {
            "numerical": "Right skew, big outliers in expenditure and trip length",
            "categorical": "Trips concentrated in Catalonia, Balearic Islands, Andalusia, Canary Islands, Valencian Community, Madrid"
        },
        "bivariate": {
            "expenditure_vs_length": "Negative correlation: longer trips → lower daily spending",
            "outliers": "High spenders vary by metric (total vs daily vs weighted)"
        },
        "temporal": "Some seasonality, but not the main driver",
        "conclusions": [
            "High dispersion in data makes modeling difficult",
            "Outliers differ by metric",
            "Clusters provide more stability for segmentation"
        ]
    }


# ---------- Init ----------
logger.info("🔍 Starting to build or update document index...")
_build_or_update_index()
for i, doc in enumerate(_DOCS):
    logger.debug(f"{i} {doc['uri']} {len(doc['text'])}")
