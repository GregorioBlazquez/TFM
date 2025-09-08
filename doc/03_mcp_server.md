# MCP Server

The **MCP Server** is the central backend of the project. It is built on
top of **FastMCP** and organizes different sub-servers that expose the
available functionality: prediction, document retrieval, and reporting.

⚠️ **Important**: The actual *orchestrator* of the system is the
**client** (multi-agent chatbot). The MCP server does not decide
autonomously; instead, it exposes **tools, prompts, and resources** that
the client can use to execute workflows.

------------------------------------------------------------------------

## 🔹 General Architecture

The main server (`mcp_code/servers/main_server.py`) mounts three
sub-servers:

1.  **API MCP (Predictor)** → Based on FastAPI, it exposes forecasting,
    clustering, and expenditure prediction models.\
2.  **RAG MCP** → Retrieval of relevant documents (EGATUR, FRONTUR,
    FAQs, destinations, etc.).\
3.  **Report MCP** → Analytics agent that interprets predictions and
    documents, generating explanations and insights.

Additionally, the server defines **prompt templates** (e.g., supervisor,
predictor) that can be invoked by the client.

------------------------------------------------------------------------

## 🔹 Sub-servers

### 1. Predictor API (FastAPI + MCP)

Location: `api/server/main.py`, `model_handler.py`, `schemas.py`.

This sub-server wraps the FastAPI application as an MCP service.\
It provides 4 main endpoints:

-   **/predict** → Forecast number of tourists by region and period
    (ARIMA models).\
-   **/historical** → Query historical tourist data.\
-   **/cluster** → Assign a tourist profile to a cluster (classification
    model).\
-   **/expenditure** → Predict daily average expenditure with SHAP-based
    interpretability.

👉 Input/Output is defined with Pydantic schemas.\
👉 Models are loaded from `/models` and datasets from `/data/processed`.

------------------------------------------------------------------------

### 2. RAG MCP

Location: `mcp_code/servers/rag_server.py`.

This sub-server enables **semantic search and retrieval** from
documents.

-   **Index**: FAISS with embeddings (`sentence-transformers`).\
-   **Sources**: EGATUR, FRONTUR, project FAQs, destinations, and EDA
    notes.\
-   **Available tools**:
    -   `rag_search(query, k)` → search top-k relevant chunks.\
    -   `rag_upsert(uri, text)` → insert new text into the index.\
-   **Resources**:
    -   `historical://...` → access to time-series data by region/year.\
    -   `clusters://profiles` → qualitative cluster descriptions.\
    -   `eda://summary` → findings from the EDA.

👉 This agent **does not summarize or interpret**. It only returns raw
text/documents.

------------------------------------------------------------------------

### 3. Report MCP

Location: `mcp_code/servers/report_server.py`.

This agent interprets outputs from predictor and RAG sub-servers:

-   Explain why results are obtained.\
-   Compare against benchmarks and trends.\
-   Provide business/tourism insights.

👉 **Limitations**: It does not generate predictions nor retrieve new
documents.

------------------------------------------------------------------------

## 🔹 Supervisor Prompt

The `supervisor_prompt` in `main_server.py` defines a classification
schema.\
It returns JSON objects with two fields:

-   `"intent"`: `"predictor"`, `"rag"`, `"reports"`, `"other"`.\
-   `"agents"`: list of sub-servers to be executed.

Example outputs:

```json 
{"intent":"predictor","agents":["predictor","rag"]}
{"intent":"reports","agents":["rag"]}
```

------------------------------------------------------------------------

## 🔹 Other Endpoints

-   `/health` → simple health check.

------------------------------------------------------------------------

## 🔹 Usage Flow

1.  The **client** receives a user query.\
2.  The client may call the **supervisor prompt** to classify the
    intent.\
3.  Based on the decision, the client invokes the appropriate
    sub-servers:
    -   Predictor for forecasts, clusters, expenditure.\
    -   RAG for document retrieval.\
    -   Report for interpretation.\
4.  The client orchestrates the combination and delivers the final
    answer.

------------------------------------------------------------------------

## 🧭 Navigation

- [⬅️ Previous: Setup](/02_setup.md)
- [🏠 Main index](../README.md#documentation)
- [➡️ Next: Client](/04_client.md)
