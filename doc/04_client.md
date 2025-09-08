# Client (Multi-Agent Chatbot)

The **Client** is the actual **orchestrator** of the system.\
While the MCP server only exposes tools, prompts, and resources, it is
the client that coordinates agents, manages state, and produces the
final answers for the user.

The client consists of two main components: 1. **Backend (FastAPI app)**
→ `api/client/backend_app.py` 2. **Frontend (Minimal Chat UI)** →
`api/client/chat.html`

------------------------------------------------------------------------

## 🔹 Backend (FastAPI Multi-Agent Orchestrator)

The backend is a **FastAPI application** that connects to the MCP server
via the `MultiServerMCPClient` adapter.\
It defines a **workflow graph** of agents, each responsible for a
different function: predictor, RAG, reports.

### Key Features

-   **Session management** with login and per-session state
    (conversation history, context summary, last outputs).\
-   **Agent orchestration** using `StateGraph` from LangGraph.\
-   **Integration with Azure OpenAI** deployments for reasoning,
    routing, prediction, and RAG.\
-   **Supervisor node** routes queries to the appropriate agents based
    on intent classification.\
-   **Conversation summarization** to maintain context efficiently.\
-   **REST endpoints** for login, chat, and asking questions.

### Main Endpoints

-   `/login` → Authenticate with username/password (from `USERS`
    environment variable).\
-   `/new_chat` → Reset session state.\
-   `/ask` → Send a query and receive orchestrated response.\
-   `/chat` → Serve the HTML frontend.\
-   `/health` → Health check.

### Workflow Overview

The backend builds a **state machine** with the following nodes: -
**supervisor** → classifies intent and decides which agents to run.\
- **dispatch** → executes requested agents (predictor, rag).\
- **prepare_reports** → prepares structured context for report agent.\
- **reports** → reasoning agent that interprets results.\
- **predictor/rag** → direct responses when needed.\
- **done/other** → terminal states.

👉 This orchestration ensures that queries like *"Forecast tourists in
Spain 2025"* trigger **predictor + rag**, while *"Explain the EGATUR
dataset"* routes to **rag** only, and *"Generate a tourism report"*
routes through **all agents**.

------------------------------------------------------------------------

## 🔹 Frontend (Chat UI)

Location: `api/client/chat.html`

A minimal **HTML + JavaScript** interface that allows interaction with
the backend API.

### Features

-   **Login screen** (username/password).\
-   **Simple chat window** for user ↔ agent interaction.\
-   **Theme toggle** (dark/light mode).\
-   **Markdown rendering** with `marked.js`.\
-   **New chat** button resets the conversation.

👉 The frontend is deliberately simple, focusing on functionality rather
than design. It communicates with the backend via `/login`, `/ask`, and
`/new_chat` endpoints.

------------------------------------------------------------------------

## 🔹 Client Responsibilities

-   Acts as the **true orchestrator** of the system.\
-   Loads and manages **prompts, tools, and resources** provided by the
    MCP server.\
-   Maintains **session state and conversation history**.\
-   Executes the **workflow graph** to combine multiple agents.\
-   Returns a **final user-facing answer** through the API and UI.

------------------------------------------------------------------------

## 🧭 Navigation

- [⬅️ Previous: MCP Server](/03_mcp_server.md)
- [🏠 Main index](../README.md#documentation)
- [➡️ Next: Data Models](/05_data_models.md)
