# Tourism Assistant – Master’s Thesis (TFM)

This project is the Master’s Thesis (TFM) of **Gregorio Blázquez Martínez**, developed for the **Máster Big Data, Data Science & Artificial Intelligence (2024–2025)** at the **Universidad Complutense de Madrid**.

The goal is to combine **predictive modeling**, **retrieval-augmented generation (RAG)**, and **multi-agent orchestration** into a single platform that analyzes, explains, and forecasts **tourism trends in Spain**.

---

## Project Architecture (high level)

- **MCP Server** → Exposes tools, prompts, and resources (sub-servers: prediction API, RAG, reports).  
- **Client (multi-agent orchestrator)** → Connects to MCP server, coordinates agents, maintains session state and delivers final answers.  
- **Frontend** → Minimal chat interface (`api/client/chat.html`) for interacting with the assistant.  
- **Inspector** → Tool for visual debugging and protocol inspection.  
- **Data & Models** → INE FRONTUR (tourist flows) and EGATUR (expenditure) datasets; ARIMA, clustering and regression models.

---

## Repository Structure (short)

```text
.
├── api/                # FastAPI applications: prediction API (server) and chatbot client
├── azure/              # Azure Container Instance (ACI) deployment YAMLs
├── config.py           # Environment and configuration loader
├── data/               # Processed and raw datasets, FAISS index for RAG
├── doc/                # Main technical documentation (overview, setup, server, client, data, deployment)
├── docker/             # Dockerfiles and docker-compose setup
├── documents/          # Additional resources, INE reports (FRONTUR, EGATUR), FAQs
├── gpt.txt             # Prompt/experiment notes
├── mcp_code/           # MCP servers (main, rag, report) and client code (multiagents)
├── models/             # Trained models (ARIMA, clustering, expenditure regression)
├── notebooks/          # Jupyter notebooks (EDA and model training)
├── requirements.*.txt  # Environment-specific dependencies
└── README.md           # This file
```

> Note: the `doc/` folder contains the generated technical documentation files used as the project's docs index.

---

## Quick Start (developer)

1. Create a Python 3.11 virtual environment and install dependencies:
```bash
python3.11 -m venv TFM_venv
source TFM_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Run the MCP server (development):
```bash
python -m mcp_code.servers.main_server
```

3. Run the client backend (development):
```bash
uvicorn api.client.backend_app:backend_app --host 0.0.0.0 --port 8000
```

4. Or start everything with Docker Compose:
```bash
docker compose -f docker/docker-compose.yml up
```

For environment-specific setup and docker details, see `doc/02_setup.md` and `doc/06_deployment.md`.

---

## Documentation Index (detailed docs)

- [01 – Overview](doc/01_overview.md)  
- [02 – Setup Guide](doc/02_setup.md)  
- [03 – MCP Server](doc/03_mcp_server.md)  
- [04 – Client](doc/04_client.md)  
- [05 – Data & Models](doc/05_data_model.md)  
- [06 – Deployment](doc/06_deployment.md)  
- Additional resources and INE files: see `documents/` (FRONTUR, EGATUR, EDA notes).
