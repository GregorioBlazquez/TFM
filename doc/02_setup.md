# Setup Guide

## Overview

This document describes how to set up and run the system in different environments:
- **Local environment with Python virtualenv** (recommended for development and notebooks)
- **Docker Compose** (for deploying all components together)
- **Inspector** (for debugging MCP servers and clients)

---

## 1. Local Environment (recommended for development)

### Requirements
- **Python**: version `3.11.13` (executable: `/usr/bin/python3.11`)
- **pip**: ensure it is up to date
- **Node.js / npm**: required for the Inspector (tested with npm `10.8.2`)

### Setup
Create and activate a Python virtual environment:
```
python3.11 -m venv TFM_venv
source TFM_venv/bin/activate
```

Install dependencies:
```
pip install --upgrade pip
pip install -r requirements.txt
```

This environment can be used for:
- Running the MCP server
- Running the multi-agent client
- Executing Jupyter notebooks

### Commands

Start the MCP server:
```
python -m mcp_code.servers.main_server
```

Start the backend client:
```
uvicorn api.client.backend_app:backend_app --host 0.0.0.0 --port 8000
```

---

## 2. Docker Compose

A preconfigured `docker-compose.yml` is provided in the `docker/` folder.

### Launch services
```
docker compose -f docker/docker-compose.yml up
```

### Build images locally (optional)
By default, DockerHub images are used. To force local builds:
```
docker compose -f docker/docker-compose.yml build --no-cache
```

---

## 3. Inspector

The [MCP Inspector](https://github.com/modelcontextprotocol/inspector) can be run outside the Python environment.  
It automatically connects to `localhost:8080` and allows visual inspection of servers, tools, and prompts.

### Launch Inspector
```
npx @modelcontextprotocol/inspector
```

Check npm version:
```
npx --version
```

---

## 4. Jupyter Notebooks

The same local Python environment (`TFM_venv`) can be used to run notebooks under the `notebooks/` folder:
- `01_EDA_FRONTUR.ipynb`
- `02_EDA_EGATUR.ipynb`
- `03_Model_ARIMA.ipynb`
- `04_Modeling_EGATUR.ipynb`

Simply start Jupyter Lab or Notebook after activating the environment.

---

## Troubleshooting

- **Port conflicts (8080 or 8000 in use)**  
  Kill the process or change the port in the launch command. Example:  
  ```
  uvicorn api.client.backend_app:backend_app --host 0.0.0.0 --port 8010
  ```

- **Inspector does not connect**  
  Ensure the MCP server is running on `localhost:8080`.  
  Check npm version (`npx --version`) and update Node.js if needed.

- **Docker build issues**  
  Run with `--no-cache` to rebuild images from scratch:  
  ```
  docker compose -f docker/docker-compose.yml build --no-cache
  ```

- **Missing dependencies in notebooks**  
  Always install from `requirements.txt`, not just `requirements.dev.txt` or `requirements.base.txt`.

---

## Summary

- For **development**: use the local virtual environment (`TFM_venv`).  
- For **quick deployment**: use Docker Compose.  
- For **debugging MCP**: use Inspector with `npx`.  
- All notebooks, server, and client share the same Python environment for consistency.

------------------------------------------------------------------------

## 🧭 Navigation

- [⬅️ Previous: Overview](/01_overview.md)
- [🏠 Main index](../README.md#documentation)
- [➡️ Next: Setup](/02_setup.md)