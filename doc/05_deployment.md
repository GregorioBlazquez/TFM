# Deployment Guide

This document explains how to build, deploy, and run the **MCP server**,
**client**, and **inspector** components, both locally with Docker
Compose and remotely on **Azure Container Instances (ACI)**.

------------------------------------------------------------------------

## 🔹 Local Deployment with Docker Compose

### Build images

```bash 
docker compose -f docker/docker-compose.yml build
--no-cache ```

### Run services

```bash 
docker compose -f docker/docker-compose.yml up
```

### Services

-   **MCP Server** → http://localhost:8080/health\
-   **Client (backend + frontend)** →
    http://localhost:8000/static/chat.html\
-   **MCP Inspector** → connects to server on port 8080

------------------------------------------------------------------------

## 🔹 Dockerfiles

### MCP Server (`Dockerfile.server`)

-   Installs Python dependencies from `requirements.base.txt` and
    `requirements.server.txt`.\
-   Copies **servers code**, **API server**, **config**, **.env**, plus
    **data/docs/models**.\
-   Exposes port **8080**.\
-   Default command: ```bash python -m
    mcp_code.servers.main_server ```

### Client (`Dockerfile.client`)

-   Installs Python dependencies from `requirements.base.txt` and
    `requirements.client.txt`.\
-   Copies **client backend code**, **config**, **.env**.\
-   Exposes port **8000**.\
-   Default command: ```bash uvicorn
    api.client.backend_app:backend_app --host 0.0.0.0 --port 8000
    ```

### docker-compose.yml

Defines three services: 1. **mcp-server** → builds and runs server (port
8080).\
2. **client** → builds and runs FastAPI backend with chat frontend (port
8000).\
3. **mcp-inspector** → connects to server for debugging.

------------------------------------------------------------------------

## 🔹 Azure Deployment (ACI)

Both **MCP server** and **client** are deployed as separate **container
groups** within resource group `TFM`.\
Images are pushed to **Azure Container Registry (ACR)**:
`tfmacrdemo.azurecr.io`.

### Push Images

```bash 
docker push gregorioblazquezm/mcp-server:latest docker
push gregorioblazquezm/client:latest 
```

### Deploy MCP Server

```bash 
az container create --resource-group TFM --file
azure/mcp-server-aci.yaml az container show --resource-group TFM --name
mcp-server --query "ipAddress.ip" -o tsv 
```

Example health check: 
```bash
 curl -i
http://`<SERVER_IP>`{=html}:8080/health 
```

### Deploy Client

```bash 
az container create\
--resource-group TFM\
--file azure/client-aci.yaml\
--environment-variables\
AZURE_OPENAI_API_KEY="\*\*\*"\
USERS='{"user":"password1234","user2":"password1234"}' az container show
--resource-group TFM --name mcp-client --query "ipAddress.ip" -o tsv
```

Access frontend at: - http://`<CLIENT_IP>`{=html}:8000/static/chat.html

### Deploy Inspector

```bash 
az container create --resource-group TFM --file
azure/inspector-aci.yaml az container show --resource-group TFM --name
mcp-inspector --query "ipAddress.ip" -o tsv 
```

------------------------------------------------------------------------

## 🔹 Testing with Docker (Local)

Run containers manually for testing.

### MCP Server

```bash 
docker run -e ENVIRONMENT=azure\
-e AZURE_OPENAI_API_KEY="\*\*\*"\
-e USERS='{"user":"password1234"}'\
-p 8080:8080 gregorioblazquezm/mcp-server:latest 
```

### Client

```bash 
docker run -e ENVIRONMENT=azure\
-e MCP_BASE="http://`<SERVER_IP>`{=html}:8080/mcp/"\
-e AZURE_OPENAI_API_KEY="\*\*\*"\
-e USERS='{"user":"password1234"}'\
-p 8000:8000 gregorioblazquezm/client:latest 
```

------------------------------------------------------------------------

## 🔹 Summary

-   **Local deployment** uses `docker-compose`.\
-   **Cloud deployment** uses **Azure Container Instances** with YAML
    manifests.\
-   **MCP Server** exposes tools, prompts, and resources.\
-   **Client** orchestrates agents and exposes a user-facing chat
    interface.\
-   **Inspector** provides debugging of the MCP protocol.

------------------------------------------------------------------------

## 🧭 Navigation

- [⬅️ Previous: Client](/04_client.md)
- [🏠 Main index](../README.md#documentation)