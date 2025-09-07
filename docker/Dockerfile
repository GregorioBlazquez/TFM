# Base image with Python
FROM python:3.11-slim

# Avoid bytecode and buffering
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create working directory
WORKDIR /app

# Install only essential system dependencies first
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies efficiently
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy only the CODE
COPY mcp_code/ ./mcp_code/
COPY api/ ./api/
COPY config.py ./
COPY .env.docker ./

# Expose ports
EXPOSE 8080 8000

# Default command
CMD ["python", "-m", "mcp_code.servers.main_server"]