ARG BASE_IMAGE=nvidia/cuda:13.0.0-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

# System deps: Python 3.12, pip, venv, git (for some HF downloads), curl
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
        git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3

WORKDIR /app

# Install Python dependencies first (cache-friendly)
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements.txt

ENV PATH="/opt/venv/bin:$PATH"

# Copy project files
COPY . .

# HF cache is a volume mount target; created at runtime if missing
VOLUME ["/app/.cache/huggingface", "/app/outputs"]

CMD ["bash"]
