# Marshal — multi-stage Dockerfile.
#
# Stage 1 (llama-builder): build llama-server with native ISA support.
# Stage 2 (runtime):       slim image with the venv, agentd, and api.
#
# The model is NOT baked into the image (license + 1.8GB). Mount it at
# /models when running:
#
#   docker run --rm -v $PWD/models:/models -p 8080:8080 marshal/inference:0.3.0
#
# For the full stack, prefer docker-compose.yml.

# ---------------------------------------------------------------------------
# Stage 1: build llama.cpp
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS llama-builder

ARG LLAMA_CPP_COMMIT=master

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
WORKDIR /build/llama.cpp
RUN if [ "$LLAMA_CPP_COMMIT" != "master" ]; then \
        git fetch --depth 1 origin "$LLAMA_CPP_COMMIT" && \
        git checkout "$LLAMA_CPP_COMMIT"; \
    fi
# LLAMA_NATIVE=OFF: container builds run on heterogenous host CPUs, do not
# bake -march=native instructions. Re-enable in your own build for ~10-20%
# inference speedup if you control the host CPU.
RUN cmake -B build \
        -DLLAMA_CURL=OFF \
        -DLLAMA_NATIVE=OFF \
        -DGGML_NATIVE=OFF \
    && cmake --build build --target llama-server -j "$(nproc)"

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.13-slim-bookworm AS runtime

# Runtime system libraries used by llama-server + Python deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libstdc++6 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Pull llama-server from the builder stage.
COPY --from=llama-builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

# Install the Python project.
WORKDIR /opt/marshal
COPY pyproject.toml requirements.txt README.md LICENSE /opt/marshal/
COPY agents /opt/marshal/agents
COPY api /opt/marshal/api
COPY cortex /opt/marshal/cortex
COPY db /opt/marshal/db
COPY inference /opt/marshal/inference
COPY rag /opt/marshal/rag
COPY scripts /opt/marshal/scripts
COPY config.py errors.py main.py agentd.py /opt/marshal/

RUN pip install --no-cache-dir --upgrade pip wheel \
    && pip install --no-cache-dir -e ".[rag,remote]"

# Volumes:
#   /models  — mount your downloaded GGUF here
#   /data    — persistent state (audit DB, KV cache slots, RAG index)
ENV MODELS_DIR=/models \
    MARSHAL_DATA_DIR=/data \
    HOME=/data \
    PYTHONUNBUFFERED=1

VOLUME ["/models", "/data"]

# Default: run agentd. Override with `command:` in compose for the API or
# the inference server.
CMD ["python", "/opt/marshal/agentd.py"]
