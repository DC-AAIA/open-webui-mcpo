FROM python:3.12-slim-bookworm
# Install uv (from official binary), nodejs, npm, and git
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*
# Install Node.js and npm via NodeSource
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
 && apt-get install -y nodejs \
 && rm -rf /var/lib/apt/lists/*
# One-time version check (optional diagnostics)
RUN node -v && npm -v
# Install MCP remote package for GitMCP support
RUN npm install -g mcp-remote
# CACHE BUST: put BEFORE COPY so subsequent layers rebuild
ARG BUILD_REV=2025-08-20T12-59-00Z
ENV BUILD_REV=${BUILD_REV}
RUN echo "BUILD_REV=${BUILD_REV}" > /build-rev.txt
# Bring in source
WORKDIR /app
COPY . /app
# Create virtual environment explicitly in known location
ENV VIRTUAL_ENV=/app/.venv
RUN uv venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Update MCP to latest version for StdioClientTransport compatibility
RUN pip install --upgrade mcp>=1.13.0
# Install mcpo from local source (pyproject.toml must be present)
RUN uv pip install --reinstall --no-cache-dir . && rm -rf ~/.cache
# Verify mcpo installed correctly
RUN which mcpo
# Uvicorn in this repo listens on 8080 by default
EXPOSE 8080
# Entrypoint for container runs
ENTRYPOINT ["mcpo"]
# Default command (can be overridden by Railway Start Command)
CMD ["--help"]
