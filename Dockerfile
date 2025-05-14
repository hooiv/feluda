# Use a multi-stage build for smaller final image
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.6.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN groupadd -g 1000 feluda && \
    useradd -u 1000 -g feluda -s /bin/bash -m feluda

# Set working directory
WORKDIR /app

# Copy only the files needed for installation
COPY pyproject.toml README.md ./

# Install dependencies
RUN pip install --upgrade pip && \
    pip install wheel setuptools && \
    pip install .

# Copy the rest of the application
COPY . .

# Build the package
RUN pip install build && \
    python -m build

# Second stage: runtime image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN groupadd -g 1000 feluda && \
    useradd -u 1000 -g feluda -s /bin/bash -m feluda

# Set working directory
WORKDIR /app

# Copy the built package from the builder stage
COPY --from=builder /app/dist/*.whl /tmp/

# Install the package
RUN pip install --upgrade pip && \
    pip install /tmp/*.whl && \
    rm -rf /tmp/*.whl

# Switch to non-root user
USER feluda

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs

# Copy the entrypoint script
COPY --chown=feluda:feluda docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command
CMD ["python", "-m", "feluda"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1
