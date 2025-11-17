# Production Dockerfile for Quant Trading Pipeline
# Multi-stage build for optimization

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app user
RUN groupadd -r quant && useradd -r -g quant quant

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ============================================================================
# Development stage
# ============================================================================
FROM base as development

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt

# Copy application code
COPY . .

# Set ownership
RUN chown -R quant:quant /app

USER quant

CMD ["python", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]

# ============================================================================
# Production stage
# ============================================================================
FROM base as production

# Copy requirements
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy only necessary application code
COPY --chown=quant:quant . /app

# Create necessary directories
RUN mkdir -p /app/data_layer /app/backtest /app/logs /app/models && \
    chown -R quant:quant /app

USER quant

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command (override as needed)
CMD ["python", "main.py"]

# ============================================================================
# Airflow worker stage (for orchestration)
# ============================================================================
FROM apache/airflow:2.7.0-python3.11 as airflow

USER root

# Install additional system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements
COPY requirements.txt /tmp/

# Install Python dependencies
RUN pip install --user -r /tmp/requirements.txt

# Copy DAGs and plugins
COPY --chown=airflow:airflow infrastructure/airflow/dags /opt/airflow/dags
COPY --chown=airflow:airflow infrastructure/airflow/plugins /opt/airflow/plugins

# ============================================================================
# Testing stage
# ============================================================================
FROM base as testing

# Install test dependencies
COPY requirements.txt requirements-test.txt ./

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-test.txt

# Copy test code
COPY . .

# Run tests
RUN chown -R quant:quant /app

USER quant

CMD ["pytest", "tests/", "-v", "--cov=.", "--cov-report=html"]
