# 🏆 PRODUCTION DOCKERFILE - Multi-stage build for optimal size and security

# =================== BUILDER STAGE ===================
FROM python:3.11-slim as builder

# Set build arguments
ARG ENVIRONMENT=production
ARG BUILD_DATE
ARG VERSION=3.0.0

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY requirements/ /tmp/requirements/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements/prod.txt

# =================== PRODUCTION STAGE ===================
FROM python:3.11-slim as production

# Set labels for container metadata
LABEL maintainer="Auto-Analyst Team" \
      version="${VERSION}" \
      description="Auto-Analyst Platform - Production Container" \
      build-date="${BUILD_DATE}"

# Create non-root user for security
RUN groupadd -r autoanalyst && useradd -r -g autoanalyst -s /bin/false autoanalyst

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create application directories
WORKDIR /app
RUN mkdir -p /app/uploads /app/models /app/logs \
    && chown -R autoanalyst:autoanalyst /app

# Copy application code
COPY --chown=autoanalyst:autoanalyst backend/ /app/backend/
COPY --chown=autoanalyst:autoanalyst frontend/ /app/frontend/
COPY --chown=autoanalyst:autoanalyst migrations/ /app/migrations/
COPY --chown=autoanalyst:autoanalyst alembic.ini /app/

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Switch to non-root user
USER autoanalyst

# Expose port
EXPOSE 8000

# Run database migrations and start application
CMD ["sh", "-c", "alembic upgrade head && uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT --workers 4"]
