FROM python:3.10-slim

# Install system dependencies
# libgl1: required for opencv/doctr (replaces libgl1-mesa-glx)
# libcairo2, libpango*: required for doctR/weasyprint type tasks
# git: for potential git dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency definitions
COPY pyproject.toml uv.lock requirements.txt ./

# Install dependencies
# Using uv sync to install into the system python environment
RUN uv sync --frozen

# Copy the rest of the application
COPY src/ src/
COPY models/README.md models/README.md

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Expose the application port (assuming FastAPI default or configured port)
EXPOSE 8000

# Run the application
# Using the installed python from the virtualenv created by uv or system python if configured
CMD ["uv", "run", "bookipedia-ai"]
