FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --upgrade pip uv && \
    uv pip install --system -e .

COPY . .

CMD ["pytest", "tests/test_pipeline.py", "-v"]
