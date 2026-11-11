# Build Stage: Compile and prepare wheels
FROM python:3.12-slim AS builder

WORKDIR /build
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# Final Runtime Stage: Minimal, hardened, non-root container
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="EvalForge" \
      org.opencontainers.image.description="Enterprise Human-in-the-Loop & LLM-as-a-Judge Evaluation Platform" \
      org.opencontainers.image.authors="SriRam Kunamsetty <mohansriramkunamsetty@gmail.com>" \
      org.opencontainers.image.source="https://github.com/SriRamKunamsetty/evalforge"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    PORT=8083

RUN useradd --uid 10001 --create-home app && \
    mkdir /data && \
    chown -R 10001:10001 /data

WORKDIR /app

COPY --from=builder /build/wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

COPY evalforge ./evalforge

USER 10001
EXPOSE 8083

HEALTHCHECK --interval=20s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8083/healthz')" || exit 1

CMD ["gunicorn", "-b", "0.0.0.0:8083", "-w", "2", "--worker-tmp-dir", "/dev/shm", "--access-logfile", "-", "evalforge.server:create_app()"]
