FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml ./
COPY requirements-api.lock ./
COPY trading_chaos_api ./trading_chaos_api
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements-api.lock \
    && /opt/venv/bin/pip install --no-cache-dir --no-deps .

FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
ARG APP_REVISION=local
ENV APP_REVISION=$APP_REVISION

RUN useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"

CMD ["uvicorn", "trading_chaos_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
