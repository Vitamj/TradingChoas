FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall six==1.17.0 && \
    python -c "import six; assert hasattr(six, 'string_types')"

COPY app ./app

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8001

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
