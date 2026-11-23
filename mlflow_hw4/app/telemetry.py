import logging
import os
import sys
from contextvars import ContextVar, Token
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

task_id_var: ContextVar[str | None] = ContextVar("task_id", default=None)
_configured = False
_tracing_configured = False


def _add_context(_logger: Any, _method: str, event: dict[str, Any]) -> dict[str, Any]:
    task_id = task_id_var.get()
    if task_id:
        event["task_id"] = task_id
    span = trace.get_current_span()
    context = span.get_span_context()
    if context.is_valid:
        event["trace_id"] = f"{context.trace_id:032x}"
        event["span_id"] = f"{context.span_id:016x}"
    event.setdefault("service", os.getenv("OTEL_SERVICE_NAME", "unknown"))
    event.setdefault("env", os.getenv("DEPLOYMENT_ENVIRONMENT", "dev"))
    return event


def configure_logging() -> None:
    global _configured
    if _configured:
        return
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    shared = [
        structlog.contextvars.merge_contextvars,
        _add_context,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    structlog.configure(
        processors=shared + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def configure_tracing() -> None:
    global _tracing_configured
    if _tracing_configured or os.getenv("OTEL_SDK_DISABLED", "false").lower() == "true":
        return
    resource = Resource.create(
        {
            "service.name": os.getenv("OTEL_SERVICE_NAME", "unknown"),
            "service.version": os.getenv("SERVICE_VERSION", "5.0.0"),
            "deployment.environment.name": os.getenv("DEPLOYMENT_ENVIRONMENT", "dev"),
            "service.instance.id": os.getenv("HOSTNAME", "local"),
        }
    )
    provider = TracerProvider(resource=resource)
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    try:
        AsyncPGInstrumentor().instrument()
    except Exception:
        logging.getLogger(__name__).debug("asyncpg instrumentation already enabled")
    _tracing_configured = True


def instrument_fastapi(app: Any) -> None:
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="healthz,metrics",
    )


def bind_task(task_id: Any) -> Token:
    return task_id_var.set(str(task_id))


def reset_task(token: Token) -> None:
    task_id_var.reset(token)


configure_logging()
configure_tracing()
logger = structlog.get_logger()
tracer = trace.get_tracer("trading-chaos.pipeline")
