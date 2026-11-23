import asyncio
import contextvars
import os
import time
from contextlib import asynccontextmanager
from uuid import UUID

import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI
from faststream.middlewares import AckPolicy
from faststream.rabbit.annotations import RabbitMessage
from faststream.rabbit import (
    Channel,
    RabbitBroker,
    RabbitQueue,
    QueueType,
)
from mlflow import MlflowClient
from opentelemetry import context as otel_context, propagate, trace
from opentelemetry.trace import SpanKind
from prometheus_client import make_asgi_app

from app.config import (
    GRACEFUL_TIMEOUT_SECONDS,
    INFERENCE_DELAY_SECONDS,
    MAX_DELIVERIES,
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    MODEL_NAME,
    PREFETCH_COUNT,
    RABBITMQ_URL,
    TASK_QUEUE,
    queue_arguments,
)
from app.database import (
    complete_task,
    create_pool,
    ensure_schema,
    fail_task,
    mark_retrying,
    start_task,
)
from app.features import build_feature_frame
from app.metrics import INFERENCE_DURATION, WORKER_ACTIVE, WORKER_REDELIVERIES, WORKER_TASKS
from app.telemetry import bind_task, instrument_fastapi, logger, reset_task, tracer

broker = RabbitBroker(
    RABBITMQ_URL,
    default_channel=Channel(prefetch_count=PREFETCH_COUNT),
    graceful_timeout=GRACEFUL_TIMEOUT_SECONDS,
)
task_queue = RabbitQueue(
    TASK_QUEUE,
    durable=True,
    queue_type=QueueType.QUORUM,
    arguments={key: value for key, value in queue_arguments(include_type=False).items()},
)
DB_POOL = None
MODEL = None
MODEL_VERSION: str | None = None
MODEL_RUN_ID: str | None = None
ACTIVE_TASKS = 0
WORKER_ID = os.getenv("HOSTNAME", "worker-local")


def _delivery_count(message: RabbitMessage) -> int:
    """Return RabbitMQ's quorum-queue failed-delivery counter.

    RabbitMQ 4.x exposes this value in the ``x-delivery-count`` header.
    A missing or malformed header is treated as the first delivery.
    """
    headers = message.headers or {}
    try:
        return max(0, int(headers.get("x-delivery-count", 0)))
    except (TypeError, ValueError):
        return 0


async def _retry_or_dead_letter(
    message: RabbitMessage,
    task_id: UUID | None,
    error: str,
) -> None:
    """Retry a failed delivery a bounded number of times, then send it to DLQ.

    ``basic.nack(requeue=True)`` does not increment RabbitMQ 4.3's
    ``delivery-count``. FastStream's ``reject(requeue=True)`` maps to
    ``basic.reject`` and therefore counts a failed delivery for quorum queues.
    """
    delivery_count = _delivery_count(message)
    if delivery_count >= MAX_DELIVERIES:
        if task_id is not None:
            try:
                # Persist FAILED before the terminal reject whenever the DB is
                # available. If this write fails, the message is still retained
                # by the durable DLQ and can be reconciled later.
                await fail_task(DB_POOL, task_id, error)
            except Exception:
                logger.exception("task_final_state_failed")
        WORKER_TASKS.labels("dlq", WORKER_ID).inc()
        logger.error(
            "task_dead_lettered", delivery_count=delivery_count, max_deliveries=MAX_DELIVERIES
        )
        await message.reject(requeue=False)
        return

    if task_id is not None:
        try:
            await mark_retrying(DB_POOL, task_id, error)
        except Exception:
            logger.exception("task_retry_state_failed")
    WORKER_REDELIVERIES.labels(WORKER_ID).inc()
    # Use reject rather than nack: RabbitMQ counts basic.reject as a failed
    # delivery for quorum-queue delivery-limit enforcement.
    await message.reject(requeue=True)


async def load_registered_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    version = await asyncio.to_thread(client.get_model_version_by_alias, MODEL_NAME, MODEL_ALIAS)
    model = await asyncio.to_thread(
        mlflow.sklearn.load_model,
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}",
    )
    return model, str(version.version), str(version.run_id)


@broker.subscriber(
    task_queue,
    ack_policy=AckPolicy.MANUAL,
    channel=Channel(prefetch_count=PREFETCH_COUNT),
)
async def handle_task(payload: dict, message: RabbitMessage) -> None:
    global ACTIVE_TASKS
    if not isinstance(payload, dict):
        logger.warning("invalid_task_payload_type", payload_type=type(payload).__name__)
        await message.reject(requeue=False)
        return

    task_id_value = payload.get("task_id")
    payload_hash = payload.get("payload_hash")
    batch_id = payload.get("batch_id")
    if not task_id_value or not payload_hash or not isinstance(payload.get("candles"), list):
        logger.warning("invalid_task_envelope", supplied_task_id=task_id_value)
        await message.reject(requeue=False)
        return

    try:
        task_id = UUID(str(task_id_value))
    except (TypeError, ValueError, AttributeError):
        logger.warning("invalid_task_id", supplied_task_id=task_id_value)
        await message.reject(requeue=False)
        return

    parent_context = propagate.extract(message.headers or {})
    process_span = tracer.start_span(
        "process",
        context=parent_context,
        kind=SpanKind.CONSUMER,
        attributes={
            "messaging.system": "rabbitmq",
            "messaging.destination.name": TASK_QUEUE,
            "messaging.message.id": str(task_id),
            "messaging.rabbitmq.delivery_count": _delivery_count(message),
        },
    )
    context_token = otel_context.attach(trace.set_span_in_context(process_span))
    task_token = bind_task(task_id)

    try:
        with tracer.start_as_current_span("db.start_task"):
            status = await start_task(DB_POOL, task_id, str(payload_hash), batch_id)
        if status in {"SUCCEEDED", "FAILED"}:
            await message.ack()
            reset_task(task_token)
            otel_context.detach(context_token)
            process_span.end()
            return
    except ValueError as exc:
        # Reusing an ID with a different payload is a permanent contract error,
        # not a transient infrastructure failure worth retrying.
        logger.error("task_envelope_conflict", error=str(exc))
        await message.reject(requeue=False)
        process_span.record_exception(exc)
        process_span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
        reset_task(task_token)
        otel_context.detach(context_token)
        process_span.end()
        return
    except Exception as exc:
        logger.exception("task_start_failed")
        process_span.record_exception(exc)
        process_span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
        await _retry_or_dead_letter(message, task_id, "Could not start task in database")
        reset_task(task_token)
        otel_context.detach(context_token)
        process_span.end()
        return

    ACTIVE_TASKS += 1
    WORKER_ACTIVE.labels(WORKER_ID).inc()
    started = time.perf_counter()
    try:
        if payload.get("simulate_failure"):
            raise RuntimeError("Simulated inference failure")
        with tracer.start_as_current_span(
            "inference",
            attributes={"model.alias": MODEL_ALIAS, "model.version": str(MODEL_VERSION)},
        ):
            inference_started = time.perf_counter()
            features = build_feature_frame(payload["candles"])
            await asyncio.sleep(INFERENCE_DELAY_SECONDS)
            copied_context = contextvars.copy_context()
            prediction = int(
                np.asarray(
                    await asyncio.to_thread(copied_context.run, MODEL.predict, features)
                ).reshape(-1)[0]
            )
            probability_up = None
            if hasattr(MODEL, "predict_proba"):
                copied_context = contextvars.copy_context()
                probabilities = np.asarray(
                    await asyncio.to_thread(copied_context.run, MODEL.predict_proba, features)
                )[0]
                classes = np.asarray(getattr(MODEL, "classes_", [0, 1]))
                indexes = np.flatnonzero(classes == 1)
                if len(indexes):
                    probability_up = float(probabilities[indexes[0]])
            INFERENCE_DURATION.labels(WORKER_ID).observe(time.perf_counter() - inference_started)
        processing_ms = (time.perf_counter() - started) * 1000
        result = {
            "prediction": prediction,
            "direction": "up" if prediction == 1 else "down_or_unchanged",
            "probability_up": probability_up,
            "model_name": MODEL_NAME,
            "model_alias": MODEL_ALIAS,
            "model_version": MODEL_VERSION,
            "run_id": MODEL_RUN_ID,
        }
        with tracer.start_as_current_span("db.write_result"):
            await complete_task(
                DB_POOL,
                task_id,
                str(payload_hash),
                batch_id,
                result,
                str(MODEL_VERSION),
                str(MODEL_RUN_ID),
                processing_ms,
            )
        await message.ack()
        WORKER_TASKS.labels("ok", WORKER_ID).inc()
        logger.info("task_succeeded", model_version=MODEL_VERSION, processing_ms=processing_ms)
    except Exception as exc:
        WORKER_TASKS.labels("error", WORKER_ID).inc()
        process_span.record_exception(exc)
        process_span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
        logger.exception("task_processing_failed", delivery_attempt=_delivery_count(message) + 1)
        await _retry_or_dead_letter(message, task_id, str(exc))
    finally:
        ACTIVE_TASKS = max(0, ACTIVE_TASKS - 1)
        WORKER_ACTIVE.labels(WORKER_ID).dec()
        reset_task(task_token)
        otel_context.detach(context_token)
        process_span.end()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global DB_POOL, MODEL, MODEL_VERSION, MODEL_RUN_ID
    DB_POOL = await create_pool()
    await ensure_schema(DB_POOL)
    MODEL, MODEL_VERSION, MODEL_RUN_ID = await load_registered_model()
    logger.info(
        "model_loaded",
        model=MODEL_NAME,
        alias=MODEL_ALIAS,
        version=MODEL_VERSION,
        run_id=MODEL_RUN_ID,
        prefetch_count=PREFETCH_COUNT,
    )
    await broker.start()
    app.state.model_version = MODEL_VERSION
    app.state.model_run_id = MODEL_RUN_ID
    try:
        yield
    finally:
        logger.info("graceful_shutdown_started", active_tasks=ACTIVE_TASKS)
        await broker.stop()
        await DB_POOL.close()
        DB_POOL = None
        logger.info("graceful_shutdown_finished")


app = FastAPI(title="FastStream Inference Worker", version="1.0.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())
instrument_fastapi(app)


@app.get("/healthz")
async def healthz() -> dict[str, object]:
    return {
        "status": "ok" if MODEL is not None else "starting",
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_version": MODEL_VERSION,
        "prefetch_count": PREFETCH_COUNT,
        "active_tasks": ACTIVE_TASKS,
        "inference_delay_seconds": INFERENCE_DELAY_SECONDS,
    }
