import json
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

import aio_pika
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import make_asgi_app

from app.config import RETRY_AFTER_SECONDS, RABBITMQ_URL, TASK_QUEUE
from app.database import (
    batch_metrics,
    create_pending,
    create_pool,
    delete_pending,
    ensure_schema,
    get_task,
)
from app.publisher import PublishRejected, TaskPublisher
from app.schemas import TaskAccepted, TaskRequest, TaskStatus, make_envelope, payload_digest
from app.metrics import API_DURATION, API_REQUESTS
from app.telemetry import bind_task, instrument_fastapi, logger, reset_task, tracer


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("service_starting")
    pool = await create_pool()
    await ensure_schema(pool)
    publisher = TaskPublisher(RABBITMQ_URL)
    await publisher.connect()
    app.state.db_pool = pool
    app.state.publisher = publisher
    try:
        yield
    finally:
        logger.info("graceful_shutdown_started")
        await publisher.close()
        await pool.close()
        logger.info("graceful_shutdown_finished")


app = FastAPI(title="Backpressure Task API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def observe_request(request: Request, call_next):
    started = time.perf_counter()
    status = 500
    route = request.url.path
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        matched_route = request.scope.get("route")
        route = getattr(matched_route, "path", route)
        if route not in {"/healthz", "/metrics", "/metrics/"}:
            API_REQUESTS.labels(route, request.method, str(status)).inc()
            API_DURATION.labels(route, request.method).observe(time.perf_counter() - started)


app.mount("/metrics", make_asgi_app())
instrument_fastapi(app)


async def submit_task(payload: TaskRequest, pool, publisher) -> tuple[TaskAccepted, int]:
    token = bind_task(payload.task_id)
    digest = payload_digest(payload)
    with tracer.start_as_current_span("db.create_pending"):
        row = await create_pending(pool, payload.task_id, digest, payload.batch_id)
    if row is None:
        existing = await get_task(pool, payload.task_id)
        if existing is None:
            reset_task(token)
            raise HTTPException(status_code=503, detail="Task storage is temporarily unavailable")
        if existing["payload_hash"] != digest:
            reset_task(token)
            raise HTTPException(
                status_code=409,
                detail="task_id was already used with a different payload",
            )
        result = (
            TaskAccepted(
                task_id=payload.task_id,
                status=existing["status"],
                idempotent_replay=True,
            ),
            200,
        )
        reset_task(token)
        return result

    envelope = make_envelope(payload)
    try:
        await publisher.publish(envelope)
    except (PublishRejected, aio_pika.exceptions.DeliveryError) as exc:
        await delete_pending(pool, payload.task_id)
        logger.warning("task_rejected", reason=str(exc), queue=TASK_QUEUE)
        reset_task(token)
        raise HTTPException(
            status_code=503,
            detail="The processing queue is full. Retry the request later.",
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
        ) from exc
    except Exception as exc:
        await delete_pending(pool, payload.task_id)
        logger.exception("task_publish_failed")
        reset_task(token)
        raise HTTPException(
            status_code=503,
            detail="The processing service is temporarily unavailable.",
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
        ) from exc

    logger.info("task_accepted", queue=TASK_QUEUE)
    reset_task(token)
    return TaskAccepted(task_id=payload.task_id, status="PENDING", idempotent_replay=False), 202


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/tasks", response_model=TaskAccepted)
async def create_task(payload: TaskRequest, request: Request, response: Response) -> Any:
    result, status_code = await submit_task(
        payload,
        request.app.state.db_pool,
        request.app.state.publisher,
    )
    response.status_code = status_code
    return result


@app.get("/api/v1/tasks/{task_id}", response_model=TaskStatus)
async def task_status(task_id: UUID, request: Request) -> TaskStatus:
    row = await get_task(request.app.state.db_pool, task_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Task not found")
    result_json = row.get("result_json")
    if isinstance(result_json, str):
        result_json = json.loads(result_json)
    return TaskStatus(
        task_id=row["task_id"],
        status=row["status"],
        result=result_json,
        model_version=row.get("model_version"),
        run_id=row.get("run_id"),
        processing_ms=row.get("processing_ms"),
        error=row.get("error"),
        batch_id=row.get("batch_id"),
    )


@app.get("/api/v1/metrics")
async def metrics(request: Request, batch_id: str | None = None) -> dict[str, Any]:
    return await batch_metrics(request.app.state.db_pool, batch_id)
