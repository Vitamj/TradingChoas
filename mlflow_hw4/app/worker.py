import asyncio
import logging
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

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
                logger.exception("task_final_state_failed task_id=%s", task_id)
        logger.error(
            "task_dead_lettered task_id=%s delivery_count=%s max_deliveries=%s",
            task_id,
            delivery_count,
            MAX_DELIVERIES,
        )
        await message.reject(requeue=False)
        return

    if task_id is not None:
        try:
            await mark_retrying(DB_POOL, task_id, error)
        except Exception:
            logger.exception("task_retry_state_failed task_id=%s", task_id)
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
        logger.warning("invalid_task_payload_type type=%s", type(payload).__name__)
        await message.reject(requeue=False)
        return

    task_id_value = payload.get("task_id")
    payload_hash = payload.get("payload_hash")
    batch_id = payload.get("batch_id")
    if not task_id_value or not payload_hash or not isinstance(payload.get("candles"), list):
        logger.warning("invalid_task_envelope task_id=%s", task_id_value)
        await message.reject(requeue=False)
        return

    try:
        task_id = UUID(str(task_id_value))
    except (TypeError, ValueError, AttributeError):
        logger.warning("invalid_task_id task_id=%s", task_id_value)
        await message.reject(requeue=False)
        return

    try:
        status = await start_task(DB_POOL, task_id, str(payload_hash), batch_id)
        if status in {"SUCCEEDED", "FAILED"}:
            await message.ack()
            return
    except ValueError as exc:
        # Reusing an ID with a different payload is a permanent contract error,
        # not a transient infrastructure failure worth retrying.
        logger.error("task_envelope_conflict task_id=%s error=%s", task_id, exc)
        await message.reject(requeue=False)
        return
    except Exception:
        logger.exception("task_start_failed task_id=%s", task_id_value)
        await _retry_or_dead_letter(message, task_id, "Could not start task in database")
        return

    ACTIVE_TASKS += 1
    started = time.perf_counter()
    try:
        if payload.get("simulate_failure"):
            raise RuntimeError("Simulated inference failure")
        features = build_feature_frame(payload["candles"])
        await asyncio.sleep(INFERENCE_DELAY_SECONDS)
        prediction = int(
            np.asarray(await asyncio.to_thread(MODEL.predict, features)).reshape(-1)[0]
        )
        probability_up = None
        if hasattr(MODEL, "predict_proba"):
            probabilities = np.asarray(await asyncio.to_thread(MODEL.predict_proba, features))[0]
            classes = np.asarray(getattr(MODEL, "classes_", [0, 1]))
            indexes = np.flatnonzero(classes == 1)
            if len(indexes):
                probability_up = float(probabilities[indexes[0]])
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
        logger.info(
            "task_succeeded task_id=%s model_version=%s processing_ms=%.3f",
            task_id,
            MODEL_VERSION,
            processing_ms,
        )
    except Exception as exc:
        logger.exception("task_processing_failed task_id=%s", task_id_value)
        await _retry_or_dead_letter(message, task_id, str(exc))
    finally:
        ACTIVE_TASKS = max(0, ACTIVE_TASKS - 1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global DB_POOL, MODEL, MODEL_VERSION, MODEL_RUN_ID
    DB_POOL = await create_pool()
    await ensure_schema(DB_POOL)
    MODEL, MODEL_VERSION, MODEL_RUN_ID = await load_registered_model()
    logger.info(
        "model_loaded model=%s alias=%s version=%s run_id=%s prefetch_count=%s",
        MODEL_NAME,
        MODEL_ALIAS,
        MODEL_VERSION,
        MODEL_RUN_ID,
        PREFETCH_COUNT,
    )
    await broker.start()
    app.state.model_version = MODEL_VERSION
    app.state.model_run_id = MODEL_RUN_ID
    try:
        yield
    finally:
        await broker.stop()
        await DB_POOL.close()
        DB_POOL = None


app = FastAPI(title="FastStream Inference Worker", version="1.0.0", lifespan=lifespan)


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
