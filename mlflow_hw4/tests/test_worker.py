from uuid import uuid4

import numpy as np
import pytest
from faststream.rabbit.testing import TestRabbitBroker

import app.worker as worker
from app.features import FEATURE_COLUMNS


class FakeModel:
    classes_ = np.array([0, 1])

    def predict(self, features):
        assert list(features.columns) == FEATURE_COLUMNS
        return np.array([1])

    def predict_proba(self, features):
        return np.array([[0.2, 0.8]])


def sample_envelope():
    closes = [80.0, 80.4, 80.1, 80.8, 81.0, 80.7, 81.2, 81.4, 81.1, 81.6, 82.0]
    candles = [
        {
            "date": f"2024-01-{index + 1:02d}T00:00:00+00:00",
            "open": close,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": 100,
        }
        for index, close in enumerate(closes)
    ]
    return {
        "task_id": str(uuid4()),
        "payload_hash": "test-hash",
        "batch_id": "test-batch",
        "candles": candles,
        "simulate_failure": False,
    }


class FakeMessage:
    def __init__(self, headers=None):
        self.headers = headers or {}
        self.reject_calls = []

    async def reject(self, requeue=False):
        self.reject_calls.append(requeue)


@pytest.mark.asyncio
async def test_test_rabbit_broker_runs_handler_and_persists_before_ack(monkeypatch):
    calls = []

    async def start_task(pool, task_id, payload_hash, batch_id):
        calls.append(("start", task_id))
        return "PROCESSING"

    async def complete_task(
        pool,
        task_id,
        payload_hash,
        batch_id,
        result,
        model_version,
        run_id,
        processing_ms,
    ):
        calls.append(("complete", result["prediction"], model_version, run_id))

    async def mark_retrying(pool, task_id, error):
        calls.append(("retry", task_id))

    monkeypatch.setattr(worker, "DB_POOL", object())
    monkeypatch.setattr(worker, "MODEL", FakeModel())
    monkeypatch.setattr(worker, "MODEL_VERSION", "2")
    monkeypatch.setattr(worker, "MODEL_RUN_ID", "run-2")
    monkeypatch.setattr(worker, "INFERENCE_DELAY_SECONDS", 0)
    monkeypatch.setattr(worker, "start_task", start_task)
    monkeypatch.setattr(worker, "complete_task", complete_task)
    monkeypatch.setattr(worker, "mark_retrying", mark_retrying)

    async with TestRabbitBroker(worker.broker) as test_broker:
        await test_broker.publish(sample_envelope(), queue=worker.TASK_QUEUE)

    assert calls[0][0] == "start"
    assert calls[1] == ("complete", 1, "2", "run-2")
    assert not any(call[0] == "retry" for call in calls)


@pytest.mark.asyncio
async def test_completed_task_is_not_inferred_twice(monkeypatch):
    calls = []

    async def start_task(pool, task_id, payload_hash, batch_id):
        return "SUCCEEDED"

    async def complete_task(*args):
        calls.append("complete")

    monkeypatch.setattr(worker, "DB_POOL", object())
    monkeypatch.setattr(worker, "MODEL", FakeModel())
    monkeypatch.setattr(worker, "MODEL_VERSION", "2")
    monkeypatch.setattr(worker, "MODEL_RUN_ID", "run-2")
    monkeypatch.setattr(worker, "start_task", start_task)
    monkeypatch.setattr(worker, "complete_task", complete_task)

    async with TestRabbitBroker(worker.broker) as test_broker:
        await test_broker.publish(sample_envelope(), queue=worker.TASK_QUEUE)

    assert calls == []


@pytest.mark.asyncio
async def test_processing_failure_uses_reject_with_requeue_before_delivery_limit(monkeypatch):
    calls = []

    async def mark_retrying(pool, task_id, error):
        calls.append(("retry", task_id, error))

    async def fail_task(pool, task_id, error):
        calls.append(("fail", task_id, error))

    monkeypatch.setattr(worker, "DB_POOL", object())
    monkeypatch.setattr(worker, "MAX_DELIVERIES", 3)
    monkeypatch.setattr(worker, "mark_retrying", mark_retrying)
    monkeypatch.setattr(worker, "fail_task", fail_task)
    message = FakeMessage({"x-delivery-count": 2})
    task_id = uuid4()

    await worker._retry_or_dead_letter(message, task_id, "temporary failure")

    assert message.reject_calls == [True]
    assert calls == [("retry", task_id, "temporary failure")]


@pytest.mark.asyncio
async def test_processing_failure_is_dead_lettered_after_delivery_limit(monkeypatch):
    calls = []

    async def mark_retrying(pool, task_id, error):
        calls.append(("retry", task_id, error))

    async def fail_task(pool, task_id, error):
        calls.append(("fail", task_id, error))

    monkeypatch.setattr(worker, "DB_POOL", object())
    monkeypatch.setattr(worker, "MAX_DELIVERIES", 3)
    monkeypatch.setattr(worker, "mark_retrying", mark_retrying)
    monkeypatch.setattr(worker, "fail_task", fail_task)
    message = FakeMessage({"x-delivery-count": 3})
    task_id = uuid4()

    await worker._retry_or_dead_letter(message, task_id, "poison message")

    assert message.reject_calls == [False]
    assert calls == [("fail", task_id, "poison message")]
