import pytest
from fastapi import HTTPException

import app.api as api
from app.publisher import PublishRejected
from app.schemas import TaskRequest, payload_digest


def sample_payload():
    closes = [80.0, 80.4, 80.1, 80.8, 81.0, 80.7, 81.2, 81.4, 81.1, 81.6, 82.0]
    candles = [
        {
            "date": f"2024-01-{index + 1:02d}T00:00:00Z",
            "open": close,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": 100,
        }
        for index, close in enumerate(closes)
    ]
    return TaskRequest(candles=candles)


@pytest.mark.asyncio
async def test_queue_rejection_returns_503_and_removes_pending(monkeypatch):
    payload = sample_payload()
    deleted = []

    async def create_pending(pool, task_id, digest, batch_id):
        return {"task_id": task_id, "payload_hash": digest, "status": "PENDING"}

    async def delete_pending(pool, task_id):
        deleted.append(task_id)

    class Publisher:
        async def publish(self, envelope):
            raise PublishRejected("queue full")

    monkeypatch.setattr(api, "create_pending", create_pending)
    monkeypatch.setattr(api, "delete_pending", delete_pending)

    with pytest.raises(HTTPException) as error:
        await api.submit_task(payload, object(), Publisher())

    assert error.value.status_code == 503
    assert error.value.headers["Retry-After"] == str(api.RETRY_AFTER_SECONDS)
    assert deleted == [payload.task_id]


@pytest.mark.asyncio
async def test_repeated_task_id_is_idempotent(monkeypatch):
    payload = sample_payload()
    digest = payload_digest(payload)
    published = []

    async def create_pending(pool, task_id, value_hash, batch_id):
        return None

    async def get_task(pool, task_id):
        return {"task_id": task_id, "payload_hash": digest, "status": "SUCCEEDED"}

    class Publisher:
        async def publish(self, envelope):
            published.append(envelope)

    monkeypatch.setattr(api, "create_pending", create_pending)
    monkeypatch.setattr(api, "get_task", get_task)

    result, status_code = await api.submit_task(payload, object(), Publisher())

    assert status_code == 200
    assert result.idempotent_replay is True
    assert result.status == "SUCCEEDED"
    assert published == []


@pytest.mark.asyncio
async def test_same_task_id_with_different_payload_is_rejected(monkeypatch):
    payload = sample_payload()
    other_hash = "different-payload-hash"

    async def create_pending(pool, task_id, digest, batch_id):
        return None

    async def get_task(pool, task_id):
        return {"task_id": task_id, "payload_hash": other_hash, "status": "SUCCEEDED"}

    monkeypatch.setattr(api, "create_pending", create_pending)
    monkeypatch.setattr(api, "get_task", get_task)

    with pytest.raises(HTTPException) as error:
        await api.submit_task(payload, object(), object())

    assert error.value.status_code == 409
