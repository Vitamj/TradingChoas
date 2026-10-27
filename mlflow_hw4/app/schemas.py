import hashlib
import json
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from app.features import MIN_HISTORY_ROWS


class Candle(BaseModel):
    date: datetime | None = None
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_ohlc(self):
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high must be greater than or equal to open, close and low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low must be less than or equal to open, close and high")
        return self


class TaskRequest(BaseModel):
    task_id: UUID = Field(default_factory=uuid4)
    candles: list[Candle] = Field(min_length=MIN_HISTORY_ROWS, max_length=200)
    simulate_failure: bool = False
    batch_id: str | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def validate_history(self):
        dates = [candle.date for candle in self.candles]
        present = [value is not None for value in dates]
        if any(present) and not all(present):
            raise ValueError("Provide dates for every candle or omit dates for every candle")
        if all(present):
            if len(set(dates)) != len(dates):
                raise ValueError("Candle dates must be unique")
            if dates != sorted(dates):
                raise ValueError("Candle dates must be in ascending order")
        return self


def payload_digest(payload: TaskRequest) -> str:
    value = payload.model_dump(mode="json", exclude={"task_id", "batch_id"})
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_envelope(payload: TaskRequest, batch_id: str | None = None) -> dict[str, Any]:
    value = payload.model_dump(mode="json")
    value["payload_hash"] = payload_digest(payload)
    value["batch_id"] = batch_id if batch_id is not None else payload.batch_id
    return value


class TaskAccepted(BaseModel):
    task_id: UUID
    status: str
    idempotent_replay: bool = False


class TaskStatus(BaseModel):
    task_id: UUID
    status: str
    result: dict[str, Any] | None = None
    model_version: str | None = None
    run_id: str | None = None
    processing_ms: float | None = None
    error: str | None = None
    batch_id: str | None = None


class WorkerHealth(BaseModel):
    status: str
    model_name: str
    model_alias: str
    model_version: str | None
    prefetch_count: int
    active_tasks: int
