import argparse
import asyncio
import csv
import hashlib
import json
import logging
import time
from collections import deque
from itertools import islice
from pathlib import Path
from uuid import uuid4

import pandas as pd

from app.config import DATA_PATH, SOURCE_TASK_COUNT
from app.features import candles_to_payload
from app.publisher import PublishRejected, TaskPublisher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def read_candles(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        sample = source.read(4096)
        source.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(source, dialect=dialect)
        for row in reader:
            normalized = {
                str(key).strip().lower(): value for key, value in row.items() if key is not None
            }
            date_value = normalized.get("date", normalized.get("time", normalized.get("datetime")))
            try:
                candle = {
                    "date": (
                        pd.to_datetime(date_value, utc=True).isoformat() if date_value else None
                    ),
                    "open": float(str(normalized["open"]).replace(",", ".")),
                    "high": float(str(normalized["high"]).replace(",", ".")),
                    "low": float(str(normalized["low"]).replace(",", ".")),
                    "close": float(str(normalized["close"]).replace(",", ".")),
                    "volume": float(
                        str(
                            normalized.get("volume", normalized.get("tick_volume", "0")) or "0"
                        ).replace(",", ".")
                    ),
                }
            except (KeyError, TypeError, ValueError):
                continue
            if min(candle["open"], candle["high"], candle["low"], candle["close"]) <= 0:
                continue
            if candle["high"] < max(candle["open"], candle["close"], candle["low"]):
                continue
            if candle["low"] > min(candle["open"], candle["close"], candle["high"]):
                continue
            yield candle


def task_stream(path: Path, batch_id: str, count: int):
    history = deque(maxlen=11)
    generated = 0
    for candle in read_candles(path):
        history.append(candle)
        if len(history) < 11:
            continue
        candles = list(history)
        task_id = str(uuid4())
        payload = {
            "task_id": task_id,
            "candles": candles_to_payload(candles),
            "simulate_failure": False,
            "batch_id": batch_id,
        }
        canonical = {
            key: value for key, value in payload.items() if key not in {"task_id", "batch_id"}
        }
        payload["payload_hash"] = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        yield payload
        generated += 1
        if generated >= count:
            return


async def run_source(path: Path, count: int, batch_id: str) -> dict[str, object]:
    generated = 0
    accepted = 0
    rejected = 0
    retries = 0
    publish_latencies = []
    started = time.perf_counter()

    def save_progress() -> None:
        elapsed = time.perf_counter() - started
        snapshot = {
            "event": "source_progress",
            "batch_id": batch_id,
            "generated": generated,
            "accepted": accepted,
            "rejected_publish_attempts": rejected,
            "retries": retries,
            "elapsed_seconds": elapsed,
            "accepted_per_second": accepted / elapsed if elapsed else 0.0,
        }
        print(json.dumps(snapshot, sort_keys=True), flush=True)

    save_progress()
    async with TaskPublisher() as publisher:
        for payload in islice(task_stream(path, batch_id, count), count):
            generated += 1
            backoff = 0.05
            while True:
                attempt_started = time.perf_counter()
                try:
                    await publisher.publish(payload)
                    publish_latencies.append((time.perf_counter() - attempt_started) * 1000)
                    accepted += 1
                    save_progress()
                    break
                except PublishRejected:
                    rejected += 1
                    retries += 1
                    save_progress()
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 1.0)
    elapsed = time.perf_counter() - started
    summary = {
        "batch_id": batch_id,
        "generated": generated,
        "accepted": accepted,
        "rejected_publish_attempts": rejected,
        "retries": retries,
        "duration_seconds": elapsed,
        "accepted_per_second": accepted / elapsed if elapsed else 0.0,
        "mean_publish_ms": (
            sum(publish_latencies) / len(publish_latencies) if publish_latencies else 0.0
        ),
    }
    logger.info("source_finished %s", json.dumps(summary, sort_keys=True))
    print(json.dumps(summary, sort_keys=True))
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=SOURCE_TASK_COUNT)
    parser.add_argument("--batch-id", default=f"batch-{uuid4().hex[:12]}")
    parser.add_argument("--data-path", default=DATA_PATH)
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("count must be positive")
    path = Path(args.data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    summary = await run_source(path, args.count, args.batch_id)
    if summary["generated"] != args.count:
        raise RuntimeError(
            f"Dataset produced only {summary['generated']} tasks, requested {args.count}"
        )


if __name__ == "__main__":
    asyncio.run(main())
