import argparse
import csv
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime, UTC
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
load_dotenv(ROOT / ".env")
BROKER_QUEUE_MAX_LENGTH = int(os.getenv("QUEUE_MAX_LENGTH", "11"))
MANAGEMENT_URL = os.getenv("RABBITMQ_MANAGEMENT_URL", "http://localhost:15672")
API_URL = "http://localhost:8000"
AUTH = (
    os.getenv("RABBITMQ_USER", "hw4"),
    os.getenv("RABBITMQ_PASSWORD", "hw4-rabbit-password"),
)


def compose_file_args(override: str, *, scaled: bool = False) -> list[str]:
    """Return Compose files, disabling host worker ports for multi-worker runs."""
    args = ["-f", "docker-compose.yml", "-f", override]
    if scaled:
        args.extend(["-f", "docker-compose.scaled.yml"])
    return args


def run_compose(
    override: str,
    *args: str,
    check: bool = True,
    scaled: bool = False,
) -> subprocess.CompletedProcess:
    command = ["docker", "compose", *compose_file_args(override, scaled=scaled), *args]
    return subprocess.run(command, cwd=ROOT, check=check, text=True, capture_output=True)


def parse_memory(value: str) -> float:
    match = re.search(r"([0-9.]+)\s*(B|KiB|MiB|GiB|KB|MB|GB)", value)
    if not match:
        return 0.0
    amount = float(match.group(1))
    factor = {
        "B": 1,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
    }[match.group(2)]
    return amount * factor / (1024**2)


def docker_memory() -> tuple[float, float]:
    result = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{.Name}}|{{.MemUsage}}"],
        text=True,
        capture_output=True,
        check=False,
    )
    rabbit_memory = 0.0
    worker_memory = 0.0
    for line in result.stdout.splitlines():
        parts = line.split("|", 1)
        if len(parts) != 2:
            continue
        name, usage = parts
        used = parse_memory(usage)
        if "rabbitmq" in name:
            rabbit_memory = used
        if "worker" in name:
            worker_memory += used
    return rabbit_memory, worker_memory


def purge_queue(queue_name: str) -> None:
    url = f"{MANAGEMENT_URL}/api/queues/%2F/{quote(queue_name, safe='')}/contents"
    response = requests.delete(url, auth=AUTH, timeout=5)
    if response.status_code not in {204, 404}:
        response.raise_for_status()


def queue_snapshot(queue_name: str) -> dict[str, float]:
    url = f"{MANAGEMENT_URL}/api/queues/%2F/{quote(queue_name, safe='')}"
    response = requests.get(url, auth=AUTH, timeout=2)
    response.raise_for_status()
    queue = response.json()
    nodes = requests.get(f"{MANAGEMENT_URL}/api/nodes", auth=AUTH, timeout=2)
    nodes.raise_for_status()
    node_data = nodes.json()
    node_memory = max(
        (float(node.get("mem_used", 0)) for node in node_data),
        default=0.0,
    ) / (1024**2)
    rabbit_container_memory, worker_memory = docker_memory()
    return {
        "ready": float(queue.get("messages_ready", 0)),
        "unacked": float(queue.get("messages_unacknowledged", 0)),
        "rabbit_node_memory_mb": node_memory,
        "rabbit_container_memory_mb": rabbit_container_memory,
        "workers_memory_mb": worker_memory,
    }


def get_batch_metrics(batch_id: str) -> dict:
    response = requests.get(f"{API_URL}/api/v1/metrics", params={"batch_id": batch_id}, timeout=2)
    response.raise_for_status()
    return response.json()


def wait_for_stack() -> None:
    for _ in range(90):
        try:
            rabbit_response = requests.get(f"{MANAGEMENT_URL}/api/overview", auth=AUTH, timeout=1)
            api_response = requests.get(f"{API_URL}/healthz", timeout=1)
            if rabbit_response.ok and api_response.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError("RabbitMQ Management API and task API did not become ready")


def run_scenario(
    name: str,
    override: str,
    queue_name: str,
    worker_count: int,
    count: int,
    sample_interval: float,
) -> tuple[dict, list[dict]]:
    scaled = worker_count > 1
    run_compose(override, "stop", "worker", check=False, scaled=scaled)
    run_compose(override, "up", "-d", "--build", "api", scaled=scaled)
    wait_for_stack()
    purge_queue(queue_name)
    run_compose(
        override,
        "up",
        "-d",
        "--build",
        "--scale",
        f"worker={worker_count}",
        "worker",
        scaled=scaled,
    )
    batch_id = f"{name}-{uuid4().hex[:8]}"
    source_command = [
        "docker",
        "compose",
        *compose_file_args(override, scaled=scaled),
        "--profile",
        "source",
        "run",
        "--rm",
        "source",
        "python",
        "-m",
        "app.source",
        "--count",
        str(count),
        "--batch-id",
        batch_id,
    ]
    samples = []
    started = time.perf_counter()
    output_lines = []
    output_lock = threading.Lock()
    process = subprocess.Popen(
        source_command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def read_output() -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            with output_lock:
                output_lines.append(line)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    while process.poll() is None:
        elapsed = time.perf_counter() - started
        with output_lock:
            current_lines = list(output_lines)
        progress = None
        for line in reversed(current_lines):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                candidate.get("event") == "source_progress"
                and candidate.get("batch_id") == batch_id
            ):
                progress = candidate
                break
        try:
            snapshot = queue_snapshot(queue_name)
            if progress is not None:
                snapshot["source_generated"] = progress["generated"]
                snapshot["source_accepted"] = progress["accepted"]
                snapshot["source_retries"] = progress["retries"]
                snapshot["source_tasks_per_second"] = progress["accepted_per_second"]
            snapshot["elapsed_seconds"] = elapsed
            snapshot["timestamp_utc"] = datetime.now(UTC).isoformat()
            samples.append(snapshot)
        except requests.RequestException:
            pass
        time.sleep(sample_interval)
    return_code = process.wait()
    reader.join(timeout=5)
    with output_lock:
        logs = "".join(output_lines)
    if return_code != 0:
        raise RuntimeError(f"Source failed in {name}:\n{logs[-5000:]}")
    source_summary = None
    for line in reversed(logs.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("batch_id") == batch_id:
            source_summary = candidate
            break
    if source_summary is None:
        raise RuntimeError(f"Source summary was not found in logs for {name}:\n{logs[-2000:]}")
    deadline = time.monotonic() + max(120, count * 2)
    while time.monotonic() < deadline:
        metrics = get_batch_metrics(batch_id)
        if int(metrics["succeeded"]) + int(metrics["failed"]) >= int(source_summary["accepted"]):
            break
        try:
            snapshot = queue_snapshot(queue_name)
            with output_lock:
                current_lines = list(output_lines)
            progress = None
            for line in reversed(current_lines):
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    candidate.get("event") == "source_progress"
                    and candidate.get("batch_id") == batch_id
                ):
                    progress = candidate
                    break
            if progress is not None:
                snapshot["source_generated"] = progress["generated"]
                snapshot["source_accepted"] = progress["accepted"]
                snapshot["source_retries"] = progress["retries"]
                snapshot["source_tasks_per_second"] = 0.0
            snapshot["elapsed_seconds"] = time.perf_counter() - started
            snapshot["timestamp_utc"] = datetime.now(UTC).isoformat()
            samples.append(snapshot)
        except requests.RequestException:
            pass
        time.sleep(sample_interval)
    else:
        raise TimeoutError(f"Batch {batch_id} did not drain before timeout")
    metrics = get_batch_metrics(batch_id)
    elapsed = time.perf_counter() - started
    ready_values = [sample["ready"] for sample in samples]
    summary = {
        "scenario": name,
        "workers": worker_count,
        "queue_limit": 0 if "unbounded" in name else BROKER_QUEUE_MAX_LENGTH + 1,
        "broker_x_max_length": 0 if "unbounded" in name else BROKER_QUEUE_MAX_LENGTH,
        "generated": source_summary["generated"],
        "accepted": source_summary["accepted"],
        "source_tasks_per_second": round(float(source_summary["accepted_per_second"]), 4),
        "source_retries": source_summary["retries"],
        "max_ready_depth": int(max(ready_values, default=0)),
        "mean_ready_depth": round(sum(ready_values) / len(ready_values), 3) if ready_values else 0,
        "max_rabbit_node_memory_mb": round(
            max((sample["rabbit_node_memory_mb"] for sample in samples), default=0),
            3,
        ),
        "max_rabbit_container_memory_mb": round(
            max((sample["rabbit_container_memory_mb"] for sample in samples), default=0),
            3,
        ),
        "max_workers_memory_mb": round(
            max((sample["workers_memory_mb"] for sample in samples), default=0),
            3,
        ),
        "avg_processing_ms": round(float(metrics["avg_processing_ms"] or 0), 3),
        "p95_processing_ms": round(float(metrics["p95_processing_ms"] or 0), 3),
        "completed": int(metrics["succeeded"]),
        "failed": int(metrics["failed"]),
        "experiment_elapsed_seconds": round(elapsed, 3),
        "batch_id": batch_id,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / f"{name}_timeseries.csv").open("w", newline="", encoding="utf-8") as target:
        fields = list(dict.fromkeys(key for sample in samples for key in sample))
        if not fields:
            fields = ["elapsed_seconds", "ready", "unacked"]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(samples)
    return summary, samples


def update_readme(summaries: list[dict]) -> None:
    readme_path = ROOT / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    prefix = text.split("## Измеренные результаты", 1)[0].rstrip()
    columns = [
        ("scenario", "Сценарий"),
        ("workers", "Воркеры"),
        ("broker_x_max_length", "RabbitMQ x-max-length"),
        ("queue_limit", "Целевой максимум Ready"),
        ("source_tasks_per_second", "Источник, задач/с"),
        ("max_ready_depth", "Макс. Ready"),
        ("mean_ready_depth", "Средний Ready"),
        ("max_rabbit_node_memory_mb", "Память RabbitMQ, MiB"),
        ("max_workers_memory_mb", "Память воркеров, MiB"),
        ("avg_processing_ms", "Среднее время задачи, мс"),
        ("p95_processing_ms", "P95 задачи, мс"),
        ("completed", "Успешно"),
        ("failed", "В DLQ"),
        ("source_retries", "Отказы публикации"),
    ]
    lines = [
        "## Измеренные результаты",
        "",
        "Сгенерировано командой `python -m app.experiment`. Время и память измеряются на текущем компьютере, поэтому значения зависят от окружения.",
        "",
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for summary in summaries:
        lines.append("| " + " | ".join(str(summary[key]) for key, _ in columns) + " |")
    lines.extend(
        [
            "",
            "Временные ряды глубины очереди, числа неподтверждённых сообщений, скорости источника и потребления памяти сохранены в `results/*_timeseries.csv`.",
            "",
            "Интерпретация: в bounded-сценариях `Ready` не должен превышать настроенный лимит, а число отказов публикации показывает, сколько раз RabbitMQ сигнализировал об ограничении ёмкости. Источник повторяет публикацию того же сообщения с экспоненциальной задержкой, поэтому задачи не теряются. Сравнение сценариев с одним и двумя воркерами показывает изменение совокупной пропускной способности. В unbounded-сценарии очередь может расти до завершения источника.",
            "Для quorum-очередей RabbitMQ 4.3 режим `reject-publish` может допускать сообщение сверх `x-max-length`: сигнал об отказе приходит издателю после того, как сообщение уже находится в полёте. Поэтому для одного последовательного source используется `x-max-length=11`, а целевой максимум `Ready` равен 12. При нескольких параллельных publisher-каналах превышение может быть больше; это документированное поведение RabbitMQ.",
            "",
        ]
    )
    readme_path.write_text(prefix + "\n\n" + "\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=180)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    args = parser.parse_args()
    if args.count < 20:
        raise ValueError("Use at least 20 tasks to observe queue dynamics")
    scenarios = [
        ("bounded_1_worker", "docker-compose.bounded.yml", "tasks.bounded", 1),
        ("bounded_2_workers", "docker-compose.bounded.yml", "tasks.bounded", 2),
        ("unbounded_1_worker", "docker-compose.unbounded.yml", "tasks.unbounded", 1),
    ]
    summaries = []
    for name, override, queue_name, workers in scenarios:
        print(f"Running {name} with {args.count} tasks and {workers} worker(s)", flush=True)
        summary, _ = run_scenario(
            name,
            override,
            queue_name,
            workers,
            args.count,
            args.sample_interval,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)
    with (RESULTS / "experiment_summary.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    update_readme(summaries)
    run_compose(
        "docker-compose.bounded.yml",
        "up",
        "-d",
        "--build",
        "--scale",
        "worker=1",
        "api",
        "worker",
    )
    print(f"Saved experiment measurements to {RESULTS}")


if __name__ == "__main__":
    main()
