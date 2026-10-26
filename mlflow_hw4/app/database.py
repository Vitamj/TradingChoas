import json
from typing import Any
from uuid import UUID

import asyncpg

from app.config import DATABASE_URL

SCHEMA = """
CREATE TABLE IF NOT EXISTS task_results (
    task_id UUID PRIMARY KEY,
    payload_hash TEXT NOT NULL,
    batch_id TEXT,
    status TEXT NOT NULL CHECK (status IN ('PENDING', 'PROCESSING', 'RETRYING', 'SUCCEEDED', 'FAILED')),
    result_json JSONB,
    model_version TEXT,
    run_id TEXT,
    processing_ms DOUBLE PRECISION,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
)
"""


async def create_pool() -> asyncpg.Pool:
    return await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=12, command_timeout=10)


async def ensure_schema(pool: asyncpg.Pool) -> None:
    await pool.execute(SCHEMA)
    await pool.execute(
        "CREATE INDEX IF NOT EXISTS ix_task_results_batch_id ON task_results(batch_id)"
    )
    await pool.execute("CREATE INDEX IF NOT EXISTS ix_task_results_status ON task_results(status)")


async def create_pending(
    pool: asyncpg.Pool,
    task_id: UUID,
    payload_hash: str,
    batch_id: str | None,
) -> dict[str, Any] | None:
    row = await pool.fetchrow(
        """
        INSERT INTO task_results(task_id, payload_hash, batch_id, status)
        VALUES($1, $2, $3, 'PENDING')
        ON CONFLICT(task_id) DO NOTHING
        RETURNING task_id, payload_hash, batch_id, status
        """,
        task_id,
        payload_hash,
        batch_id,
    )
    return dict(row) if row else None


async def get_task(pool: asyncpg.Pool, task_id: UUID) -> dict[str, Any] | None:
    row = await pool.fetchrow(
        """
        SELECT task_id, payload_hash, batch_id, status, result_json, model_version,
               run_id, processing_ms, error, created_at, updated_at, completed_at
        FROM task_results WHERE task_id = $1
        """,
        task_id,
    )
    if row is None:
        return None
    value = dict(row)
    if isinstance(value.get("result_json"), str):
        value["result_json"] = json.loads(value["result_json"])
    return value


async def delete_pending(pool: asyncpg.Pool, task_id: UUID) -> None:
    await pool.execute(
        "DELETE FROM task_results WHERE task_id = $1 AND status = 'PENDING'",
        task_id,
    )


async def start_task(
    pool: asyncpg.Pool,
    task_id: UUID,
    payload_hash: str,
    batch_id: str | None,
) -> str:
    row = await pool.fetchrow(
        """
        INSERT INTO task_results(task_id, payload_hash, batch_id, status)
        VALUES($1, $2, $3, 'PROCESSING')
        ON CONFLICT(task_id) DO UPDATE SET
            status = CASE
                WHEN task_results.status IN ('SUCCEEDED', 'FAILED') THEN task_results.status
                ELSE 'PROCESSING'
            END,
            updated_at = NOW()
        WHERE task_results.payload_hash = EXCLUDED.payload_hash
        RETURNING status
        """,
        task_id,
        payload_hash,
        batch_id,
    )
    if row is None:
        raise ValueError(f"Task id {task_id} was reused with a different payload")
    return str(row["status"])


async def mark_retrying(pool: asyncpg.Pool, task_id: UUID, error: str) -> None:
    await pool.execute(
        """
        UPDATE task_results
        SET status = CASE WHEN status IN ('SUCCEEDED', 'FAILED') THEN status ELSE 'RETRYING' END,
            error = $2,
            updated_at = NOW()
        WHERE task_id = $1
        """,
        task_id,
        error[:2000],
    )


async def complete_task(
    pool: asyncpg.Pool,
    task_id: UUID,
    payload_hash: str,
    batch_id: str | None,
    result: dict[str, Any],
    model_version: str,
    run_id: str,
    processing_ms: float,
) -> None:
    await pool.execute(
        """
        INSERT INTO task_results(
            task_id, payload_hash, batch_id, status, result_json, model_version,
            run_id, processing_ms, created_at, updated_at, completed_at
        )
        VALUES($1, $2, $3, 'SUCCEEDED', $4::jsonb, $5, $6, $7, NOW(), NOW(), NOW())
        ON CONFLICT(task_id) DO UPDATE SET
            status = 'SUCCEEDED',
            result_json = EXCLUDED.result_json,
            model_version = EXCLUDED.model_version,
            run_id = EXCLUDED.run_id,
            processing_ms = EXCLUDED.processing_ms,
            error = NULL,
            updated_at = NOW(),
            completed_at = NOW()
        WHERE task_results.payload_hash = EXCLUDED.payload_hash
          AND task_results.status <> 'SUCCEEDED'
        """,
        task_id,
        payload_hash,
        batch_id,
        json.dumps(result, separators=(",", ":")),
        model_version,
        run_id,
        processing_ms,
    )


async def fail_task(pool: asyncpg.Pool, task_id: UUID, error: str) -> None:
    await pool.execute(
        """
        UPDATE task_results
        SET status = CASE WHEN status = 'SUCCEEDED' THEN status ELSE 'FAILED' END,
            error = CASE WHEN status = 'SUCCEEDED' THEN error ELSE $2 END,
            updated_at = NOW(),
            completed_at = CASE WHEN status = 'SUCCEEDED' THEN completed_at ELSE NOW() END
        WHERE task_id = $1
        """,
        task_id,
        error[:2000],
    )


async def batch_metrics(pool: asyncpg.Pool, batch_id: str | None) -> dict[str, Any]:
    if batch_id:
        where = "WHERE batch_id = $1"
        args = (batch_id,)
    else:
        where = ""
        args = ()
    row = await pool.fetchrow(
        f"""
        SELECT COUNT(*) AS total,
               COUNT(*) FILTER (WHERE status = 'SUCCEEDED') AS succeeded,
               COUNT(*) FILTER (WHERE status = 'FAILED') AS failed,
               COUNT(*) FILTER (WHERE status IN ('PENDING', 'PROCESSING', 'RETRYING')) AS unfinished,
               AVG(processing_ms) AS avg_processing_ms,
               percentile_cont(0.95) WITHIN GROUP (ORDER BY processing_ms)
                   FILTER (WHERE processing_ms IS NOT NULL) AS p95_processing_ms
        FROM task_results {where}
        """,
        *args,
    )
    result = dict(row)
    for key in ("avg_processing_ms", "p95_processing_ms"):
        if result[key] is not None:
            result[key] = float(result[key])
    return result
