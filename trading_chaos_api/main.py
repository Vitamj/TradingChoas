import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from time import perf_counter

import asyncpg
from fastapi import FastAPI, Response

from trading_chaos_api import __version__

DatabaseProbe = Callable[[], Awaitable[str]]
logger = logging.getLogger("uvicorn.access")


async def probe_database() -> str:
    connection = await asyncpg.connect(os.environ["DATABASE_URL"], timeout=2)
    try:
        await connection.execute("SELECT 1")
        version = connection.get_server_version()
        return f"{version.major}.{version.minor}.{version.micro}"
    finally:
        await connection.close()


def create_app(database_probe: DatabaseProbe = probe_database) -> FastAPI:
    app = FastAPI(title="Trading Chaos API", version=__version__)

    @app.middleware("http")
    async def log_request(request, call_next):
        started = perf_counter()
        response = await call_next(request)
        logger.info(
            "%s %s %s %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            (perf_counter() - started) * 1000,
        )
        return response

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/version")
    async def version() -> dict[str, str]:
        return {
            "version": __version__,
            "revision": os.getenv("APP_REVISION", "local"),
        }

    @app.get("/api/v1/health")
    async def health(response: Response):
        started = perf_counter()
        try:
            database_version = await asyncio.wait_for(database_probe(), timeout=3)
            database = {"status": "ok", "version": database_version}
        except (TimeoutError, KeyError, OSError, asyncpg.PostgresError) as error:
            logger.warning("Database health check failed: %s", error)
            database = {"status": "error", "error": type(error).__name__}
            response.status_code = 503

        return {
            "status": "ok" if database["status"] == "ok" else "degraded",
            "components": {
                "api": {"status": "ok", "version": __version__},
                "database": {
                    **database,
                    "response_ms": round((perf_counter() - started) * 1000, 2),
                },
            },
        }

    return app


app = create_app()
