import math
import os

RABBITMQ_USER = os.getenv("RABBITMQ_USER", "hw4")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "hw4-rabbit-password")
RABBITMQ_URL = os.getenv(
    "RABBITMQ_URL",
    f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASSWORD}@localhost:5672/",
)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://hw4:hw4-local-password@localhost:5433/hw4")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "BrentDirectionModel")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
TASK_QUEUE = os.getenv("TASK_QUEUE", "tasks.bounded")
DLQ_NAME = f"{TASK_QUEUE}.dlq"
DLX_NAME = f"{TASK_QUEUE}.dlx"
QUEUE_MAX_LENGTH = int(os.getenv("QUEUE_MAX_LENGTH", "11"))
PREFETCH_COUNT = int(os.getenv("PREFETCH_COUNT", "1"))
MAX_DELIVERIES = max(1, int(os.getenv("MAX_DELIVERIES", "3")))
INFERENCE_DELAY_SECONDS = float(os.getenv("INFERENCE_DELAY_SECONDS", "0.25"))
GRACEFUL_TIMEOUT_SECONDS = float(os.getenv("GRACEFUL_TIMEOUT_SECONDS", "90"))
RETRY_AFTER_SECONDS = max(1, math.ceil(INFERENCE_DELAY_SECONDS))
DATA_PATH = os.getenv("DATA_PATH", "/data/brent_prices.csv")
SOURCE_TASK_COUNT = int(os.getenv("SOURCE_TASK_COUNT", "180"))
RABBITMQ_MANAGEMENT_URL = os.getenv("RABBITMQ_MANAGEMENT_URL", "http://localhost:15672")


def queue_arguments(include_type: bool = True) -> dict[str, object]:
    arguments: dict[str, object] = {
        "x-overflow": "reject-publish",
        "x-delivery-limit": MAX_DELIVERIES,
        "x-dead-letter-exchange": DLX_NAME,
        "x-dead-letter-routing-key": DLQ_NAME,
        "x-dead-letter-strategy": "at-least-once",
    }
    if QUEUE_MAX_LENGTH > 0:
        arguments["x-max-length"] = QUEUE_MAX_LENGTH
    if include_type:
        arguments["x-queue-type"] = "quorum"
    return arguments
