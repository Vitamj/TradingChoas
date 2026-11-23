from prometheus_client import Counter, Gauge, Histogram

API_REQUESTS = Counter("pipeline_api_requests_total", "API requests", ("route", "method", "status"))
API_DURATION = Histogram(
    "pipeline_api_request_duration_seconds",
    "API response duration",
    ("route", "method"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
SOURCE_PUBLISHED = Counter("pipeline_source_published_total", "Published source tasks")
SOURCE_REJECTED = Counter("pipeline_source_rejected_total", "Rejected source publishes")
SOURCE_WAIT = Counter("pipeline_source_wait_seconds_total", "Backpressure wait seconds")
WORKER_TASKS = Counter("pipeline_worker_tasks_total", "Worker results", ("result", "worker"))
WORKER_REDELIVERIES = Counter(
    "pipeline_worker_redeliveries_total", "Redelivered messages", ("worker",)
)
WORKER_ACTIVE = Gauge("pipeline_worker_active_tasks", "Tasks currently processing", ("worker",))
INFERENCE_DURATION = Histogram(
    "pipeline_inference_duration_seconds",
    "Model inference duration",
    ("worker",),
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5),
)
