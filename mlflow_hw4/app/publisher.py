import json
from typing import Any

import aio_pika
from pamqp.commands import Basic
from aiormq.exceptions import DeliveryError

from app.config import RABBITMQ_URL, TASK_QUEUE, queue_arguments


class PublishRejected(Exception):
    pass


class TaskPublisher:
    def __init__(self, url: str = RABBITMQ_URL) -> None:
        self.url = url
        self.connection: aio_pika.RobustConnection | None = None
        self.channel: aio_pika.abc.AbstractChannel | None = None

    async def connect(self) -> None:
        self.connection = await aio_pika.connect_robust(self.url)
        self.channel = await self.connection.channel(publisher_confirms=True, on_return_raises=True)
        await self.channel.declare_queue(
            TASK_QUEUE,
            durable=True,
            arguments=queue_arguments(),
        )

    async def publish(self, envelope: dict[str, Any]) -> None:
        if self.channel is None:
            raise RuntimeError("Publisher is not connected")
        message = aio_pika.Message(
            body=json.dumps(envelope, separators=(",", ":")).encode("utf-8"),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            message_id=str(envelope["task_id"]),
            headers={
                "batch_id": envelope.get("batch_id") or "",
                "payload_hash": envelope["payload_hash"],
            },
        )
        try:
            confirmation = await self.channel.default_exchange.publish(
                message,
                routing_key=TASK_QUEUE,
                mandatory=True,
            )
        except DeliveryError as exc:
            raise PublishRejected("RabbitMQ rejected the publish") from exc

        if isinstance(confirmation, Basic.Nack):
            raise PublishRejected("RabbitMQ rejected the publish because the queue is full")
        if not isinstance(confirmation, Basic.Ack):
            raise PublishRejected("RabbitMQ did not positively confirm the publish")

    async def close(self) -> None:
        if self.connection is not None and not self.connection.is_closed:
            await self.connection.close()
            self.connection = None
            self.channel = None

    async def __aenter__(self) -> "TaskPublisher":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()
