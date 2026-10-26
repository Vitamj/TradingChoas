import asyncio

import aio_pika

from app.config import DLQ_NAME, DLX_NAME, RABBITMQ_URL, TASK_QUEUE, queue_arguments


async def declare_topology() -> None:
    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    try:
        channel = await connection.channel(publisher_confirms=True)
        exchange = await channel.declare_exchange(
            DLX_NAME,
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        await channel.declare_queue(
            TASK_QUEUE,
            durable=True,
            arguments=queue_arguments(),
        )
        dead_queue = await channel.declare_queue(DLQ_NAME, durable=True)
        await dead_queue.bind(exchange, routing_key=DLQ_NAME)
    finally:
        await connection.close()


async def main() -> None:
    last_error = None
    for attempt in range(60):
        try:
            await declare_topology()
            print(f"Topology ready: queue={TASK_QUEUE}, dlq={DLQ_NAME}")
            return
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(min(0.25 * (attempt + 1), 2.0))
    raise RuntimeError(f"Could not declare RabbitMQ topology: {last_error}")


if __name__ == "__main__":
    asyncio.run(main())
