import asyncio

import aio_pika
import pytest
from pamqp.commands import Basic
from testcontainers.core.container import DockerContainer
from aiormq.exceptions import DeliveryError

pytestmark = pytest.mark.integration


def test_real_rabbitmq_rejects_publish_when_bounded_queue_is_full():
    container = (
        DockerContainer("rabbitmq:4.3-management-alpine")
        .with_env("RABBITMQ_DEFAULT_USER", "hw4test")
        .with_env("RABBITMQ_DEFAULT_PASS", "hw4test-password")
        .with_exposed_ports(5672)
    )
    with container as rabbit:
        host = rabbit.get_container_host_ip()
        port = rabbit.get_exposed_port(5672)
        url = f"amqp://hw4test:hw4test-password@{host}:{port}/"

        async def exercise_queue():
            connection = None
            last_error = None
            for attempt in range(60):
                try:
                    connection = await aio_pika.connect_robust(url, timeout=2)
                    break
                except Exception as exc:
                    last_error = exc
                    await asyncio.sleep(min(0.25 + attempt * 0.05, 1.0))
            if connection is None:
                raise RuntimeError(f"RabbitMQ container did not become ready: {last_error}")
            try:
                channel = await connection.channel(publisher_confirms=True)
                dlx = await channel.declare_exchange(
                    "overflow.dlx",
                    aio_pika.ExchangeType.DIRECT,
                    durable=True,
                )
                dlq = await channel.declare_queue("overflow.dlq", durable=True)
                await dlq.bind(dlx, routing_key="overflow.dlq")
                queue = await channel.declare_queue(
                    "overflow.test",
                    durable=True,
                    arguments={
                        "x-queue-type": "quorum",
                        "x-max-length": 2,
                        "x-overflow": "reject-publish",
                        "x-delivery-limit": 3,
                        "x-dead-letter-exchange": "overflow.dlx",
                        "x-dead-letter-routing-key": "overflow.dlq",
                        "x-dead-letter-strategy": "at-least-once",
                    },
                )


                confirmations = []
                rejected = False

                for index in range(20):
                    try:
                        confirmation = await channel.default_exchange.publish(
                            aio_pika.Message(
                                body=f"task-{index}".encode(),
                                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                            ),
                            routing_key="overflow.test",
                            mandatory=True,
                        )
                        confirmations.append(confirmation)

                    except DeliveryError:
                        rejected = True
                        break

                declared = await queue.declare()

                assert isinstance(confirmations[0], Basic.Ack)
                assert isinstance(confirmations[1], Basic.Ack)
                assert rejected, "RabbitMQ never rejected a publish"
                assert declared.message_count >= 2
                assert declared.message_count <= 3, (
                    f"Unexpected queue overshoot: {declared.message_count}"
                )

            finally:
                await connection.close()

        asyncio.run(exercise_queue())


def test_quorum_queue_bounds_failed_deliveries_and_routes_poison_message_to_dlq():
    container = (
        DockerContainer("rabbitmq:4.3-management-alpine")
        .with_env("RABBITMQ_DEFAULT_USER", "hw4test")
        .with_env("RABBITMQ_DEFAULT_PASS", "hw4test-password")
        .with_exposed_ports(5672)
    )
    with container as rabbit:
        host = rabbit.get_container_host_ip()
        port = rabbit.get_exposed_port(5672)
        url = f"amqp://hw4test:hw4test-password@{host}:{port}/"

        async def exercise_queue():
            connection = None
            last_error = None
            for attempt in range(60):
                try:
                    connection = await aio_pika.connect_robust(url, timeout=2)
                    break
                except Exception as exc:
                    last_error = exc
                    await asyncio.sleep(min(0.25 + attempt * 0.05, 1.0))
            if connection is None:
                raise RuntimeError(f"RabbitMQ container did not become ready: {last_error}")

            try:
                channel = await connection.channel(publisher_confirms=True)
                dlx = await channel.declare_exchange(
                    "retry.dlx",
                    aio_pika.ExchangeType.DIRECT,
                    durable=True,
                )
                dlq = await channel.declare_queue(
                    "retry.dlq",
                    durable=True,
                    arguments={"x-queue-type": "quorum"},
                )
                await dlq.bind(dlx, routing_key="retry.dlq")
                queue = await channel.declare_queue(
                    "retry.test",
                    durable=True,
                    arguments={
                        "x-queue-type": "quorum",
                        "x-max-length": 10,
                        "x-overflow": "reject-publish",
                        "x-delivery-limit": 3,
                        "x-dead-letter-exchange": "retry.dlx",
                        "x-dead-letter-routing-key": "retry.dlq",
                        "x-dead-letter-strategy": "at-least-once",
                    },
                )
                confirmation = await channel.default_exchange.publish(
                    aio_pika.Message(
                        body=b"poison-message",
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    ),
                    routing_key="retry.test",
                    mandatory=True,
                )
                assert isinstance(confirmation, Basic.Ack)

                delivery_counts = []

                async def reject_poison_message(message):
                    delivery_counts.append(int(message.headers.get("x-delivery-count", 0)))
                    # Simulate the worker rejecting every failed attempt. RabbitMQ's
                    # quorum-queue delivery limit must eventually route it to the DLQ.
                    await message.reject(requeue=True)

                consumer_tag = await queue.consume(reject_poison_message, no_ack=False)
                dlq_state = await dlq.declare()
                for _ in range(100):
                    dlq_state = await dlq.declare()
                    if dlq_state.message_count:
                        break
                    await asyncio.sleep(0.1)
                await queue.cancel(consumer_tag)

                assert dlq_state.message_count == 1, "Poison message was not retained in DLQ"
                assert 1 < len(delivery_counts) <= 4, (
                    f"Failed deliveries were not bounded as expected: {delivery_counts}"
                )
                assert max(delivery_counts) <= 3, (
                    f"Delivery limit was exceeded: {delivery_counts}"
                )
            finally:
                await connection.close()

        asyncio.run(exercise_queue())
