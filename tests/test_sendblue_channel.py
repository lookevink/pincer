"""Sendblue's public HTTP boundary, real agent loop, and provider acceptance contract."""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer
from pydantic import SecretStr

from pincer.channels.base import ChannelType
from pincer.channels.middleware import IdentityMiddleware
from pincer.channels.router import ChannelRouter
from pincer.channels.sendblue import SendblueChannel
from pincer.core.agent import Agent

LINE = "+15555550100"
SENDER = "+15555550101"
SECRET = "test-webhook-secret"


def event(**changes):
    return {
        "message_handle": "fixture-inbound-1",
        "is_outbound": False,
        "status": "RECEIVED",
        "from_number": SENDER,
        "to_number": LINE,
        "content": "Hello!",
        "group_id": "",
        **changes,
    }


@pytest.fixture
async def channel(settings):
    settings.sendblue_api_key = SecretStr("test-key")
    settings.sendblue_api_secret = SecretStr("test-secret")
    settings.sendblue_signing_secret = SecretStr(SECRET)
    settings.sendblue_from_number = LINE
    settings.sendblue_allow_from = [SENDER]
    # An ephemeral listener is only a test override (production settings require >0).
    settings.sendblue_webhook_port = 0
    instance = SendblueChannel(settings)
    await instance.start(AsyncMock(return_value=""))
    yield instance
    await instance.stop()
    from pincer.db.engine import dispose_engines

    await dispose_engines()


@pytest.fixture
async def client(channel):
    port = channel._runner.addresses[0][1]
    async with ClientSession(base_url=f"http://127.0.0.1:{port}") as client:
        yield client


async def post(client, payload, secret=SECRET):
    return await client.post("/webhooks/sendblue", json=payload, headers={"sb-signing-secret": secret})


async def test_http_agent_reply_persisted_and_proactive_send(
    channel, client, settings, mock_llm, session_manager, cost_tracker, tool_registry, identity_resolver
):
    """Only the LLM and remote provider are fixtures; channel, agent, identity and SQLite are real."""
    agent = Agent(settings, mock_llm, session_manager, cost_tracker, tool_registry)
    identity = IdentityMiddleware(identity_resolver)
    canonical_ids = []

    async def handle(message):
        message = await identity(message)
        canonical_ids.append(message.pincer_user_id)
        response = await agent.handle_message(message.pincer_user_id, message.channel, message.text)
        return response.text

    received = []

    async def provider(request):
        assert request.headers["sb-api-key-id"] == "test-key"
        assert request.headers["sb-api-secret-key"] == "test-secret"
        received.append(await request.json())
        return web.json_response({"message_handle": "fixture-outbound-1", "status": "QUEUED", "error_code": 0})

    provider_app = web.Application()
    provider_app.router.add_post("/api/send-message", provider)
    async with TestServer(provider_app) as server:
        await channel._client.aclose()
        channel._client = httpx.AsyncClient(
            base_url=str(server.make_url("/")),
            headers={"sb-api-key-id": "test-key", "sb-api-secret-key": "test-secret"},
        )
        channel._handler = handle
        assert (await post(client, event())).status == 200
        await asyncio.wait_for(channel._queue.join(), 10)
        assert received == [
            {
                "number": SENDER,
                "from_number": LINE,
                "content": (
                    "Hello! I'm Pincer.\n\nBy the way — what's your name, "
                    "and what will you mainly use me for? Feel free to also mention your preferred language."
                ),
            }
        ]
        session = await session_manager.get_or_create(canonical_ids[0], "sendblue")
        assert any(m.content == "Hello!" for m in session.messages)
        assert any(m.content == received[0]["content"] for m in session.messages)
        assert (await post(client, event())).status == 200
        await channel._queue.join()
        assert len(received) == 1
        router = ChannelRouter(identity_resolver)
        router.register(ChannelType.SENDBLUE, channel)
        assert await router.send(ChannelType.SENDBLUE, SENDER, "Reminder")
        assert received[-1]["content"] == "Reminder"


@pytest.mark.parametrize(
    "change,status",
    [
        ({"is_outbound": True}, 200),
        ({"is_outbound": "false"}, 200),
        ({"status": "DELIVERED"}, 200),
        ({"group_id": "group-1"}, 200),
        ({"to_number": "+15555550999"}, 200),
        ({"from_number": "+15555550999"}, 200),
        ({"message_handle": ""}, 400),
        ({"content": {"bad": True}}, 400),
        ({"media_url": []}, 400),
    ],
)
async def test_untrusted_and_unsupported_events_never_reach_agent(channel, client, change, status):
    assert (await post(client, event(**change))).status == status
    await channel._queue.join()
    channel._handler.assert_not_called()


async def test_authentication_and_bounded_payload(channel, client):
    assert (await post(client, event(), secret="wrong")).status == 401
    assert (await post(client, event(), secret="\u00e9")).status == 401
    assert (await post(client, [])).status == 400
    response = await client.post("/webhooks/sendblue", data="{", headers={"sb-signing-secret": SECRET})
    assert response.status == 400
    assert (await post(client, event(content="x" * 70000))).status == 413
    channel._handler.assert_not_called()


async def test_queue_pressure_does_not_poison_retry(channel, client):
    channel._worker.cancel()
    await asyncio.gather(channel._worker, return_exceptions=True)
    channel._worker = None
    for n in range(128):
        assert (await post(client, event(message_handle=f"m-{n}"))).status == 200
    assert (await post(client, event(message_handle="retry"))).status == 503
    channel._queue.get_nowait()
    channel._queue.task_done()
    assert (await post(client, event(message_handle="retry"))).status == 200


async def test_media_is_not_downloaded(channel, client):
    assert (await post(client, event(content=None, media_url="http://169.254.169.254/secret"))).status == 200
    await channel._queue.join()
    text = channel._handler.call_args.args[0].text
    assert "supports text only" in text
    assert "169.254" not in text


@pytest.mark.parametrize(
    "response",
    [
        {"status": "ERROR", "message_handle": "m"},
        {"status": "QUEUED", "message_handle": "m", "error_code": 400},
        {"status": "QUEUED"},
        [],
    ],
)
async def test_provider_rejection_is_not_success(channel, response):
    calls = []

    def provider(request):
        calls.append(request)
        return httpx.Response(200, json=response)

    await channel._client.aclose()
    channel._client = httpx.AsyncClient(transport=httpx.MockTransport(provider), base_url="https://api.sendblue.com")
    with pytest.raises(RuntimeError, match="did not confirm"):
        await channel.send(SENDER, "hello")
    assert len(calls) == 1


async def test_timeout_no_retry_and_chunking(channel):
    calls = []

    def provider(request):
        calls.append(json.loads(request.content))
        if len(calls) == 2:
            raise httpx.ReadTimeout("ambiguous acceptance")
        return httpx.Response(200, json={"status": "QUEUED", "message_handle": "m"})

    await channel._client.aclose()
    channel._client = httpx.AsyncClient(transport=httpx.MockTransport(provider), base_url="https://api.sendblue.com")
    with pytest.raises(RuntimeError, match="unconfirmed"):
        await channel.send(SENDER, "x" * 4500)
    assert len(calls) == 2
    assert len(calls[0]["content"]) == 2000
    with pytest.raises(ValueError, match="allowlist"):
        await channel.send("+15555550999", "blocked")
    assert len(calls) == 2


async def test_missing_config_fails_closed(settings):
    with pytest.raises(ValueError, match="requires"):
        await SendblueChannel(settings).start(AsyncMock())


async def test_failed_handler_is_not_replayed(channel, client):
    channel._handler = AsyncMock(side_effect=RuntimeError("fixture failure"))
    assert (await post(client, event())).status == 200
    await channel._queue.join()
    assert (await post(client, event())).status == 200
    await channel._queue.join()
    channel._handler.assert_awaited_once()
