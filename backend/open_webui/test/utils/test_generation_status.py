import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from open_webui.utils.generation_status import (
    GenerationStatus,
    default_effort,
    explicit_effort,
    merge_generation_status,
    start_generation_status,
)
from open_webui.utils.middleware import (
    MESSAGE_REPLAY_KEYS,
    load_messages_from_db,
    process_messages_with_output,
)
from open_webui.utils.post_chat_memory import _compact_messages


@pytest.mark.parametrize(
    "payload,effort",
    [
        ({"reasoning": {"effort": "max", "summary": "auto"}}, "max"),
        ({"reasoning_effort": "xhigh"}, "xhigh"),
        ({"output_config": {"effort": "high"}}, "high"),
        ({"thinking": {"type": "enabled", "budget_tokens": 2048}}, "budget:2048"),
        ({"reasoning": {"enabled": False}}, "none"),
        ({}, None),
    ],
)
def test_effective_request_controls(payload, effort):
    assert explicit_effort(payload) == effort


@pytest.mark.parametrize(
    "model,effort",
    [
        (
            {
                "reasoning": {
                    "supported_efforts": ["low", "high"],
                    "default_effort": "high",
                }
            },
            "high",
        ),
        ({"reasoning": {"supported_efforts": ["low"], "default_effort": "high"}}, None),
        (
            {
                "thinking": {"levels": ["medium", "high"]},
                "default_reasoning_level": "medium",
            },
            "medium",
        ),
        ({"thinking": {"levels": ["low"]}}, None),
    ],
)
def test_default_requires_provider_evidence(model, effort):
    assert default_effort(model) == effort


@pytest.mark.asyncio
async def test_response_created_resolves_actual_unspecified_default_without_changing_request():
    emitter = AsyncMock()
    payload = {"model": "gpt-6-astra", "messages": [{"role": "user", "content": "Hi"}]}
    before = deepcopy(payload)
    status = GenerationStatus(
        emitter, "m1", payload, {"thinking": {"levels": ["low", "medium", "high"]}}
    )
    await status.publish(False)
    initial = emitter.call_args.args[0]["data"]
    assert initial["reasoning_effort"] is None and initial["done"] is False
    await status.observe(
        {
            "type": "response.created",
            "response": {"model": "gpt-6-astra", "reasoning": {"effort": "medium"}},
        }
    )
    final = emitter.call_args.args[0]["data"]
    assert final["reasoning_effort"] == "medium"
    assert final["reasoning_effort_source"] == "response"
    assert final["done"] is True
    assert final["id"] == initial["id"]
    assert "medium" in final["description"]
    assert payload == before
    await status.finish()
    assert emitter.await_count == 2


@pytest.mark.asyncio
async def test_response_overrides_requested_effort_and_reports_actual_base_model():
    emitter = AsyncMock()
    status = GenerationStatus(
        emitter, "m1", {"model": "alias", "reasoning_effort": "xhigh"}, {}
    )
    await status.observe(
        {
            "type": "response.created",
            "response": {"model": "real-model", "reasoning": {"effort": "high"}},
        }
    )
    data = emitter.call_args.args[0]["data"]
    assert (
        data["base_model"] == "real-model" and data["requested_base_model"] == "alias"
    )
    assert (
        data["reasoning_effort"] == "high"
        and data["reasoning_effort_source"] == "response"
    )


@pytest.mark.asyncio
async def test_chat_completions_default_is_concrete_and_labeled_without_mutation():
    payload = {"model": "anthropic/claude-sonnet-5", "messages": []}
    emitter = AsyncMock()
    status = GenerationStatus(
        emitter,
        "m1",
        payload,
        {"reasoning": {"supported_efforts": ["high", "low"], "default_effort": "high"}},
    )
    await status.publish(False)
    await status.finish()
    data = emitter.call_args.args[0]["data"]
    assert data["reasoning_effort"] == "high"
    assert data["reasoning_effort_source"] == "provider_default"
    assert "high (provider default)" in data["description"]
    assert "reasoning" not in payload and "reasoning_effort" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,effort,source",
    [
        (
            {"id": "openai/gpt-4o", "supported_parameters": ["temperature"]},
            "none",
            "not_applicable",
        ),
        ({"id": "unknown"}, None, "unavailable"),
    ],
)
async def test_no_reasoning_and_unconfirmed_are_distinct(model, effort, source):
    emitter = AsyncMock()
    status = GenerationStatus(emitter, "m1", {"model": model["id"]}, model)
    await status.finish()
    data = emitter.call_args.args[0]["data"]
    assert data["reasoning_effort"] == effort
    assert data["reasoning_effort_source"] == source
    assert data["done"] is True
    if effort is None:
        assert "unconfirmed" in data["description"]


@pytest.mark.asyncio
async def test_stream_bytes_remain_identical_and_actual_effort_is_emitted_before_content():
    emitter = AsyncMock()
    status = GenerationStatus(emitter, "m1", {"model": "gpt-6-astra"}, {})
    lines = [
        b"event: response.created\n",
        b'data: {"type":"response.created","response":{"model":"gpt-6-astra","reasoning":{"effort":"medium"}}}\n',
        b"\n",
        b'data: {"type":"response.output_text.delta","delta":"42"}\n',
        b"\n",
    ]
    closed = []

    async def stream():
        try:
            for line in lines:
                if b"output_text.delta" in line:
                    assert (
                        emitter.call_args.args[0]["data"]["reasoning_effort"]
                        == "medium"
                    )
                yield line
        finally:
            closed.append(True)

    output = [line async for line in status.wrap(stream())]
    assert output == lines and closed == [True]
    assert emitter.await_count == 1


@pytest.mark.asyncio
async def test_stream_error_preserves_error_and_finishes_display():
    emitter = AsyncMock()
    status = GenerationStatus(emitter, "m1", {"model": "unknown"}, {})

    async def stream():
        yield b": keepalive\n"
        raise RuntimeError("upstream disconnected")

    with pytest.raises(RuntimeError, match="upstream disconnected"):
        _ = [line async for line in status.wrap(stream())]
    assert emitter.call_args.args[0]["data"]["reasoning_effort_source"] == "unavailable"


@pytest.mark.asyncio
async def test_display_failure_does_not_fail_generation():
    status = GenerationStatus(
        AsyncMock(side_effect=RuntimeError("socket gone")),
        "m1",
        {"model": "gpt-6-astra", "reasoning_effort": "high"},
        {},
    )
    await status.finish()
    assert status.done


def test_status_updates_preserve_other_statuses_and_do_not_duplicate():
    pending = {
        "action": "generation_config",
        "id": "generation-config:m1",
        "done": False,
    }
    final = {**pending, "done": True, "reasoning_effort": "high"}
    search = {"action": "web_search", "done": False}
    history = merge_generation_status([], pending)
    history = merge_generation_status(history, search)
    assert merge_generation_status(history, final) == [final, search]
    assert history == [pending, search]
    assert merge_generation_status(history, search) == [pending, search, search]


@pytest.mark.asyncio
async def test_once_per_message_not_per_tool_retry_or_background_task(monkeypatch):
    from open_webui.socket import main as socket

    emitter = AsyncMock()
    monkeypatch.setattr(socket, "get_event_emitter", AsyncMock(return_value=emitter))
    request = SimpleNamespace(
        state=SimpleNamespace(metadata={"user_id": "u", "chat_id": "c"})
    )
    metadata = {"user_id": "u", "chat_id": "c", "message_id": "m1"}
    payload = {"model": "gpt-6-astra", "reasoning_effort": "high"}
    assert await start_generation_status(request, metadata, payload, {}) is not None
    assert await start_generation_status(request, metadata, payload, {}) is None
    assert (
        await start_generation_status(
            request, {**metadata, "task": "title"}, payload, {}
        )
        is None
    )
    assert (
        await start_generation_status(
            request, {**metadata, "internal": True}, payload, {}
        )
        is None
    )
    assert await start_generation_status(request, {}, payload, {}) is None
    assert (
        await start_generation_status(
            request, {**metadata, "message_id": "regenerated"}, payload, {}
        )
        is not None
    )
    assert emitter.await_count == 2


@pytest.mark.asyncio
async def test_display_status_is_excluded_from_replay_reasoning_and_memory(monkeypatch):
    from open_webui.utils import middleware

    marker = "DISPLAY_ONLY_BASE_MODEL_REASONING"
    saved = {
        "u1": {"id": "u1", "parentId": None, "role": "user", "content": "hello"},
        "a1": {
            "id": "a1",
            "parentId": "u1",
            "role": "assistant",
            "content": "42",
            "statusHistory": [{"description": marker, "action": "generation_config"}],
        },
    }
    monkeypatch.setattr(
        middleware.Chats, "get_messages_map_by_chat_id", AsyncMock(return_value=saved)
    )
    assert "statusHistory" not in MESSAGE_REPLAY_KEYS
    replay = await load_messages_from_db("chat", "a1")
    assert marker not in json.dumps(replay)
    cleaned = process_messages_with_output(list(saved.values()))
    assert marker not in json.dumps(cleaned)
    assert marker not in _compact_messages(list(saved.values()))
    assert cleaned[-1]["content"] == "42"


@pytest.mark.asyncio
async def test_untrusted_metadata_cannot_emit_to_a_chat(monkeypatch):
    from open_webui.socket import main as socket

    get_emitter = AsyncMock()
    monkeypatch.setattr(socket, "get_event_emitter", get_emitter)
    metadata = {"user_id": "victim", "chat_id": "private", "message_id": "m"}
    for trusted in (
        None,
        {"user_id": "other", "chat_id": "private"},
        {"user_id": "victim", "chat_id": "other"},
    ):
        request = SimpleNamespace(state=SimpleNamespace(metadata=trusted))
        assert (
            await start_generation_status(
                request, metadata, {"model": "gpt-6-astra"}, {}
            )
            is None
        )
    get_emitter.assert_not_awaited()
