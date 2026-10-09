import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from open_webui.routers import openai
from open_webui.utils.payload import apply_params_to_form_data
from open_webui.utils.reasoning_capabilities import (
    parse_abliteration_levels,
    supplement_reasoning_capabilities,
)
from open_webui.utils.reasoning_levels import (
    advertised_levels,
    apply_client_reasoning_effort,
    client_has_reasoning_effort,
    default_config,
)


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "thinking": {
                "levels": [
                    "none",
                    "minimal",
                    "low",
                    "medium",
                    "high",
                    "xhigh",
                    "max",
                    "ultra",
                    "future_level",
                ]
            }
        },
        {
            "supported_reasoning_levels": [
                {"effort": value}
                for value in ["none", "low", "high", "xhigh", "max", "future_level"]
            ]
        },
        {
            "reasoning": {
                "supported_efforts": ["max", "xhigh", "high", "medium", "low", "none"]
            }
        },
    ],
)
def test_every_advertised_value_survives_even_with_saved_subset(metadata):
    config = default_config("gpt-6-astra")  # Deliberately only low/medium/high.
    for value in advertised_levels(metadata):
        body = {
            "reasoning": {"effort": "old", "summary": "auto"},
            "tools": [{"type": "web_search"}],
        }
        result = apply_client_reasoning_effort(
            body, config, "gpt-6-astra", metadata, "", "  " + value.upper() + " "
        )
        assert result["reasoning_effort"] == value
        assert result["reasoning"] == {"summary": "auto"}
        assert result["tools"] == [{"type": "web_search"}]


@pytest.mark.parametrize(
    "model_id,levels,expected",
    [
        ("gpt-6-astra", ["low", "high", "xhigh", "max"], "xhigh"),
        (
            "anthropic/claude-sonnet-5",
            ["max", "xhigh", "high", "medium", "low"],
            "xhigh",
        ),
        ("anthropic/claude-sonnet-4", ["low", "medium", "high"], "high"),
        ("gpt-5", ["high", "low"], "high"),
        ("z-ai/glm-5.3", ["low", "high", "max"], "max"),
        ("ablit-glm53", ["low", "high", "max"], "max"),
    ],
)
@pytest.mark.parametrize("requested", ["typo", "ultra", "", "auto"])
def test_unknown_effort_uses_family_fallback_only_when_advertised(
    model_id, levels, expected, requested
):
    model = {"thinking": {"levels": levels}}
    result = apply_client_reasoning_effort(
        {}, None, model_id, model, "https://openrouter.ai/api/v1", requested
    )
    assert result == {"reasoning": {"effort": expected}}


def test_glm_manual_default_does_not_override_explicit_client_value():
    model = {
        "name": "abliterated-model-large-v2",
        "thinking": {"levels": ["low", "high", "max"]},
    }
    config = {
        **default_config("gpt-6-astra"),
        "field": "reasoning.effort",
        "default_id": "max",
        "options": [
            {"id": v, "value": v, "labels": {}, "bindings": {}}
            for v in ["low", "high", "max"]
        ],
    }
    for value in ["low", "high", "max"]:
        result = apply_client_reasoning_effort(
            {"reasoning": {"effort": "max", "summary": "auto"}},
            config,
            "ablit-glm53",
            model,
            "",
            value,
        )
        assert result == {"reasoning": {"effort": value, "summary": "auto"}}


@pytest.mark.parametrize(
    "model", [{}, {"thinking": {"levels": []}}, {"thinking": {"levels": "high"}}]
)
def test_nonreasoning_model_drops_client_default_without_inventing_effort(model):
    assert (
        apply_client_reasoning_effort(
            {"reasoning_effort": "high"}, None, "openai/gpt-4o", model, "", "high"
        )
        == {}
    )


def test_missing_capabilities_and_missing_fallback_fail_closed():
    with pytest.raises(ValueError, match="unavailable"):
        apply_client_reasoning_effort({}, None, "gpt-6-astra", {}, "", "low")
    with pytest.raises(ValueError, match="upstream supports"):
        apply_client_reasoning_effort(
            {}, None, "claude-sonnet-5", {"thinking": {"levels": ["low"]}}, "", "typo"
        )
    with pytest.raises(ValueError, match="upstream supports"):
        apply_client_reasoning_effort(
            {}, None, "glm-5.3", {"thinking": {"levels": ["low", "high"]}}, "", "typo"
        )


def test_disabled_config_and_presets_with_bindings():
    model = {"thinking": {"levels": ["low", "high"]}}
    config = {**default_config("gpt-5"), "enabled": False}
    assert (
        apply_client_reasoning_effort(
            {"reasoning_effort": "high"}, config, "gpt-5", model, "", "high"
        )
        == {}
    )
    config = {
        "enabled": True,
        "field": "output_config.effort",
        "default_id": "deep",
        "options": [
            {"id": "deep", "value": "high", "bindings": {"reasoning.exclude": True}},
            {"id": "quick", "value": "low", "bindings": {}},
        ],
    }
    body = {
        "reasoning": {"summary": "auto", "exclude": True},
        "output_config": {"effort": "high"},
    }
    result = apply_client_reasoning_effort(
        body, config, "claude-model", model, "", "quick"
    )
    assert result == {
        "reasoning": {"summary": "auto"},
        "output_config": {"effort": "low"},
    }


def test_official_document_parser_reads_all_modes_and_fails_closed():
    doc = "`abliterated-model-large-v2` runs in four reasoning modes — **low**, **high**, **max**, and **future** — so it maps aliases."
    assert parse_abliteration_levels(doc, "abliterated-model-large-v2") == [
        "low",
        "high",
        "max",
        "future",
    ]
    assert (
        parse_abliteration_levels(
            doc.replace("four", "three"), "abliterated-model-large-v2"
        )
        is None
    )
    assert (
        parse_abliteration_levels(doc + "\n" + doc, "abliterated-model-large-v2")
        is None
    )
    assert (
        parse_abliteration_levels("minimal low high max", "abliterated-model-large-v2")
        is None
    )


@pytest.mark.asyncio
async def test_supplement_uses_upstream_identity_and_preserves_explicit_empty(
    monkeypatch,
):
    from open_webui.utils import reasoning_capabilities as capabilities

    fetch = AsyncMock(
        return_value={"abliterated-model-large-v2": ["low", "high", "max", "future"]}
    )
    monkeypatch.setattr(capabilities, "fetch_abliteration_levels", fetch)
    models = [
        {"id": "alias", "name": "abliterated-model-large-v2"},
        {
            "id": "other",
            "name": "abliterated-model-large-v2",
            "thinking": {"levels": []},
        },
    ]
    await supplement_reasoning_capabilities(models)
    assert advertised_levels(models[0]) == ["low", "high", "max", "future"]
    assert advertised_levels(models[1]) == []
    assert models[0]["reasoning_capabilities_source"].startswith(
        "https://docs.abliteration.ai/"
    )
    fetch.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        None,
        {"error": "offline"},
        {
            "data": [
                {
                    "id": "allowed",
                    "reasoning": {"supported_efforts": ["xhigh", "high", "low"]},
                },
                {"id": "hidden", "thinking": {"levels": ["max"]}},
            ]
        },
    ],
)
async def test_manual_model_allowlist_retains_provider_metadata_and_offline_entries(
    monkeypatch, response
):
    monkeypatch.setattr(openai, "get_models_request", AsyncMock(return_value=response))
    result = await openai.get_configured_models_request(
        None, "https://provider/v1", "test", None, {"model_ids": ["allowed", "missing"]}
    )
    assert [m["id"] for m in result["data"]] == ["allowed", "missing"]
    if response and "data" in response:
        assert advertised_levels(result["data"][0]) == ["xhigh", "high", "low"]
    assert advertised_levels(result["data"][1]) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "api_type,url,model_id,levels,requested,expected",
    [
        (
            "responses",
            "https://provider/v1",
            "gpt-6-astra",
            ["low", "high", "xhigh", "max"],
            "low",
            "low",
        ),
        (
            "responses",
            "https://provider/v1",
            "gpt-6-astra",
            ["low", "high", "xhigh", "max"],
            "typo",
            "xhigh",
        ),
        (
            "",
            "https://openrouter.ai/api/v1",
            "anthropic/claude-sonnet-5",
            ["low", "high", "xhigh", "max"],
            "max",
            "max",
        ),
        (
            "",
            "https://openrouter.ai/api/v1",
            "anthropic/claude-sonnet-5",
            ["low", "high"],
            "typo",
            "high",
        ),
        (
            "responses",
            "https://provider/v1",
            "ablit-glm53",
            ["low", "high", "max"],
            "low",
            "low",
        ),
        (
            "responses",
            "https://provider/v1",
            "ablit-glm53",
            ["low", "high", "max"],
            "typo",
            "max",
        ),
        ("", "https://openrouter.ai/api/v1", "openai/gpt-4o", [], "xhigh", None),
    ],
)
async def test_router_serializes_relay_params_after_defaults(
    monkeypatch,
    api_type,
    url,
    model_id,
    levels,
    requested,
    expected,
    *,
    client_supplied=True,
):
    glm = "glm" in model_id
    config = (
        {
            "enabled": True,
            "field": "reasoning.effort",
            "default_id": "max",
            "options": [{"id": v, "value": v} for v in levels],
        }
        if glm
        else None
    )
    meta = {"reasoning_effort_config": config} if glm else {}
    params = (
        {"custom_params": {"reasoning": {"summary": "auto", "effort": "max"}}}
        if levels
        else {}
    )
    info = SimpleNamespace(
        id="alias",
        base_model_id=model_id,
        params=SimpleNamespace(model_dump=lambda: deepcopy(params)),
    )
    info.model_dump = lambda: {"id": "alias", "base_model_id": model_id, "meta": meta}
    monkeypatch.setattr(
        openai.Models,
        "get_model_by_id",
        AsyncMock(side_effect=lambda id: info if id == "alias" else None),
    )
    monkeypatch.setattr(openai.Config, "get", AsyncMock(return_value=True))
    monkeypatch.setattr(openai, "check_model_access", AsyncMock())
    monkeypatch.setattr(
        openai,
        "get_openai_connection",
        AsyncMock(return_value=(url, "test-key", {"api_type": api_type})),
    )
    monkeypatch.setattr(
        openai, "get_headers_and_cookies", AsyncMock(return_value=({}, {}))
    )
    monkeypatch.setattr(
        openai,
        "inject_openai_files_into_messages",
        AsyncMock(side_effect=lambda request, payload, *a, **kw: payload),
    )
    monkeypatch.setattr(openai, "cleanup_response", AsyncMock())
    model = {"id": model_id, "urlIdx": 0, "thinking": {"levels": levels}}
    refresh = AsyncMock(return_value={"data": [model]})
    monkeypatch.setattr(openai, "get_all_models", refresh)
    response = SimpleNamespace(
        status=200,
        headers={"Content-Type": "application/json"},
        json=AsyncMock(return_value={"choices": []}),
    )
    session = SimpleNamespace(request=AsyncMock(return_value=response))
    monkeypatch.setattr(openai, "get_session", AsyncMock(return_value=session))
    request = SimpleNamespace(
        state=SimpleNamespace(
            bypass_system_prompt=True, client_reasoning_effort_supplied=client_supplied
        ),
        app=SimpleNamespace(state=SimpleNamespace(OPENAI_MODELS={model_id: model})),
    )
    body = apply_params_to_form_data(
        {
            "model": "alias",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "params": {"reasoning_effort": requested},
        },
        {"owned_by": "openai"},
    )
    await openai.generate_chat_completion(request, body, SimpleNamespace(role="admin"))
    sent = json.loads(session.request.call_args.kwargs["data"])
    assert sent["model"] == model_id
    assert "reasoning_effort" not in sent
    assert "reasoning_effort_level" not in sent
    if expected is None:
        assert "reasoning" not in sent
    else:
        assert sent["reasoning"] == {"effort": expected, "summary": "auto"}
    if client_supplied:
        refresh.assert_awaited_once()
    else:
        refresh.assert_not_awaited()


@pytest.mark.parametrize(
    "body,expected",
    [
        ({}, False),
        ({"params": {}}, False),
        ({"params": {"reasoning_effort": None}}, False),
        ({"reasoning_effort": "low"}, True),
        ({"params": {"reasoning_effort": "low"}}, True),
        ({"params": {"reasoning_effort": ""}}, True),
        ({"params": {"custom_params": {"reasoning_effort": "max"}}}, True),
        ({"params": {"reasoning": {"effort": "high"}}}, False),
    ],
)
def test_explicit_client_input_is_captured_before_model_defaults(body, expected):
    assert client_has_reasoning_effort(body) is expected


@pytest.mark.asyncio
async def test_server_default_is_not_mistaken_for_explicit_client_effort(monkeypatch):
    # medium is intentionally absent from the mock catalog: a model-owned
    # default must not be silently remapped or require an extra discovery call.
    await test_router_serializes_relay_params_after_defaults(
        monkeypatch,
        "responses",
        "https://provider/v1",
        "gpt-6-astra",
        ["low", "high"],
        "medium",
        "medium",
        client_supplied=False,
    )


@pytest.mark.parametrize(
    "model",
    [
        {"supported_parameters": ["tools", "temperature"]},
        {"supported_parameters": []},
        {"info": {"meta": {"capabilities": {"reasoning": False}}}},
    ],
)
def test_explicit_nonreasoning_capability_wins_over_family_name(model):
    assert (
        apply_client_reasoning_effort(
            {"reasoning_effort": "xhigh"}, None, "claude-3-haiku", model, "", "xhigh"
        )
        == {}
    )
