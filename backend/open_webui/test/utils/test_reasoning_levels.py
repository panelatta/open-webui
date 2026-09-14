import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from open_webui.models.models import ModelMeta
from open_webui.routers import openai
from open_webui.utils.payload import apply_params_to_form_data
from open_webui.utils.reasoning_levels import ReasoningConfig, apply_reasoning_level, default_config, model_chain


def test_alias_inheritance_disable_and_cycle():
    custom = default_config('gpt-5.5')
    infos = {
        'alias': {'base_model_id': 'inner'},
        'inner': {'base_model_id': 'gpt-5.5', 'meta': {'reasoning_effort_config': custom}},
    }
    assert model_chain('alias', infos) == ('gpt-5.5', custom)
    infos['alias']['meta'] = {'reasoning_effort_config': {**custom, 'enabled': False}}
    assert model_chain('alias', infos)[1]['enabled'] is False
    infos['inner']['base_model_id'] = 'alias'
    with pytest.raises(ValueError, match='Cyclic'):
        model_chain('alias', infos)


def test_defaults_unknown_models_and_gateway_protocol():
    assert default_config('anthropic/claude-sonnet-5', 'https://openrouter.ai/api/v1')['field'] == 'reasoning.effort'
    assert default_config('custom-alias') is None
    assert default_config('gpt-4o') is None
    assert default_config('gpt-5-chat') is None
    assert default_config('gpt-6-astra')['options'][0]['id'] == 'low'


def test_preserve_summary_tools_defaults_and_remove_conflicting_alias():
    config = default_config('gpt-5.5')
    original = {'reasoning': {'effort': 'high', 'summary': 'auto'}, 'tools': [{'type': 'web_search'}]}
    assert apply_reasoning_level(deepcopy(original), config, '') == original
    result = openai.convert_to_responses_payload(apply_reasoning_level(deepcopy(original), config, 'low'))
    assert result['reasoning'] == {'effort': 'low', 'summary': 'auto'}
    assert result['tools'] == original['tools']
    with pytest.raises(ValueError, match='no longer exists'):
        apply_reasoning_level({}, config, 'deleted')


def test_budget_binding_validation_and_switch_cleanup():
    config = {
        'field': 'thinking.budget_tokens',
        'options': [
            {'id': 'on', 'value': 1024, 'bindings': {'thinking.type': 'enabled'}},
            {'id': 'off', 'value': None, 'bindings': {'thinking.type': 'disabled'}},
        ],
    }
    config = ReasoningConfig.model_validate(config).model_dump()
    payload = apply_reasoning_level({'max_tokens': 4096}, config, 'on')
    assert payload['thinking'] == {'type': 'enabled', 'budget_tokens': 1024}
    assert apply_reasoning_level(payload, config, 'off')['thinking'] == {'type': 'disabled'}
    with pytest.raises(ValueError, match='smaller'):
        apply_reasoning_level({'max_tokens': 1024}, config, 'on')
    with pytest.raises(ValueError):
        ModelMeta(reasoning_effort_config={**config, 'field': 'model'})
    with pytest.raises(ValueError):
        ReasoningConfig.model_validate({**config, 'options': config['options'] * 2})


def test_ui_selection_map_never_leaks_upstream():
    body = apply_params_to_form_data(
        {'params': {'reasoning_effort_levels': {'alias': 'high'}, 'temperature': 0.5}}, {'owned_by': 'openai'}
    )
    assert body == {'temperature': 0.5}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'url,api_type,field',
    [
        ('https://api.openai.com/v1', 'responses', 'reasoning'),
        ('https://api.openai.com/v1', '', 'reasoning_effort'),
        ('https://openrouter.ai/api/v1', '', 'reasoning'),
    ],
)
async def test_router_serializes_real_transport_after_model_defaults(monkeypatch, url, api_type, field):
    base = 'anthropic/claude-sonnet-5' if 'openrouter' in url else 'gpt-6-astra'
    info = SimpleNamespace(
        id='alias',
        base_model_id=base,
        params=SimpleNamespace(
            model_dump=lambda: {'custom_params': {'reasoning': {'summary': 'auto', 'effort': 'high'}}}
        ),
    )
    info.model_dump = lambda: {'id': 'alias', 'base_model_id': base, 'meta': {}}
    monkeypatch.setattr(
        openai.Models, 'get_model_by_id', AsyncMock(side_effect=lambda id: info if id == 'alias' else None)
    )
    monkeypatch.setattr(openai.Config, 'get', AsyncMock(return_value=True))
    monkeypatch.setattr(openai, 'check_model_access', AsyncMock())
    monkeypatch.setattr(
        openai, 'get_openai_connection', AsyncMock(return_value=(url, 'test-key', {'api_type': api_type}))
    )
    monkeypatch.setattr(openai, 'get_headers_and_cookies', AsyncMock(return_value=({}, {})))
    monkeypatch.setattr(
        openai, 'inject_openai_files_into_messages', AsyncMock(side_effect=lambda request, payload, *a, **kw: payload)
    )
    monkeypatch.setattr(openai, 'cleanup_response', AsyncMock())
    response = SimpleNamespace(
        status=200, headers={'Content-Type': 'application/json'}, json=AsyncMock(return_value={'choices': []})
    )
    session = SimpleNamespace(request=AsyncMock(return_value=response))
    monkeypatch.setattr(openai, 'get_session', AsyncMock(return_value=session))
    request = SimpleNamespace(
        state=SimpleNamespace(bypass_system_prompt=True),
        app=SimpleNamespace(state=SimpleNamespace(OPENAI_MODELS={base: {'urlIdx': 0}})),
    )
    await openai.generate_chat_completion(
        request,
        {
            'model': 'alias',
            'messages': [{'role': 'user', 'content': 'hi'}],
            'stream': False,
            'reasoning_effort_level': 'low',
        },
        SimpleNamespace(role='admin'),
    )
    sent = json.loads(session.request.call_args.kwargs['data'])
    assert sent['model'] == base
    assert 'reasoning_effort_level' not in sent
    assert sent[field] == ({'effort': 'low', 'summary': 'auto'} if field == 'reasoning' else 'low')
    assert session.request.call_args.kwargs['url'].endswith(
        '/responses' if api_type == 'responses' else '/chat/completions'
    )


def test_responses_model_params_merge_effort_without_losing_summary():
    body = openai.apply_model_params_to_body_responses(
        {'reasoning_effort': 'low', 'custom_params': {'reasoning': {'summary': 'auto', 'effort': 'high'}}},
        {},
    )
    assert body['reasoning'] == {'effort': 'low', 'summary': 'auto'}
