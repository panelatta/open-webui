from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from open_webui.utils import middleware


@pytest.mark.anyio
@pytest.mark.parametrize('caller_tools', [None, [], [{'type': 'web_search'}]])
@pytest.mark.parametrize('session_id', [None, 'browser-session'])
async def test_configured_search_coexists_with_builtins_but_explicit_tools_opt_out(
    monkeypatch, caller_tools, session_id
):
    model = {
        'id': 'glm-preset',
        'owned_by': 'openai',
        'info': {'meta': {'builtinTools': {'knowledge': False}}},
    }
    request = SimpleNamespace(
        state=SimpleNamespace(),
        app=SimpleNamespace(state=SimpleNamespace(MODELS={model['id']: model})),
    )
    user = SimpleNamespace(id='test-user', role='admin')
    metadata = {'chat_id': '', 'session_id': session_id, 'params': {'function_calling': 'native'}}
    form_data = {
        'model': model['id'],
        'messages': [{'role': 'user', 'content': 'Create tasks and search.'}],
        'params': {'custom_params': {'tools': [{'type': 'web_search'}]}},
    }
    if caller_tools is not None:
        form_data['tools'] = deepcopy(caller_tools)

    async def config_get(key, default=None):
        return default

    async def identity(value, *args, **kwargs):
        return value

    async def pipeline(request, body, user, models):
        return body

    builtin = AsyncMock(return_value={'create_tasks': {'spec': {'name': 'create_tasks'}}})
    monkeypatch.setattr(middleware.Config, 'get', config_get)
    monkeypatch.setattr(middleware, 'ENABLE_PLUGINS', False)
    monkeypatch.setattr(middleware, 'convert_url_images_to_base64', identity)
    monkeypatch.setattr(middleware, 'get_event_emitter', AsyncMock(return_value=None))
    monkeypatch.setattr(middleware, 'get_event_call', AsyncMock(return_value=None))
    monkeypatch.setattr(middleware, 'get_system_oauth_token', AsyncMock(return_value=None))
    monkeypatch.setattr(middleware, 'get_task_model_id', lambda *args: model['id'])
    monkeypatch.setattr(middleware, 'process_pipeline_inlet_filter', pipeline)
    monkeypatch.setattr(middleware, 'add_file_context', identity)
    monkeypatch.setattr(middleware, 'get_builtin_tools', builtin)

    body, _, _ = await middleware.process_chat_payload(request, form_data, user, metadata, model)

    if caller_tools is not None:
        assert body['tools'] == caller_tools
        builtin.assert_not_awaited()
    elif session_id:
        assert body['tools'] == [
            {'type': 'function', 'function': {'name': 'create_tasks'}},
            {'type': 'web_search'},
        ]
        assert 'create_tasks' in body['metadata']['tools']
        builtin.assert_awaited_once()
    else:
        assert body['tools'] == [{'type': 'web_search'}]
        builtin.assert_not_awaited()
