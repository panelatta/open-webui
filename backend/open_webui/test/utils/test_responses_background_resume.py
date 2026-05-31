import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from open_webui.routers import openai
from open_webui.utils import middleware


def _sse_response(events, route_idx=2, route_url='https://cpa.example/v1'):
    async def event_stream():
        for event in events:
            yield f'data: {json.dumps(event)}\n\n'

    headers = {'content-type': 'text/event-stream'}
    if route_idx is not None:
        headers['x-openwebui-openai-url-idx'] = str(route_idx)
    if route_url is not None:
        headers['x-openwebui-openai-base-url'] = route_url

    return StreamingResponse(event_stream(), media_type='text/event-stream', headers=headers)


def _completed_response_event(response_id='resp_1', sequence_number=2, text='done'):
    output = [
        {
            'type': 'message',
            'id': 'msg_1',
            'status': 'completed',
            'role': 'assistant',
            'content': [{'type': 'output_text', 'text': text}],
        }
    ]
    return {
        'type': 'response.completed',
        'sequence_number': sequence_number,
        'response': {
            'id': response_id,
            'output': output,
            'usage': {},
        },
    }


async def _run_streaming_handler(monkeypatch, response, form_data=None, event_caller_result=None):
    events = []
    upserts = []

    async def fake_get_system_oauth_token(request, user):
        return None

    async def fake_get_sorted_filter_ids(request, model, filter_ids):
        return []

    async def fake_process_filter_functions(**kwargs):
        return kwargs['form_data'], {}

    async def fake_get_message_by_id_and_message_id(chat_id, message_id):
        return {'content': '', 'output': []}

    async def fake_upsert_message_to_chat_by_id_and_message_id(chat_id, message_id, message):
        upserts.append(message)
        return None

    async def fake_get_chat_title_by_id(chat_id):
        return 'New Chat'

    async def fake_background_tasks_handler(ctx):
        return None

    async def fake_outlet_filter_handler(ctx):
        return None

    async def fake_get_updated_tool_function(function, extra_params):
        return function

    async def fake_process_tool_result(
        request,
        tool_function_name,
        tool_result,
        tool_type,
        direct_tool,
        metadata,
        user,
    ):
        return str(tool_result) if tool_result else '', None, None

    async def fake_terminal_event_handler(*args, **kwargs):
        return None

    async def event_emitter(event):
        events.append(event)

    async def event_caller(event):
        return event_caller_result

    monkeypatch.setattr(middleware, 'get_system_oauth_token', fake_get_system_oauth_token)
    monkeypatch.setattr(middleware, 'get_sorted_filter_ids', fake_get_sorted_filter_ids)
    monkeypatch.setattr(middleware, 'process_filter_functions', fake_process_filter_functions)
    monkeypatch.setattr(middleware, 'background_tasks_handler', fake_background_tasks_handler)
    monkeypatch.setattr(middleware, 'outlet_filter_handler', fake_outlet_filter_handler)
    monkeypatch.setattr(middleware, 'get_updated_tool_function', fake_get_updated_tool_function)
    monkeypatch.setattr(middleware, 'process_tool_result', fake_process_tool_result)
    monkeypatch.setattr(middleware, 'terminal_event_handler', fake_terminal_event_handler)
    monkeypatch.setattr(
        middleware.Chats,
        'get_message_by_id_and_message_id',
        fake_get_message_by_id_and_message_id,
    )
    monkeypatch.setattr(
        middleware.Chats,
        'upsert_message_to_chat_by_id_and_message_id',
        fake_upsert_message_to_chat_by_id_and_message_id,
    )
    monkeypatch.setattr(middleware.Chats, 'get_chat_title_by_id', fake_get_chat_title_by_id)
    monkeypatch.setattr(middleware, 'ENABLE_REALTIME_CHAT_SAVE', False)
    monkeypatch.setattr(middleware, 'CHAT_RESPONSE_STREAM_RETRY_DELAY', 0)

    request = SimpleNamespace(
        state=SimpleNamespace(),
        app=SimpleNamespace(
            state=SimpleNamespace(
                config=SimpleNamespace(
                    ENABLE_USER_WEBHOOKS=False,
                    CODE_INTERPRETER_ENGINE='pyodide',
                )
            )
        ),
    )
    metadata = (form_data or {}).get('metadata') or {
        'chat_id': 'chat_1',
        'message_id': 'message_1',
        'session_id': 'session_1',
        'params': {},
    }
    form_data = form_data or {
        'model': 'gpt-5-long',
        'stream': True,
        'messages': [{'role': 'user', 'content': 'Hello'}],
        'metadata': metadata,
    }
    form_data.setdefault('metadata', metadata)

    ctx = {
        'request': request,
        'form_data': form_data,
        'user': SimpleNamespace(id='user_1'),
        'model': {
            'id': 'gpt-5-long',
            'provider': 'openai',
            'info': {'meta': {'capabilities': {'citations': True}}},
        },
        'metadata': metadata,
        'events': [],
        'event_emitter': event_emitter,
        'event_caller': event_caller,
        'tasks': None,
    }

    await middleware.streaming_chat_response_handler(response, ctx)
    return events, upserts


def test_background_resume_requires_global_switch(monkeypatch):
    monkeypatch.setattr(openai, 'ENABLE_RESPONSES_API_BACKGROUND_RESUME', False)
    monkeypatch.setattr(openai, 'RESPONSES_API_BACKGROUND_RESUME_MODEL_ALLOWLIST', {'gpt-5-long'})
    monkeypatch.setattr(openai, 'RESPONSES_API_BACKGROUND_RESUME_BASE_URL_ALLOWLIST', set())

    assert not openai.responses_background_resume_enabled('gpt-5-long', 'https://cpa.example/v1', {})


def test_background_resume_requires_provider_allowlist(monkeypatch):
    monkeypatch.setattr(openai, 'ENABLE_RESPONSES_API_BACKGROUND_RESUME', True)
    monkeypatch.setattr(openai, 'RESPONSES_API_BACKGROUND_RESUME_MODEL_ALLOWLIST', set())
    monkeypatch.setattr(openai, 'RESPONSES_API_BACKGROUND_RESUME_BASE_URL_ALLOWLIST', set())

    assert not openai.responses_background_resume_enabled('gpt-5-long', 'https://cpa.example/v1', {})


def test_background_resume_accepts_model_allowlist(monkeypatch):
    monkeypatch.setattr(openai, 'ENABLE_RESPONSES_API_BACKGROUND_RESUME', True)
    monkeypatch.setattr(openai, 'RESPONSES_API_BACKGROUND_RESUME_MODEL_ALLOWLIST', {'gpt-5-long'})
    monkeypatch.setattr(openai, 'RESPONSES_API_BACKGROUND_RESUME_BASE_URL_ALLOWLIST', set())

    assert openai.responses_background_resume_enabled('gpt-5-long', 'https://cpa.example/v1', {})


def test_background_resume_accepts_any_model_candidate(monkeypatch):
    monkeypatch.setattr(openai, 'ENABLE_RESPONSES_API_BACKGROUND_RESUME', True)
    monkeypatch.setattr(openai, 'RESPONSES_API_BACKGROUND_RESUME_MODEL_ALLOWLIST', {'gpt-5-long'})
    monkeypatch.setattr(openai, 'RESPONSES_API_BACKGROUND_RESUME_BASE_URL_ALLOWLIST', set())

    assert openai.responses_background_resume_enabled(
        ['ui-custom-model', 'gpt-5-long'],
        'https://cpa.example/v1',
        {},
    )


def test_background_resume_model_ids_dedupe_custom_and_base_model():
    assert openai._responses_background_resume_model_ids(
        'ui-custom-model',
        'gpt-5-long',
        ['ui-custom-model', 'gpt-5-long'],
    ) == ['ui-custom-model', 'gpt-5-long']


def test_background_resume_accepts_base_url_allowlist(monkeypatch):
    monkeypatch.setattr(openai, 'ENABLE_RESPONSES_API_BACKGROUND_RESUME', True)
    monkeypatch.setattr(openai, 'RESPONSES_API_BACKGROUND_RESUME_MODEL_ALLOWLIST', set())
    monkeypatch.setattr(
        openai,
        'RESPONSES_API_BACKGROUND_RESUME_BASE_URL_ALLOWLIST',
        {'https://cpa.example/v1'},
    )

    assert openai.responses_background_resume_enabled(
        'not-allowlisted',
        'https://cpa.example/v1/',
        {},
    )


def test_background_resume_azure_requires_explicit_route_opt_in(monkeypatch):
    monkeypatch.setattr(openai, 'ENABLE_RESPONSES_API_BACKGROUND_RESUME', True)
    monkeypatch.setattr(openai, 'RESPONSES_API_BACKGROUND_RESUME_MODEL_ALLOWLIST', {'gpt-5-long'})
    monkeypatch.setattr(openai, 'RESPONSES_API_BACKGROUND_RESUME_BASE_URL_ALLOWLIST', {'https://azure.example'})

    assert not openai.responses_background_resume_enabled(
        'gpt-5-long',
        'https://azure.example',
        {'azure': True},
    )
    assert openai.responses_background_resume_enabled(
        'gpt-5-long',
        'https://azure.example',
        {'azure': True, 'responses_background_resume': True},
    )


@pytest.mark.asyncio
async def test_resolve_openai_route_rejects_missing_model_without_defaulting(monkeypatch):
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                OPENAI_MODELS={},
                config=SimpleNamespace(
                    OPENAI_API_BASE_URLS=['https://wrong.example/v1'],
                    OPENAI_API_KEYS=['key'],
                    OPENAI_API_CONFIGS={},
                ),
            )
        )
    )

    async def fake_get_all_models(request, user=None):
        request.app.state.OPENAI_MODELS = {}

    monkeypatch.setattr(openai, 'get_all_models', fake_get_all_models)

    with pytest.raises(HTTPException) as exc_info:
        await openai.resolve_openai_route(request, model_id='missing-model', user=SimpleNamespace())

    assert exc_info.value.status_code == 400
    assert 'refusing to default to upstream index 0' in exc_info.value.detail


@pytest.mark.asyncio
async def test_resolve_openai_route_uses_stored_route_idx():
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                OPENAI_MODELS={},
                config=SimpleNamespace(
                    OPENAI_API_BASE_URLS=['https://first.example/v1', 'https://second.example/v1'],
                    OPENAI_API_KEYS=['key-1', 'key-2'],
                    OPENAI_API_CONFIGS={'1': {'responses_background_resume': True}},
                ),
            )
        )
    )

    idx, url, key, api_config = await openai.resolve_openai_route(
        request,
        model_id='missing-model',
        user=SimpleNamespace(),
        route_idx=1,
    )

    assert idx == 1
    assert url == 'https://second.example/v1'
    assert key == 'key-2'
    assert api_config == {'responses_background_resume': True}


@pytest.mark.asyncio
async def test_resolve_openai_route_rejects_invalid_stored_route_idx():
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                OPENAI_MODELS={},
                config=SimpleNamespace(
                    OPENAI_API_BASE_URLS=['https://first.example/v1'],
                    OPENAI_API_KEYS=['key-1'],
                    OPENAI_API_CONFIGS={},
                ),
            )
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        await openai.resolve_openai_route(
            request,
            model_id='any-model',
            user=SimpleNamespace(),
            route_idx=3,
        )

    assert exc_info.value.status_code == 400
    assert 'Stored response route is invalid' in exc_info.value.detail


@pytest.mark.asyncio
async def test_stream_retries_resume_background_before_contextual_retry(monkeypatch):
    monkeypatch.setattr(middleware, 'ENABLE_RESPONSES_API_BACKGROUND_RESUME', True)
    monkeypatch.setattr(middleware, 'RESPONSES_API_BACKGROUND_RESUME_ATTEMPTS', 1)
    monkeypatch.setattr(middleware, 'CHAT_RESPONSE_STREAM_RETRY_ATTEMPTS', 1)

    resume_calls = []

    async def fake_resume_response_stream(**kwargs):
        resume_calls.append(kwargs)
        return _sse_response(
            [_completed_response_event(response_id=kwargs['response_id'], sequence_number=2)],
            route_idx=kwargs['route_idx'],
        )

    async def fake_generate_chat_completion(*args, **kwargs):
        raise AssertionError('contextual retry should not run when background resume succeeds')

    monkeypatch.setattr(middleware, 'resume_response_stream', fake_resume_response_stream)
    monkeypatch.setattr(middleware, 'generate_chat_completion', fake_generate_chat_completion)

    response = _sse_response(
        [
            {
                'type': 'response.in_progress',
                'sequence_number': 1,
                'response': {'id': 'resp_background'},
            }
        ],
        route_idx=2,
    )

    events, upserts = await _run_streaming_handler(monkeypatch, response)

    assert resume_calls == [
        {
            'request': resume_calls[0]['request'],
            'model_id': 'gpt-5-long',
            'response_id': 'resp_background',
            'starting_after': 1,
            'route_idx': 2,
            'user': resume_calls[0]['user'],
        }
    ]
    assert any(update.get('done') is True for update in upserts)
    assert not any(event.get('data', {}).get('error') for event in events)


@pytest.mark.asyncio
async def test_stream_retries_fall_back_after_non_stream_background_resume(monkeypatch):
    monkeypatch.setattr(middleware, 'ENABLE_RESPONSES_API_BACKGROUND_RESUME', True)
    monkeypatch.setattr(middleware, 'RESPONSES_API_BACKGROUND_RESUME_ATTEMPTS', 1)
    monkeypatch.setattr(middleware, 'CHAT_RESPONSE_STREAM_RETRY_ATTEMPTS', 1)

    call_order = []
    contextual_forms = []

    async def fake_resume_response_stream(**kwargs):
        call_order.append('background')
        return JSONResponse(status_code=404, content={'error': {'message': 'not found'}})

    async def fake_generate_chat_completion(request, form_data, user, **kwargs):
        call_order.append('contextual')
        contextual_forms.append(form_data)
        return _sse_response(
            [_completed_response_event(response_id='resp_contextual', sequence_number=2)],
            route_idx=2,
        )

    monkeypatch.setattr(middleware, 'resume_response_stream', fake_resume_response_stream)
    monkeypatch.setattr(middleware, 'generate_chat_completion', fake_generate_chat_completion)

    form_data = {
        'model': 'gpt-5-long',
        'stream': True,
        'previous_response_id': 'resp_completed_previous',
        'messages': [{'role': 'user', 'content': 'Hello'}],
    }
    response = _sse_response(
        [
            {
                'type': 'response.in_progress',
                'sequence_number': 1,
                'response': {'id': 'resp_original'},
            }
        ],
        route_idx=2,
    )

    events, upserts = await _run_streaming_handler(monkeypatch, response, form_data=form_data)

    assert call_order == ['background', 'contextual']
    assert contextual_forms
    assert 'previous_response_id' not in contextual_forms[0]
    assert any(update.get('done') is True for update in upserts)
    assert not any(event.get('data', {}).get('error') for event in events)


@pytest.mark.asyncio
async def test_stream_retries_missing_route_idx_fails_closed_without_resume(monkeypatch):
    monkeypatch.setattr(middleware, 'ENABLE_RESPONSES_API_BACKGROUND_RESUME', True)
    monkeypatch.setattr(middleware, 'RESPONSES_API_BACKGROUND_RESUME_ATTEMPTS', 1)
    monkeypatch.setattr(middleware, 'CHAT_RESPONSE_STREAM_RETRY_ATTEMPTS', 0)

    async def fake_resume_response_stream(**kwargs):
        raise AssertionError('resume must not be called without a stored response route')

    monkeypatch.setattr(middleware, 'resume_response_stream', fake_resume_response_stream)

    response = _sse_response(
        [
            {
                'type': 'response.in_progress',
                'sequence_number': 7,
                'response': {'id': 'resp_no_route'},
            }
        ],
        route_idx=None,
        route_url=None,
    )

    events, upserts = await _run_streaming_handler(monkeypatch, response)

    final_error = next(update for update in reversed(upserts) if update.get('error'))
    assert final_error['done'] is False
    assert final_error['error']['content']['code'] == 'stream_incomplete_eof'
    assert final_error['response_id'] == 'resp_no_route'
    assert final_error['response_sequence_number'] == 7
    assert 'response_route_idx' not in final_error
    assert events[-1]['data']['done'] is False


@pytest.mark.asyncio
async def test_empty_responses_stream_eof_persists_done_false(monkeypatch):
    monkeypatch.setattr(middleware, 'ENABLE_RESPONSES_API_BACKGROUND_RESUME', False)
    monkeypatch.setattr(middleware, 'CHAT_RESPONSE_STREAM_RETRY_ATTEMPTS', 0)

    response = _sse_response([], route_idx=2)

    events, upserts = await _run_streaming_handler(monkeypatch, response)

    final_error = next(update for update in reversed(upserts) if update.get('error'))
    assert final_error['done'] is False
    assert final_error['error']['content']['code'] == 'stream_empty_eof'
    assert final_error['response_route_idx'] == 2
    assert events[-1]['data']['done'] is False


@pytest.mark.asyncio
async def test_tool_followup_contextual_retry_sanitizes_base_messages_and_preserves_bypass(monkeypatch):
    monkeypatch.setattr(middleware, 'ENABLE_RESPONSES_API_BACKGROUND_RESUME', False)
    monkeypatch.setattr(middleware, 'CHAT_RESPONSE_STREAM_RETRY_ATTEMPTS', 1)
    monkeypatch.setattr(middleware, 'RESPONSES_API_CONTEXTUAL_RETRY_TOOL_ALLOWLIST', set())

    async def unsafe_tool():
        return 'secret side effect result'

    generate_calls = []

    async def fake_generate_chat_completion(request, form_data, user, bypass_system_prompt=False):
        generate_calls.append(
            {
                'form_data': form_data,
                'bypass_system_prompt': bypass_system_prompt,
            }
        )
        if len(generate_calls) == 1:
            return _sse_response(
                [
                    {
                        'type': 'response.in_progress',
                        'sequence_number': 10,
                        'response': {'id': 'resp_tool_followup'},
                    }
                ],
                route_idx=2,
            )

        return _sse_response(
            [_completed_response_event(response_id='resp_contextual', sequence_number=11)],
            route_idx=2,
        )

    monkeypatch.setattr(middleware, 'generate_chat_completion', fake_generate_chat_completion)

    metadata = {
        'chat_id': 'chat_1',
        'message_id': 'message_1',
        'session_id': 'session_1',
        'params': {},
        'tools': {
            'unsafe_tool': {
                'callable': unsafe_tool,
                'spec': {'parameters': {'properties': {}}},
                'type': 'function',
                'direct': False,
            }
        },
    }
    form_data = {
        'model': 'gpt-5-long',
        'stream': True,
        'messages': [{'role': 'user', 'content': 'Use the tool'}],
        'metadata': metadata,
    }
    function_call = {
        'type': 'function_call',
        'id': 'fc_1',
        'call_id': 'call_unsafe',
        'name': 'unsafe_tool',
        'arguments': '{}',
        'status': 'completed',
    }
    response = _sse_response(
        [
            {
                'type': 'response.output_item.done',
                'sequence_number': 1,
                'response_id': 'resp_initial',
                'output_index': 0,
                'item': function_call,
            },
            {
                'type': 'response.completed',
                'sequence_number': 2,
                'response': {
                    'id': 'resp_initial',
                    'output': [function_call],
                    'usage': {},
                },
            },
        ],
        route_idx=2,
    )

    events, upserts = await _run_streaming_handler(monkeypatch, response, form_data=form_data)

    assert len(generate_calls) == 2
    assert generate_calls[0]['bypass_system_prompt'] is True
    assert generate_calls[1]['bypass_system_prompt'] is True

    contextual_messages = generate_calls[1]['form_data']['messages']
    assert {'role': 'user', 'content': 'Use the tool'} in contextual_messages
    assert all(message.get('role') != 'tool' for message in contextual_messages)
    assert all(not message.get('tool_calls') for message in contextual_messages)
    assert any(
        'interrupted before a final response.completed event' in message.get('content', '')
        for message in contextual_messages
        if message.get('role') == 'user'
    )
    assert any(update.get('done') is True for update in upserts)
    assert not any(event.get('data', {}).get('error') for event in events)


def test_contextual_retry_cleaning_drops_unallowlisted_completed_tool_items(monkeypatch):
    monkeypatch.setattr(middleware, 'RESPONSES_API_CONTEXTUAL_RETRY_TOOL_ALLOWLIST', {'safe_tool'})

    cleaned = middleware._clean_output_for_contextual_retry(
        [
            {
                'type': 'message',
                'status': 'completed',
                'content': [{'type': 'output_text', 'text': 'partial answer'}],
            },
            {
                'type': 'function_call',
                'call_id': 'safe_call',
                'name': 'safe_tool',
                'arguments': '{}',
                'status': 'completed',
            },
            {
                'type': 'function_call_output',
                'call_id': 'safe_call',
                'output': [{'type': 'input_text', 'text': 'safe result'}],
            },
            {
                'type': 'function_call',
                'call_id': 'unsafe_call',
                'name': 'unsafe_tool',
                'arguments': '{}',
                'status': 'completed',
            },
            {
                'type': 'function_call_output',
                'call_id': 'unsafe_call',
                'output': [{'type': 'input_text', 'text': 'unsafe result'}],
            },
            {'type': 'web_search_call', 'status': 'completed', 'action': {'type': 'search'}},
            {'type': 'file_search_call', 'status': 'completed', 'queries': ['private']},
            {'type': 'code_interpreter_call', 'status': 'completed', 'code': 'print(secret)'},
            {'type': 'image_generation_call', 'status': 'completed', 'result': 'generated'},
            {'type': 'unknown_completed_tool', 'status': 'completed', 'result': 'unknown'},
        ]
    )

    assert [item['type'] for item in cleaned] == [
        'message',
        'function_call',
        'function_call_output',
        'web_search_call',
    ]
    assert all('unsafe' not in json.dumps(item) for item in cleaned)
    assert all('private' not in json.dumps(item) for item in cleaned)
    assert all('secret' not in json.dumps(item) for item in cleaned)


@pytest.mark.asyncio
async def test_hosted_tool_items_are_dropped_before_contextual_retry(monkeypatch):
    monkeypatch.setattr(middleware, 'ENABLE_RESPONSES_API_BACKGROUND_RESUME', False)
    monkeypatch.setattr(middleware, 'CHAT_RESPONSE_STREAM_RETRY_ATTEMPTS', 1)

    generate_calls = []

    async def fake_generate_chat_completion(request, form_data, user, bypass_system_prompt=False):
        generate_calls.append(
            {
                'form_data': form_data,
                'bypass_system_prompt': bypass_system_prompt,
            }
        )
        return _sse_response(
            [_completed_response_event(response_id='resp_contextual', sequence_number=11)],
            route_idx=2,
        )

    monkeypatch.setattr(middleware, 'generate_chat_completion', fake_generate_chat_completion)

    response = _sse_response(
        [
            {
                'type': 'response.output_item.done',
                'sequence_number': 1,
                'response_id': 'resp_hosted_tool',
                'output_index': 0,
                'item': {
                    'type': 'file_search_call',
                    'id': 'fs_1',
                    'status': 'completed',
                    'queries': ['secret private index'],
                },
            }
        ],
        route_idx=2,
    )

    events, upserts = await _run_streaming_handler(monkeypatch, response)

    assert len(generate_calls) == 1
    contextual_messages = generate_calls[0]['form_data']['messages']
    serialized_context = json.dumps(contextual_messages)
    assert 'file_search_call' not in serialized_context
    assert 'secret private index' not in serialized_context
    assert any(
        'interrupted before a final response.completed event' in message.get('content', '')
        for message in contextual_messages
        if message.get('role') == 'user'
    )
    assert any(update.get('done') is True for update in upserts)
    assert not any(event.get('data', {}).get('error') for event in events)


@pytest.mark.asyncio
async def test_code_interpreter_followup_stream_error_does_not_run_contextual_retry(monkeypatch):
    monkeypatch.setattr(middleware, 'ENABLE_RESPONSES_API_BACKGROUND_RESUME', False)
    monkeypatch.setattr(middleware, 'CHAT_RESPONSE_STREAM_RETRY_ATTEMPTS', 1)

    generate_calls = []

    async def fake_generate_chat_completion(request, form_data, user, bypass_system_prompt=False):
        generate_calls.append(
            {
                'form_data': form_data,
                'bypass_system_prompt': bypass_system_prompt,
            }
        )
        if len(generate_calls) == 1:
            return _sse_response(
                [
                    {
                        'type': 'response.in_progress',
                        'sequence_number': 10,
                        'response': {'id': 'resp_ci_followup'},
                    }
                ],
                route_idx=2,
            )

        return _sse_response(
            [_completed_response_event(response_id='resp_ci_contextual', sequence_number=11)],
            route_idx=2,
        )

    monkeypatch.setattr(middleware, 'generate_chat_completion', fake_generate_chat_completion)

    form_data = {
        'model': 'gpt-5-long',
        'stream': True,
        'messages': [{'role': 'user', 'content': 'Run code'}],
        'metadata': {
            'chat_id': 'chat_1',
            'message_id': 'message_1',
            'session_id': 'session_1',
            'params': {},
            'features': {'code_interpreter': True},
        },
    }
    code_item = {
        'type': 'open_webui:code_interpreter',
        'id': 'ci_1',
        'status': 'in_progress',
        'attributes': {'type': 'code'},
        'code': 'print("secret local computation")',
    }
    response = _sse_response(
        [
            {
                'type': 'response.completed',
                'sequence_number': 1,
                'response': {
                    'id': 'resp_initial',
                    'output': [code_item],
                    'usage': {},
                },
            },
        ],
        route_idx=2,
    )

    events, upserts = await _run_streaming_handler(
        monkeypatch,
        response,
        form_data=form_data,
        event_caller_result={'stdout': 'secret local computation output'},
    )

    assert len(generate_calls) == 1
    assert generate_calls[0]['bypass_system_prompt'] is True

    followup_messages = generate_calls[0]['form_data']['messages']
    serialized_followup = json.dumps(followup_messages)
    assert 'code_interpreter' in serialized_followup
    assert 'secret local computation' in serialized_followup
    assert not any(
        'interrupted before a final response.completed event' in message.get('content', '')
        for message in followup_messages
        if message.get('role') == 'user'
    )

    final_error = next(update for update in reversed(upserts) if update.get('error'))
    assert final_error['done'] is False
    assert final_error['error']['content']['code'] == 'stream_incomplete_eof'
    assert events[-1]['data']['done'] is False
