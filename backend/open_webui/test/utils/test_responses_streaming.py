from open_webui.utils.middleware import (
    ResponsesStreamState,
    build_responses_reasoning_placeholder,
    build_responses_web_search_status,
    _clean_output_for_contextual_retry,
    _count_stream_retry_visible_chars,
    _is_retryable_stream_error,
    _next_response_background_resume_attempt,
    _responses_stream_cursor_from_error,
    _responses_stream_error_from_exception,
    _stream_retryable,
    handle_responses_streaming_event,
    serialize_output,
)
from open_webui.utils.misc import convert_web_search_output_to_resume_message
from open_webui.env import CHAT_RESPONSE_STREAM_RETRY_VISIBLE_CHAR_LIMIT


def test_output_item_added_appends_to_empty_output():
    output, metadata = handle_responses_streaming_event(
        {
            'type': 'response.output_item.added',
            'output_index': 0,
            'item': {
                'type': 'message',
                'id': 'msg_123',
                'status': 'in_progress',
                'role': 'assistant',
                'content': [],
            },
        },
        [],
    )

    assert metadata is None
    assert output == [
        {
            'type': 'message',
            'id': 'msg_123',
            'status': 'in_progress',
            'role': 'assistant',
            'content': [],
        }
    ]


def test_output_item_added_replaces_reasoning_placeholder():
    placeholder = build_responses_reasoning_placeholder()

    output, metadata = handle_responses_streaming_event(
        {
            'type': 'response.output_item.added',
            'output_index': 0,
            'item': {
                'type': 'reasoning',
                'id': 'rs_123',
                'summary': [],
                'content': [],
            },
        },
        [placeholder],
    )

    assert metadata is None
    assert len(output) == 1
    assert output[0]['id'] == 'rs_123'
    assert output[0]['type'] == 'reasoning'
    assert output[0]['status'] == 'in_progress'
    assert output[0]['started_at'] == placeholder['started_at']
    assert '_placeholder' not in output[0]


def test_web_search_completed_event_renders_before_response_completed():
    output, metadata = handle_responses_streaming_event(
        {
            'type': 'response.web_search_call.completed',
            'output_index': 0,
            'item_id': 'ws_123',
        },
        [
            {
                'type': 'web_search_call',
                'id': 'ws_123',
                'status': 'in_progress',
                'action': {
                    'type': 'search',
                    'query': 'codex sandbox settings',
                    'sources': [{'type': 'url', 'url': 'https://example.com/codex'}],
                },
            }
        ],
    )

    assert metadata == {}
    assert output[0]['status'] == 'completed'

    rendered = serialize_output(output)
    assert 'done="true"' in rendered
    assert 'codex sandbox settings' in rendered
    assert 'https://example.com/codex' in rendered


def test_web_search_output_item_done_merges_sources_before_response_completed():
    output, metadata = handle_responses_streaming_event(
        {
            'type': 'response.output_item.done',
            'output_index': 0,
            'item': {
                'type': 'web_search_call',
                'id': 'ws_123',
                'status': 'completed',
                'action': {
                    'type': 'search',
                    'query': 'codex sandbox settings',
                    'sources': [
                        {
                            'type': 'url',
                            'url': 'https://example.com/codex',
                            'title': 'Codex sandbox',
                        }
                    ],
                },
            },
        },
        [
            {
                'type': 'web_search_call',
                'id': 'ws_123',
                'status': 'in_progress',
            }
        ],
    )

    assert metadata == {}
    assert output[0]['status'] == 'completed'
    assert output[0]['action']['query'] == 'codex sandbox settings'

    status = build_responses_web_search_status(output[0])
    assert status == {
        'action': 'web_search',
        'description': 'Searched {{count}} sites',
        'done': True,
        'id': 'ws_123',
        'query': 'codex sandbox settings',
        'items': [{'link': 'https://example.com/codex', 'title': 'Codex sandbox'}],
    }


def test_web_search_call_builds_realtime_status_with_sources():
    status = build_responses_web_search_status(
        {
            'type': 'web_search_call',
            'id': 'ws_123',
            'status': 'completed',
            'action': {
                'type': 'search',
                'query': 'codex sandbox settings',
                'sources': [
                    {
                        'type': 'url',
                        'url': 'https://example.com/codex',
                        'title': 'Codex sandbox',
                    }
                ],
            },
        }
    )

    assert status == {
        'action': 'web_search',
        'description': 'Searched {{count}} sites',
        'done': True,
        'id': 'ws_123',
        'query': 'codex sandbox settings',
        'items': [{'link': 'https://example.com/codex', 'title': 'Codex sandbox'}],
    }


def test_web_search_call_builds_realtime_status_while_searching():
    status = build_responses_web_search_status(
        {
            'type': 'web_search_call',
            'id': 'ws_123',
            'status': 'in_progress',
            'action': {
                'type': 'search',
                'queries': ['codex sandbox settings'],
            },
        }
    )

    assert status == {
        'action': 'web_search',
        'description': 'Searching "{{searchQuery}}"',
        'done': False,
        'id': 'ws_123',
        'query': 'codex sandbox settings',
    }


def test_retryable_stream_error_detection_accepts_stream_read_error():
    assert _is_retryable_stream_error(
        {
            'code': 'stream_read_error',
            'message': 'stream_read_error',
            'type': 'upstream_error',
        }
    )


def test_retryable_stream_error_detection_accepts_incomplete_eof():
    assert _is_retryable_stream_error(
        {
            'code': 'stream_incomplete_eof',
            'message': 'Responses API stream ended before response.completed.',
            'type': 'upstream_error',
        }
    )


def test_retryable_stream_error_detection_accepts_empty_eof():
    assert _is_retryable_stream_error(
        {
            'code': 'stream_empty_eof',
            'message': 'Responses API stream ended before any response event was received.',
            'type': 'upstream_error',
        }
    )


def test_retryable_stream_error_detection_rejects_other_provider_errors():
    assert not _is_retryable_stream_error(
        {
            'code': 'rate_limit_exceeded',
            'message': 'quota exceeded',
            'type': 'invalid_request_error',
        }
    )


def test_stream_retry_budget_blocks_large_partial_outputs():
    output = [
        {
            'type': 'message',
            'content': [{'type': 'output_text', 'text': 'x' * (CHAT_RESPONSE_STREAM_RETRY_VISIBLE_CHAR_LIMIT + 1)}],
        }
    ]

    assert _count_stream_retry_visible_chars(output) > CHAT_RESPONSE_STREAM_RETRY_VISIBLE_CHAR_LIMIT
    assert not _stream_retryable(output)


def test_stream_retry_budget_allows_small_outputs():
    output = [
        {
            'type': 'reasoning',
            'content': [{'type': 'output_text', 'text': 'brief progress'}],
        }
    ]

    assert _count_stream_retry_visible_chars(output) == len('brief progress')
    assert _stream_retryable(output)


def test_responses_stream_state_records_response_id_and_sequence_number():
    state = ResponsesStreamState(route_idx=2, route_url='https://cpa.example/v1')
    state.observe(
        {
            'type': 'response.in_progress',
            'sequence_number': 7,
            'response': {
                'id': 'resp_123',
            },
        },
        [],
    )

    assert state.response_id == 'resp_123'
    assert state.last_sequence_number == 7


def test_responses_stream_state_reports_empty_eof_as_error():
    state = ResponsesStreamState(route_idx=2, route_url='https://cpa.example/v1')

    error = state.incomplete_error()

    assert error['code'] == 'stream_empty_eof'
    assert error['response_route_idx'] == 2
    assert error['response_route_url'] == 'https://cpa.example/v1'


def test_responses_stream_state_reports_incomplete_eof_with_cursor():
    state = ResponsesStreamState(route_idx=2, route_url='https://cpa.example/v1')
    state.observe(
        {
            'type': 'response.output_item.done',
            'sequence_number': 42,
            'response_id': 'resp_123',
            'output_index': 0,
            'item': {
                'type': 'web_search_call',
                'id': 'ws_123',
                'status': 'completed',
            },
        },
        [],
    )

    error = state.incomplete_error()

    assert error['code'] == 'stream_incomplete_eof'
    assert error['response_id'] == 'resp_123'
    assert error['last_sequence_number'] == 42
    assert error['response_route_idx'] == 2
    assert error['last_event_type'] == 'response.output_item.done'
    assert error['last_output_item_type'] == 'web_search_call'
    assert error['last_output_item_status'] == 'completed'


def test_responses_stream_state_accepts_completed():
    state = ResponsesStreamState()
    state.observe(
        {
            'type': 'response.completed',
            'sequence_number': 99,
            'response': {
                'id': 'resp_123',
            },
        },
        [],
    )

    assert state.incomplete_error() is None


def test_response_failed_without_error_is_fatal_metadata():
    output, metadata = handle_responses_streaming_event(
        {
            'type': 'response.failed',
            'sequence_number': 12,
            'response': {
                'id': 'resp_failed',
            },
        },
        [],
    )

    assert output == []
    assert metadata['error']['code'] == 'response_failed'
    assert metadata['error']['response_id'] == 'resp_failed'
    assert metadata['error']['last_sequence_number'] == 12
    assert metadata['error']['last_event_type'] == 'response.failed'
    assert not _is_retryable_stream_error(metadata['error'])


def test_responses_transport_error_preserves_resume_cursor():
    state = ResponsesStreamState(route_idx=2, route_url='https://cpa.example/v1')
    state.observe(
        {
            'type': 'response.output_item.done',
            'sequence_number': 42,
            'response_id': 'resp_123',
            'item': {'type': 'message', 'status': 'completed'},
        },
        [],
    )

    error = _responses_stream_error_from_exception(TimeoutError('read timeout'), state)

    assert error['code'] == 'stream_incomplete_eof'
    assert error['response_id'] == 'resp_123'
    assert error['last_sequence_number'] == 42
    assert error['response_route_idx'] == 2
    assert error['transport_error']['code'] == 'TimeoutError'


def test_responses_cursor_payload_falls_back_to_latest_state():
    state = ResponsesStreamState(route_idx=2, route_url='https://cpa.example/v1')
    state.observe(
        {
            'type': 'response.output_item.done',
            'sequence_number': 42,
            'response_id': 'resp_123',
            'item': {'type': 'message', 'status': 'completed'},
        },
        [],
    )

    cursor = _responses_stream_cursor_from_error(
        {
            'code': 'stream_retry_non_streaming_response',
            'message': 'retry returned a non-streaming response',
            'type': 'upstream_error',
        },
        state,
    )

    assert cursor == {
        'response_id': 'resp_123',
        'response_sequence_number': 42,
        'response_route_idx': 2,
        'response_route_url': 'https://cpa.example/v1',
    }


def test_responses_cursor_payload_prefers_error_cursor():
    state = ResponsesStreamState(route_idx=2, route_url='https://cpa.example/v1')
    state.observe(
        {
            'type': 'response.output_item.done',
            'sequence_number': 42,
            'response_id': 'resp_123',
            'item': {'type': 'message', 'status': 'completed'},
        },
        [],
    )

    cursor = _responses_stream_cursor_from_error(
        {
            'response_id': 'resp_456',
            'last_sequence_number': 77,
            'response_route_idx': 3,
            'response_route_url': 'https://other.example/v1',
        },
        state,
    )

    assert cursor == {
        'response_id': 'resp_456',
        'response_sequence_number': 77,
        'response_route_idx': 3,
        'response_route_url': 'https://other.example/v1',
    }


def test_background_resume_attempts_are_counted_per_response_id():
    attempts = {}

    assert _next_response_background_resume_attempt(
        attempts,
        {'response_id': 'resp_original'},
        1,
    ) == 1
    assert _next_response_background_resume_attempt(
        attempts,
        {'response_id': 'resp_original'},
        1,
    ) is None
    assert _next_response_background_resume_attempt(
        attempts,
        {'response_id': 'resp_contextual'},
        1,
    ) == 1


def test_background_resume_attempt_requires_response_id():
    attempts = {}

    assert _next_response_background_resume_attempt(
        attempts,
        {'code': 'stream_read_error'},
        1,
    ) is None
    assert attempts == {}


def test_responses_transport_error_after_completed_is_not_retryable():
    state = ResponsesStreamState(route_idx=2, route_url='https://cpa.example/v1')
    state.observe(
        {
            'type': 'response.completed',
            'sequence_number': 99,
            'response': {'id': 'resp_123'},
        },
        [],
    )

    assert _responses_stream_error_from_exception(TimeoutError('late close'), state) is None


def test_non_responses_transport_error_stays_generic():
    error = _responses_stream_error_from_exception(TimeoutError('read timeout'), ResponsesStreamState())

    assert error == {
        'code': 'TimeoutError',
        'message': 'read timeout',
        'type': 'transport_error',
    }


def test_stream_retry_budget_allows_small_function_call_output():
    output = [
        {
            'type': 'function_call_output',
            'call_id': 'call_123',
            'output': [{'type': 'input_text', 'text': 'tool result'}],
        }
    ]

    assert _stream_retryable(output)


def test_contextual_retry_drops_function_call_without_output():
    output = [
        {
            'type': 'function_call',
            'call_id': 'call_123',
            'name': 'expensive_tool',
            'arguments': '{"q":"x"}',
        }
    ]

    assert _clean_output_for_contextual_retry(output) == []


def test_contextual_retry_drops_function_call_output_when_tool_not_allowlisted():
    output = [
        {
            'type': 'function_call',
            'call_id': 'call_123',
            'name': 'expensive_or_unknown_tool',
            'arguments': '{"q":"x"}',
        },
        {
            'type': 'function_call_output',
            'call_id': 'call_123',
            'output': [{'type': 'input_text', 'text': 'done'}],
        },
    ]

    assert _clean_output_for_contextual_retry(output) == []


def test_contextual_retry_keeps_allowlisted_function_call_with_output(monkeypatch):
    monkeypatch.setattr(
        'open_webui.utils.middleware.RESPONSES_API_CONTEXTUAL_RETRY_TOOL_ALLOWLIST',
        {'safe_readonly_tool'},
    )

    output = [
        {
            'type': 'function_call',
            'call_id': 'call_123',
            'name': 'safe_readonly_tool',
            'arguments': '{"q":"x"}',
        },
        {
            'type': 'function_call_output',
            'call_id': 'call_123',
            'output': [{'type': 'input_text', 'text': 'done'}],
        },
    ]

    cleaned = _clean_output_for_contextual_retry(output)

    assert [item['type'] for item in cleaned] == ['function_call', 'function_call_output']


def test_contextual_retry_drops_reasoning_summary():
    output = [
        {
            'type': 'reasoning',
            'status': 'in_progress',
            'summary': [{'type': 'summary_text', 'text': 'internal plan'}],
        }
    ]

    assert _clean_output_for_contextual_retry(output) == []


def test_web_search_output_builds_resume_message():
    message = convert_web_search_output_to_resume_message(
        [
            {
                'type': 'web_search_call',
                'id': 'ws_123',
                'status': 'completed',
                'action': {
                    'type': 'search',
                    'query': 'openwebui stream interrupted',
                    'sources': [
                        {
                            'type': 'url',
                            'url': 'https://example.com/openwebui',
                            'title': 'OpenWebUI stream',
                        }
                    ],
                },
            }
        ]
    )

    assert message['role'] == 'assistant'
    assert 'openwebui stream interrupted' in message['content']
    assert 'https://example.com/openwebui' in message['content']
