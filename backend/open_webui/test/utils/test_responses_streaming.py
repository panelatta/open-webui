from open_webui.utils.middleware import (
    build_responses_reasoning_placeholder,
    build_responses_web_search_status,
    _count_stream_retry_visible_chars,
    _is_retryable_stream_error,
    _stream_retryable,
    handle_responses_streaming_event,
    serialize_output,
)
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
