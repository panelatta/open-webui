from open_webui.utils.middleware import (
    build_responses_reasoning_placeholder,
    handle_responses_streaming_event,
    serialize_output,
)


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
