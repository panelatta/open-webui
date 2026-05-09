from open_webui.utils.middleware import (
    build_responses_reasoning_placeholder,
    handle_responses_streaming_event,
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
