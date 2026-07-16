import asyncio

from open_webui.utils.memory import _emit_memory_search_status, build_memory_query


def test_build_memory_query_keeps_latest_message_and_caps_total_length():
    messages = [
        {'role': 'user', 'content': 'a' * 200},
        {'role': 'assistant', 'content': 'ignored assistant text'},
        {'role': 'user', 'content': 'latest'},
    ]

    query = build_memory_query(messages, message_limit=2, char_limit=30)

    assert query == f'{"a" * 22}\n\nlatest'
    assert len(query) == 30


def test_build_memory_query_limits_number_of_user_messages():
    messages = [
        {'role': 'user', 'content': 'oldest'},
        {'role': 'assistant', 'content': 'ignored assistant text'},
        {'role': 'user', 'content': 'context'},
        {'role': 'user', 'content': 'latest'},
    ]

    query = build_memory_query(messages, message_limit=2, char_limit=100)

    assert query == 'context\n\nlatest'


def test_build_memory_query_preserves_head_and_tail_of_long_latest_message():
    content = f'BEGIN{"x" * 100}END'

    query = build_memory_query(
        [{'role': 'user', 'content': content}],
        message_limit=1,
        char_limit=30,
    )

    assert len(query) == 30
    assert query.startswith('BEGIN')
    assert query.endswith('END')
    assert '\n…\n' in query


def test_emit_memory_search_status_uses_status_history_shape():
    events = []

    async def event_emitter(event):
        events.append(event)

    asyncio.run(
        _emit_memory_search_status(
            event_emitter,
            'Searching memories',
            done=False,
        )
    )

    assert events == [
        {
            'type': 'status',
            'data': {
                'action': 'memory_search',
                'description': 'Searching memories',
                'done': False,
                'error': False,
            },
        }
    ]
