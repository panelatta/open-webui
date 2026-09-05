from open_webui.routers.openai import convert_to_responses_payload


def test_responses_keeps_hosted_files_proxy_file_data_and_image_detail():
    payload = convert_to_responses_payload({
        'model': 'gpt-5',
        'messages': [{'role': 'user', 'content': [
            {'type': 'input_file', 'file_id': 'file-hosted'},
            {'type': 'file', 'file': {'file_data': 'data:text/plain;base64,aGk=', 'filename': 'test.txt'}},
            {'type': 'image_url', 'image_url': {'url': 'https://example.com/image.png', 'detail': 'high'}},
        ]}],
    })
    assert payload['input'][0]['content'] == [
        {'type': 'input_file', 'file_id': 'file-hosted'},
        {'type': 'input_file', 'file_data': 'data:text/plain;base64,aGk=', 'filename': 'test.txt'},
        {'type': 'input_image', 'image_url': 'https://example.com/image.png', 'detail': 'high'},
    ]


def test_convert_to_responses_payload_moves_developer_prompt_to_instructions():
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {
                "role": "developer",
                "content": "Use web search for current documentation questions.",
            },
            {
                "role": "user",
                "content": "Find the current Codex documentation.",
            },
        ],
        "tools": [{"type": "web_search"}],
        "tool_choice": "auto",
        "include": ["web_search_call.action.sources"],
    }

    result = convert_to_responses_payload(payload)

    assert result["instructions"] == "Use web search for current documentation questions."
    assert result["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Find the current Codex documentation.",
                }
            ],
        }
    ]
    assert result["tools"] == [{"type": "web_search"}]
    assert result["tool_choice"] == "auto"
    assert result["include"] == ["web_search_call.action.sources"]


def test_convert_to_responses_payload_moves_developer_text_parts_to_instructions():
    payload = {
        "messages": [
            {
                "role": "developer",
                "content": [
                    {"type": "text", "text": "A"},
                    {"type": "input_text", "text": "B"},
                ],
            }
        ]
    }

    assert convert_to_responses_payload(payload)["instructions"] == "A\nB"
