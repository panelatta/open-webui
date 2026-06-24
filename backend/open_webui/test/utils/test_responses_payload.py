from open_webui.routers.openai import convert_to_responses_payload


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
