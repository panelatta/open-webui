from open_webui.routers.openai import filter_disabled_response_tools


def test_filter_disabled_response_tools_removes_hosted_tool_and_include():
    payload = {
        "tools": [
            {"type": "web_search"},
            {"type": "function", "name": "lookup", "parameters": {}},
        ],
        "tool_choice": "auto",
        "include": [
            "web_search_call.action.sources",
            "file_search_call.results",
        ],
    }

    result = filter_disabled_response_tools(
        payload,
        {"disabled_response_tools": ["web_search"]},
    )

    assert result["tools"] == [
        {"type": "function", "name": "lookup", "parameters": {}}
    ]
    assert result["tool_choice"] == "auto"
    assert result["include"] == ["file_search_call.results"]


def test_filter_disabled_response_tools_removes_tool_choice_when_no_tools_remain():
    payload = {
        "tools": [{"type": "web_search"}],
        "tool_choice": "auto",
        "include": ["web_search_call.action.sources"],
    }

    result = filter_disabled_response_tools(
        payload,
        {"disabled_response_tools": "web_search"},
    )

    assert "tools" not in result
    assert "tool_choice" not in result
    assert "include" not in result


def test_filter_disabled_response_tools_removes_matching_specific_tool_choice():
    payload = {
        "tools": [{"type": "function", "name": "lookup", "parameters": {}}],
        "tool_choice": {"type": "web_search"},
    }

    result = filter_disabled_response_tools(
        payload,
        {"disabled_response_tools": ["web_search"]},
    )

    assert result["tools"] == [
        {"type": "function", "name": "lookup", "parameters": {}}
    ]
    assert "tool_choice" not in result
