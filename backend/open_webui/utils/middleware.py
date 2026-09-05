import ast
import asyncio
import aiohttp
import base64
import copy
import html
import inspect
import json
import logging
import mimetypes
import os
import random
import re
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional
from uuid import uuid4

from aiocache import cached
from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse
from open_webui.config import (
    CACHE_DIR,
    CODE_INTERPRETER_BLOCKED_MODULES,
    CODE_INTERPRETER_PYODIDE_PROMPT,
    DEFAULT_CODE_INTERPRETER_PROMPT,
    DEFAULT_TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE,
    DEFAULT_VOICE_MODE_PROMPT_TEMPLATE,
)
from open_webui.constants import TASKS
from open_webui.env import (
    BYPASS_MODEL_ACCESS_CONTROL,
    CHAT_RESPONSE_MAX_TOOL_CALL_ITERATIONS,
    CHAT_RESPONSE_STREAM_DELTA_CHUNK_SIZE,
    CHAT_RESPONSE_STREAM_RETRY_ATTEMPTS,
    CHAT_RESPONSE_STREAM_RETRY_DELAY,
    CHAT_RESPONSE_STREAM_RETRY_VISIBLE_CHAR_LIMIT,
    ENABLE_API_OUTLET_FILTERS,
    ENABLE_CHAT_RESPONSE_BASE64_IMAGE_URL_CONVERSION,
    ENABLE_PLUGINS,
    ENABLE_QUERIES_CACHE,
    ENABLE_REALTIME_CHAT_SAVE,
    ENABLE_RESPONSES_API_BACKGROUND_RESUME,
    ENABLE_RESPONSES_API_STATEFUL,
    GLOBAL_LOG_LEVEL,
    RAG_SYSTEM_CONTEXT,
    RESPONSES_API_BACKGROUND_RESUME_ATTEMPTS,
    RESPONSES_API_CONTEXTUAL_RETRY_TOOL_ALLOWLIST,
)
from open_webui.events import EVENTS, publish_event
from open_webui.models.access_grants import AccessGrants
from open_webui.models.chats import Chats
from open_webui.models.config import Config
from open_webui.models.folders import Folders
from open_webui.models.models import Models
from open_webui.models.notes import Notes
from open_webui.models.oauth_sessions import OAuthSessions
from open_webui.models.users import UserModel, Users
from open_webui.retrieval.utils import get_sources_from_items
from open_webui.routers.images import (
    CreateImageForm,
    EditImageForm,
    image_edits,
    image_generations,
)
from open_webui.routers.pipelines import (
    get_sorted_filters,
    process_pipeline_inlet_filter,
    process_pipeline_outlet_filter,
)
from open_webui.routers.retrieval import (
    SearchForm,
    process_web_search,
)
from open_webui.routers.tasks import (
    generate_chat_tags,
    generate_follow_ups,
    generate_image_prompt,
    generate_queries,
    generate_title,
)
from open_webui.socket.main import (
    get_event_call,
    get_event_emitter,
)
from open_webui.tasks import clear_response_stream, save_response_stream
from open_webui.utils.access_control import has_connection_access, has_permission
from open_webui.utils.access_control.files import get_accessible_folder_files
from open_webui.utils.access_control.files import get_owner_accessible_folder_files
from open_webui.utils.access_control.folders import has_folder_access
from open_webui.utils.chat import generate_chat_completion, resume_response_stream
from open_webui.utils.ask_user import stage_ask_user_tool_calls
from open_webui.utils.chat_id import is_saved_chat_id
from open_webui.utils.code_interpreter import execute_code_jupyter
from open_webui.utils.context_compaction import compact_messages_for_request
from open_webui.utils.files import (
    convert_markdown_base64_images,
    get_file_url_from_base64,
    get_image_base64_from_url,
    get_image_url_from_base64,
)
from open_webui.utils.filter import (
    FilterContext,
    get_filter_context,
    get_filter_functions,
    process_filter_functions,
)
from open_webui.utils.json_codec import JSONCodec
from open_webui.utils.mcp.client import MCPClient
from open_webui.utils.memory import add_memory_context, review_memory_after_turn
from open_webui.utils.misc import (
    add_or_update_system_message,
    add_or_update_user_message,
    convert_web_search_output_to_resume_message,
    convert_output_to_messages,
    extract_urls,
    get_content_from_message,
    get_last_assistant_message,
    get_last_user_message,
    get_last_user_message_item,
    get_message_list,
    get_output_text,
    get_response_error_detail,
    get_reasoning_details,
    get_system_message,
    is_string_allowed,
    merge_system_messages,
    prepend_to_first_user_message_content,
    replace_system_message_content,
    set_last_user_message_content,
    strip_empty_content_blocks,
)
from open_webui.utils.payload import apply_params_to_form_data, apply_system_prompt_to_body, resolve_system_prompt
from open_webui.utils.plugin import load_function_module_by_id
from open_webui.utils.post_chat_memory import run_post_chat_memory_extractor
from open_webui.utils.response import merge_usage, normalize_usage
from open_webui.utils.sanitize import sanitize_code
from open_webui.utils.task import (
    get_task_model_id,
    rag_template,
    tools_function_calling_generation_template,
)
from open_webui.utils.tools import (
    build_tool_server_headers,
    get_attached_knowledge,
    get_builtin_tools,
    get_terminal_tools,
    get_tools,
    get_updated_tool_function,
)
from starlette.responses import JSONResponse, Response, StreamingResponse

logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)


def _is_tool_result_error(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if (
            text.startswith('error:')
            or text.startswith('exception:')
            or text.startswith('traceback')
            or text.startswith('http error!')
        ):
            return True

    parsed = value
    while isinstance(parsed, str):
        try:
            parsed = JSONCodec.loads(parsed)
        except (JSONCodec.JSONDecodeError, TypeError, ValueError):
            break

    if not isinstance(parsed, dict):
        return False

    error = parsed.get('error')
    if isinstance(error, str):
        has_error = bool(error.strip())
    else:
        has_error = isinstance(error, (dict, list)) and bool(error)
    if has_error:
        return True

    status = parsed.get('status')
    if isinstance(status, str) and status.strip().lower() in {'error', 'failed'}:
        return True

    if parsed.get('success') is False or parsed.get('ok') is False:
        message = parsed.get('message')
        return has_error or (
            bool(message.strip()) if isinstance(message, str) else isinstance(message, (dict, list)) and bool(message)
        )

    return False


def normalize_messages_for_model(form_data: dict) -> dict:
    form_data['messages'] = strip_empty_content_blocks(form_data.get('messages', []))
    form_data['messages'] = merge_system_messages(form_data.get('messages', []))
    return form_data


async def publish_chat_finished_event(
    request: Request, user: UserModel, metadata: dict, title: str, content: str, output: list | None = None
):
    chat_id = metadata.get('chat_id')
    if getattr(request.state, 'internal', False) is True or not is_saved_chat_id(chat_id):
        return

    content = content or get_output_text(output)
    webui_url = await Config.get('webui.url')
    await publish_event(
        request,
        EVENTS.CHAT_FINISHED,
        actor=user,
        subject_id=chat_id,
        subject_type='chat',
        data={
            'user_id': user.id,
            'chat_id': chat_id,
            'message_id': metadata.get('message_id'),
            'model_id': metadata.get('model_id'),
            'title': title,
            'url': f'{webui_url}/c/{chat_id}' if webui_url else f'/c/{chat_id}',
            'message': content,
        },
        message=title or 'Chat finished',
    )
    event_emitter = await get_event_emitter(metadata, update_db=False)
    if event_emitter:
        folder_id = metadata.get('folder_id') or await Chats.get_chat_folder_id(chat_id, metadata.get('user_id'))
        await event_emitter({'type': 'chat:list', 'data': {'chat_id': chat_id, 'folder_id': folder_id}})


# We believe in one maker of all models, seen and unseen,
# and in the reasoning which proceeds from the architect.
# We look for the resurrection of dead processes and the
# inference of the world to come.
DEFAULT_REASONING_TAGS = [
    ('<think>', '</think>'),
    ('<thinking>', '</thinking>'),
    ('<reason>', '</reason>'),
    ('<reasoning>', '</reasoning>'),
    ('<thought>', '</thought>'),
    ('<Thought>', '</Thought>'),
    ('<|begin_of_thought|>', '<|end_of_thought|>'),
    ('◁think▷', '◁/think▷'),
]

DEFAULT_SOLUTION_TAGS = [('<|begin_of_solution|>', '<|end_of_solution|>')]
DEFAULT_CODE_INTERPRETER_TAGS = [('<code_interpreter>', '</code_interpreter>')]


def _start_tag_pattern(start_tag: str) -> str:
    if start_tag.startswith('<') and start_tag.endswith('>'):
        return rf'<{re.escape(start_tag[1:-1])}(\s.*?)?>'
    return re.escape(start_tag)


def output_id(prefix: str) -> str:
    """Generate OR-style ID: prefix + 24-char hex UUID."""
    return f'{prefix}_{uuid4().hex[:24]}'


def build_terminal_file_tool_result(
    tool_function_name: str,
    tool_function_params: dict,
    tool_result: Any,
    tool: dict | None,
    metadata: dict | None,
) -> dict | None:
    if isinstance(tool_result, (list, tuple)) and tool_result and isinstance(tool_result[0], dict):
        tool_result = tool_result[0]

    if tool_function_name != 'display_file' or not isinstance(tool_result, dict) or tool_result.get('exists') is False:
        return None

    tool_id = (tool or {}).get('tool_id', '')
    terminal_id = metadata.get('terminal_id') if metadata else None
    if isinstance(tool_id, str) and tool_id.startswith('terminal:'):
        terminal_id = tool_id.split(':', 1)[1]

    server_url = ((tool or {}).get('server') or {}).get('url')
    terminal_selector = terminal_id or server_url
    path = tool_result.get('path') or tool_function_params.get('path')
    if not terminal_selector or not path:
        return None
    mime_type, _ = mimetypes.guess_type(path)
    mime_type = mime_type or 'application/octet-stream'
    page = tool_result.get('page') or tool_function_params.get('page')

    return {
        **tool_result,
        'type': 'file',
        'source': 'open_terminal',
        **({'displayed': True} if tool_function_params.get('inline') is True else {}),
        'terminal_selector': terminal_selector,
        **({'terminal_id': terminal_id} if terminal_id else {}),
        **({'terminal_url': server_url} if server_url and not terminal_id else {}),
        'session_id': metadata.get('chat_id') if metadata else None,
        'path': path,
        'full_path': tool_result.get('full_path') or path,
        'name': tool_result.get('name') or os.path.basename(path),
        'mime_type': tool_result.get('mime_type') or tool_result.get('content_type') or mime_type,
        'content_type': tool_result.get('content_type') or tool_result.get('mime_type') or mime_type,
        **({'page': page} if page else {}),
    }


def tool_result_content(tool_result: Any) -> str:
    if not tool_result:
        return ''
    if isinstance(tool_result, (dict, list)):
        return JSONCodec.dumps(tool_result, ensure_ascii=False)
    return str(tool_result)


def merge_streamed_reasoning_details(target: list, details) -> None:
    items = details if isinstance(details, list) else [details]
    for item in items:
        if not isinstance(item, dict):
            continue

        index = item.get('index')
        existing = (
            next((detail for detail in target if detail.get('index') == index), None)
            if isinstance(index, int)
            else None
        )
        if existing is None:
            target.append(dict(item))
            continue

        for key, value in item.items():
            if key in ('text', 'summary') and isinstance(value, str) and isinstance(existing.get(key), str):
                existing[key] += value
            else:
                existing[key] = value


def _split_tool_calls(
    tool_calls: list[dict],
) -> list[dict]:
    """Expand tool calls whose arguments contain multiple back-to-back JSON objects.

    Some models (e.g. GPT-5.4) send multiple complete JSON argument objects
    under the same tool call index, producing concatenated invalid JSON like:
        '{"query":"A","count":5}{"query":"B","count":5}'

    Each such tool call is split into separate entries so each gets executed
    independently. Single-object arguments pass through unchanged.
    """

    def split_json_objects(raw: str) -> list[str]:
        if not isinstance(raw, str):
            raw = '' if raw is None else JSONCodec.dumps(raw)

        decoder = json.JSONDecoder()
        results = []
        position = 0

        while position < len(raw):
            while position < len(raw) and raw[position].isspace():
                position += 1
            if position >= len(raw):
                break
            try:
                _, end = decoder.raw_decode(raw, position)
                results.append(raw[position:end].strip())
                position = end
            except JSONCodec.JSONDecodeError:
                return [raw]

        return results or [raw]

    expanded = []
    for tool_call in tool_calls:
        function = tool_call.setdefault('function', {})
        arguments = function.get('arguments')
        if not isinstance(arguments, str):
            arguments = '' if arguments is None else JSONCodec.dumps(arguments)
            function['arguments'] = arguments
        split_arguments = split_json_objects(arguments)

        if len(split_arguments) <= 1:
            expanded.append(tool_call)
        else:
            for argument in split_arguments:
                cloned = copy.deepcopy(tool_call)
                cloned['id'] = f'call_{uuid4().hex[:24]}'
                cloned['function']['arguments'] = argument
                expanded.append(cloned)

    return expanded


def get_citation_source_from_tool_result(
    tool_name: str, tool_params: dict, tool_result: str, tool_id: str = ''
) -> list[dict]:
    """
    Parse a tool's result and convert it to source dicts for citation display.

    Follows the source format conventions from get_sources_from_items:
    - source: file/item info object with id, name, type
    - document: list of document contents
    - metadata: list of metadata objects with source, file_id, name fields

    Returns a list of sources (usually one, but query_knowledge_files/query_chat_files may return multiple).
    """
    _EXPECTS_LIST = {'search_web', 'query_knowledge_files', 'query_chat_files'}
    _EXPECTS_DICT = {'view_knowledge_file', 'view_file'}

    try:
        try:
            tool_result = JSONCodec.loads(tool_result)
        except (JSONCodec.JSONDecodeError, TypeError):
            pass  # keep tool_result as-is (e.g. fetch_url returns plain text)
        if isinstance(tool_result, dict) and 'error' in tool_result:
            return []

        # Validate tool_result type based on what the branch expects
        if tool_name in _EXPECTS_LIST and not isinstance(tool_result, list):
            return []
        elif tool_name in _EXPECTS_DICT and not isinstance(tool_result, dict):
            return []

        if tool_name == 'search_web':
            # Parse JSON array: [{"title": "...", "link": "...", "snippet": "..."}]
            results = tool_result
            documents = []
            metadata = []

            for result in results:
                title = result.get('title', '')
                link = result.get('link', '')
                snippet = result.get('snippet', '')

                documents.append(f'{title}\n{snippet}')
                metadata.append(
                    {
                        'source': link,
                        'name': title,
                        'url': link,
                    }
                )

            return [
                {
                    'source': {'name': 'search_web', 'id': 'search_web'},
                    'document': documents,
                    'metadata': metadata,
                }
            ]

        elif tool_name in ('view_knowledge_file', 'view_file'):
            file_data = tool_result
            filename = file_data.get('filename', 'Unknown File')
            file_id = file_data.get('id', '')
            knowledge_name = file_data.get('knowledge_name', '')

            return [
                {
                    'source': {
                        'id': file_id,
                        'name': filename,
                        'type': 'file',
                    },
                    'document': [file_data.get('content', '')],
                    'metadata': [
                        {
                            'file_id': file_id,
                            'name': filename,
                            'source': filename,
                            **({'knowledge_name': knowledge_name} if knowledge_name else {}),
                        }
                    ],
                }
            ]

        elif tool_name == 'fetch_url':
            url = tool_params.get('url', '')
            content = tool_result if isinstance(tool_result, str) else str(tool_result)
            snippet = content[:500] + ('...' if len(content) > 500 else '')

            return [
                {
                    'source': {'name': url or 'fetch_url', 'id': url or 'fetch_url'},
                    'document': [snippet],
                    'metadata': [
                        {
                            'source': url,
                            'name': url,
                            'url': url,
                        }
                    ],
                }
            ]

        elif tool_name in ('query_knowledge_files', 'query_chat_files'):
            chunks = tool_result

            # Group chunks by source for better citation display
            # Each unique source becomes a separate source entry
            sources_by_file = {}

            for chunk in chunks:
                source_name = chunk.get('source', 'Unknown')
                file_id = chunk.get('file_id', '')
                note_id = chunk.get('note_id', '')
                chunk_type = chunk.get('type', 'file')
                content = chunk.get('content', '')

                # Use file_id or note_id as the key
                key = file_id or note_id or source_name

                if key not in sources_by_file:
                    sources_by_file[key] = {
                        'source': {
                            'id': file_id or note_id,
                            'name': source_name,
                            'type': chunk_type,
                        },
                        'document': [],
                        'metadata': [],
                    }

                sources_by_file[key]['document'].append(content)
                sources_by_file[key]['metadata'].append(
                    {
                        'file_id': file_id,
                        'name': source_name,
                        'source': source_name,
                        **({'note_id': note_id} if note_id else {}),
                    }
                )

            # Return all grouped sources as a list
            if sources_by_file:
                return list(sources_by_file.values())

            # Empty result fallback
            return []

        else:
            # Fallback for other tools
            return [
                {
                    'source': {
                        'name': tool_name,
                        'type': 'tool',
                        'id': tool_id or tool_name,
                    },
                    'document': [str(tool_result)],
                    'metadata': [{'source': tool_name, 'name': tool_name}],
                }
            ]
    except Exception as e:
        log.exception(f'Error parsing tool result for {tool_name}: {e}')
        return [
            {
                'source': {'name': tool_name, 'type': 'tool'},
                'document': [str(tool_result)],
                'metadata': [{'source': tool_name}],
            }
        ]


def split_content_and_whitespace(content):
    content_stripped = content.rstrip()
    original_whitespace = content[len(content_stripped) :] if len(content) > len(content_stripped) else ''
    return content_stripped, original_whitespace


def is_opening_code_block(content):
    backtick_segments = content.split('```')
    # Even number of segments means the last backticks are opening a new block
    return len(backtick_segments) > 1 and len(backtick_segments) % 2 == 0


_OPENAI_TOOL_DISPLAY_NAMES = {
    'web_search_call': 'Web Search',
    'file_search_call': 'File Search',
    'computer_call': 'Computer Use',
}


class RetryableStreamError(Exception):
    def __init__(self, error: Any):
        self.error = error
        super().__init__(str(error))


class StreamFatalError(Exception):
    def __init__(self, error: Any):
        self.error = error
        super().__init__(str(error))


class ResponsesStreamState:
    def __init__(self, route_idx: int | None = None, route_url: str | None = None):
        self.started_at = time.monotonic()
        self.last_event_at = None
        self.seen_event = False
        self.completed = False
        self.failed = False
        self.last_event_type = None
        self.response_id = None
        self.last_sequence_number = None
        self.last_output_item_type = None
        self.last_output_item_status = None
        self.route_idx = route_idx
        self.route_url = route_url
        self.is_responses_stream = route_idx is not None or bool(route_url)

    def observe(self, data: dict, output: list):
        event_type = data.get('type', '')
        if not event_type.startswith('response.'):
            return

        now = time.monotonic()
        self.is_responses_stream = True
        self.seen_event = True
        self.last_event_at = now
        self.last_event_type = event_type

        sequence_number = data.get('sequence_number')
        if isinstance(sequence_number, int):
            self.last_sequence_number = sequence_number

        response = data.get('response')
        if isinstance(response, dict) and response.get('id'):
            self.response_id = response.get('id')

        response_id = data.get('response_id')
        if response_id:
            self.response_id = response_id

        item = data.get('item')
        if isinstance(item, dict):
            self.last_output_item_type = item.get('type')
            self.last_output_item_status = item.get('status')
        elif output:
            last_item = output[-1]
            if isinstance(last_item, dict):
                self.last_output_item_type = last_item.get('type')
                self.last_output_item_status = last_item.get('status')

        if event_type == 'response.completed':
            self.completed = True
        elif event_type == 'response.failed':
            self.failed = True

    def incomplete_error(self) -> dict | None:
        if not self.is_responses_stream or self.completed or self.failed:
            return None

        now = time.monotonic()
        last_event_age = None
        if self.last_event_at is not None:
            last_event_age = round(now - self.last_event_at, 3)

        if not self.seen_event:
            return {
                'code': 'stream_empty_eof',
                'message': 'Responses API stream ended before any response event was received.',
                'type': 'upstream_error',
                'last_event_type': None,
                'response_id': self.response_id,
                'last_sequence_number': self.last_sequence_number,
                'response_route_idx': self.route_idx,
                'response_route_url': self.route_url,
                'duration': round(now - self.started_at, 3),
                'idle': last_event_age,
            }

        return {
            'code': 'stream_incomplete_eof',
            'message': 'Responses API stream ended before response.completed.',
            'type': 'upstream_error',
            'last_event_type': self.last_event_type,
            'response_id': self.response_id,
            'last_sequence_number': self.last_sequence_number,
            'response_route_idx': self.route_idx,
            'response_route_url': self.route_url,
            'last_output_item_type': self.last_output_item_type,
            'last_output_item_status': self.last_output_item_status,
            'duration': round(now - self.started_at, 3),
            'idle': last_event_age,
        }


def _responses_stream_error_from_exception(
    error: Exception,
    responses_stream_state: ResponsesStreamState | None,
) -> dict | None:
    if responses_stream_state is not None:
        if responses_stream_state.completed:
            return None

        incomplete_error = responses_stream_state.incomplete_error()
        if incomplete_error:
            incomplete_error['transport_error'] = {
                'code': error.__class__.__name__,
                'message': str(error),
                'type': 'transport_error',
            }
            return incomplete_error

    return {
        'code': error.__class__.__name__,
        'message': str(error),
        'type': 'transport_error',
    }


def _responses_stream_cursor_from_error(
    error: Any,
    responses_stream_state: ResponsesStreamState | None = None,
) -> dict:
    cursor = {}

    if isinstance(error, dict):
        response_id = error.get('response_id')
        sequence_number = error.get('last_sequence_number')
        if sequence_number is None:
            sequence_number = error.get('response_sequence_number')

        values = {
            'response_id': response_id,
            'response_sequence_number': sequence_number,
            'response_route_idx': error.get('response_route_idx'),
            'response_route_url': error.get('response_route_url'),
        }
        cursor.update({key: value for key, value in values.items() if value is not None})

    if responses_stream_state is not None:
        fallback_values = {
            'response_id': responses_stream_state.response_id,
            'response_sequence_number': responses_stream_state.last_sequence_number,
            'response_route_idx': responses_stream_state.route_idx,
            'response_route_url': responses_stream_state.route_url,
        }
        for key, value in fallback_values.items():
            if value is not None and cursor.get(key) is None:
                cursor[key] = value

    return cursor


def _next_response_background_resume_attempt(
    attempts_by_response_id: dict[str, int],
    error: Any,
    max_attempts: int,
) -> int | None:
    if not isinstance(error, dict) or max_attempts <= 0:
        return None

    response_id = error.get('response_id')
    if not response_id:
        return None

    attempt = attempts_by_response_id.get(response_id, 0)
    if attempt >= max_attempts:
        return None

    attempt += 1
    attempts_by_response_id[response_id] = attempt
    return attempt


def _is_retryable_stream_error(error: Any) -> bool:
    if isinstance(error, dict):
        code = str(error.get('code', '')).lower()
        message = str(error.get('message', '')).lower()
        error_type = str(error.get('type', '')).lower()

        if code == 'response_failed' or error.get('last_event_type') == 'response.failed':
            return False

        if code in {'stream_read_error', 'stream_incomplete_eof', 'stream_empty_eof'}:
            return True

        if message in {'stream_read_error', 'stream_incomplete_eof', 'stream_empty_eof'}:
            return True

        if error_type == 'upstream_error' and (
            'stream' in code or 'stream' in message or 'read' in code or 'read' in message
        ):
            return True

        return False

    return 'stream_read_error' in str(error).lower()


def _count_stream_retry_visible_chars(output: list) -> int:
    total = 0

    for item in output or []:
        if not isinstance(item, dict):
            continue

        content = item.get('content')
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get('text')
                    if isinstance(text, str):
                        total += len(text)

        for field in ('code', 'output', 'result', 'text'):
            value = item.get(field)
            if isinstance(value, str):
                total += len(value)

    return total


def _stream_retryable(output: list) -> bool:
    if _count_stream_retry_visible_chars(output) > CHAT_RESPONSE_STREAM_RETRY_VISIBLE_CHAR_LIMIT:
        return False

    blocked_item_types = {
        'open_webui:code_interpreter',
        'computer_call',
    }

    for item in output or []:
        if isinstance(item, dict) and item.get('type') in blocked_item_types:
            return False

    return True


def _output_item_text(item: dict) -> str:
    parts = []

    content = item.get('content')
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                text = part.get('text')
                if isinstance(text, str):
                    parts.append(text)
    elif isinstance(content, str):
        parts.append(content)

    summary = item.get('summary')
    if isinstance(summary, list):
        for part in summary:
            if isinstance(part, dict):
                text = part.get('text')
                if isinstance(text, str):
                    parts.append(text)

    return ''.join(parts)


def _clean_output_for_contextual_retry(items: list) -> list:
    cleaned = []
    function_calls_by_call_id = {}

    for item in items or []:
        if not isinstance(item, dict):
            continue

        if item.get('type') != 'function_call':
            continue

        call_id = item.get('call_id')
        name = item.get('name')
        arguments = item.get('arguments')
        if call_id and name and str(arguments or '').strip():
            function_calls_by_call_id[call_id] = copy.deepcopy(item)

    for item in items or []:
        if not isinstance(item, dict):
            continue

        item_type = item.get('type')
        cloned = copy.deepcopy(item)

        if item_type == 'message':
            if not _output_item_text(cloned).strip():
                continue
            cloned['status'] = 'completed'
            cleaned.append(cloned)
            continue

        if item_type == 'reasoning':
            # Do not replay ordinary reasoning summaries as durable state.
            # If a provider exposes encrypted reasoning carry-over accepted by
            # CPA/OpenAI, add a separate allowlisted branch for that format.
            continue

        if item_type == 'function_call_output':
            call_id = cloned.get('call_id')
            call_item = function_calls_by_call_id.get(call_id)
            tool_name = call_item.get('name') if call_item else None

            if (
                call_id
                and call_item
                and tool_name in RESPONSES_API_CONTEXTUAL_RETRY_TOOL_ALLOWLIST
            ):
                cleaned.append(call_item)
                cleaned.append(cloned)
            continue

        if item_type == 'function_call':
            # Function calls are replayed only together with an allowlisted
            # completed function_call_output in the branch above.
            continue

        if item_type == 'web_search_call':
            if cloned.get('status') == 'completed':
                cleaned.append(cloned)
            continue

        # Fail closed for hosted tools and any unknown item shapes. Only the
        # explicit branches above are safe to replay into contextual retry.

    return cleaned


def _build_stream_resume_instruction(error: Any, retry_output: list) -> dict:
    last_text = ''
    for item in reversed(retry_output or []):
        if isinstance(item, dict) and item.get('type') == 'message':
            last_text = _output_item_text(item).strip()
            if last_text:
                break

    instruction = (
        'The previous streaming response was interrupted before a final '
        'response.completed event. Continue the same task from the context above. '
        'Do not repeat completed tool work. If the previous assistant message '
        'already contains partial prose, continue from the last unfinished point '
        'instead of restarting the answer.'
    )

    if last_text:
        instruction += f'\n\nLast partial assistant text:\n{last_text[-1200:]}'

    if isinstance(error, dict):
        instruction += (
            f"\n\nInterruption detail: {error.get('code') or 'stream_error'}; "
            f"last event: {error.get('last_event_type') or 'unknown'}."
        )

    return {'role': 'user', 'content': instruction}


def _format_openai_web_search_sources(sources: list) -> str:
    lines = []

    for idx, source in enumerate(sources, start=1):
        if isinstance(source, dict):
            url = source.get('url') or source.get('source') or ''
            title = (
                source.get('title')
                or source.get('name')
                or source.get('type')
                or url
                or f'Source {idx}'
            )

            if url:
                lines.append(f'{idx}. {title}\n   {url}')
            else:
                details = json.dumps(source, ensure_ascii=False)
                lines.append(f'{idx}. {title}\n   {details}')
        else:
            lines.append(f'{idx}. {source}')

    return '\n'.join(lines)


def _render_openai_tool_call_handler(item: dict, done: bool) -> str:
    """Render an OpenAI Responses API server-side tool item as a <details> block.

    Handles web_search_call, file_search_call, and computer_call items whose
    schemas are defined in the openai-python SDK (generated from OpenAPI spec).
    """
    item_type = item.get('type', '')
    call_id = item.get('id', '')
    display_name = _OPENAI_TOOL_DISPLAY_NAMES.get(item_type, item_type)

    # Build a short summary of what the tool did
    summary = ''
    arguments = ''
    if item_type == 'web_search_call':
        action = item.get('action', {})
        if isinstance(action, dict):
            arguments = json.dumps(
                {k: v for k, v in action.items() if k != 'sources'},
                ensure_ascii=False,
            )
            atype = action.get('type', '')
            if atype == 'search':
                queries = action.get('queries') or []
                query = action.get('query', '')
                summary = (
                    f'Search: {", ".join(str(q) for q in queries)}'
                    if queries
                    else (f'Search: {query}' if query else '')
                )
            elif atype == 'open_page':
                summary = f'Open page: {action.get("url", "")}' if action.get('url') else ''
            elif atype == 'find_in_page':
                summary = f'Find in page: {action.get("pattern", "")}' if action.get('pattern') else ''

            sources = action.get('sources') or []
            if isinstance(sources, list) and sources:
                sources_summary = _format_openai_web_search_sources(sources)
                summary = (
                    f'{summary}\n\nSources:\n{sources_summary}'
                    if summary
                    else f'Sources:\n{sources_summary}'
                )
    elif item_type == 'file_search_call':
        queries = item.get('queries', [])
        if queries:
            arguments = json.dumps({'queries': queries}, ensure_ascii=False)
        if queries:
            summary = f'Queries: {", ".join(str(q) for q in queries)}'
    elif item_type == 'computer_call':
        action = item.get('action')
        actions = item.get('actions')
        if isinstance(action, dict):
            arguments = json.dumps(action, ensure_ascii=False)
            summary = f'Action: {action.get("type", "unknown")}'
        elif isinstance(actions, list) and actions:
            arguments = json.dumps({'actions': actions}, ensure_ascii=False)
            summary = f'Actions: {", ".join(a.get("type", "?") for a in actions if isinstance(a, dict))}'

    escaped_name = html.escape(display_name)
    escaped_arguments = html.escape(arguments)
    if done:
        return f'<details type="tool_calls" done="true" id="{call_id}" name="{escaped_name}" arguments="{escaped_arguments}">\n<summary>Tool Executed</summary>\n{html.escape(summary)}\n</details>\n'
    return f'<details type="tool_calls" done="false" id="{call_id}" name="{escaped_name}" arguments="{escaped_arguments}">\n<summary>Executing...</summary>\n</details>\n'


def serialize_output(output: list) -> str:
    """
    Convert OR-aligned output items to HTML for display.
    For LLM consumption, use convert_output_to_messages() instead.
    """
    parts: list[str] = []

    # First pass: collect function_call_output items by call_id for lookup
    tool_outputs = {}
    for item in output:
        if item.get('type') == 'function_call_output':
            tool_outputs[item.get('call_id')] = item

    # Second pass: render items in order
    for idx, item in enumerate(output):
        item_type = item.get('type', '')

        if item_type == 'message':
            for content_part in item.get('content', []):
                if 'text' in content_part:
                    text = content_part.get('text', '').strip()
                    if text:
                        parts.append(text)

        elif item_type == 'function_call':
            call_id = item.get('call_id', '')
            name = item.get('name', '')
            arguments = item.get('arguments', '')

            result_item = tool_outputs.get(call_id)
            if result_item:
                result_parts: list[str] = []
                for result_output in result_item.get('output', []):
                    if 'text' in result_output:
                        output_text = result_output.get('text', '')
                        result_parts.append(str(output_text) if not isinstance(output_text, str) else output_text)
                result_text = ''.join(result_parts)
                files = result_item.get('files')
                embeds = result_item.get('embeds', '')

                parts.append(
                    f'<details type="tool_calls" done="true" id="{call_id}" name="{name}" arguments="{html.escape(json.dumps(arguments))}" files="{html.escape(json.dumps(files)) if files else ""}" embeds="{html.escape(json.dumps(embeds))}">\n<summary>Tool Executed</summary>\n{html.escape(json.dumps(result_text, ensure_ascii=False))}\n</details>'
                )
            else:
                parts.append(
                    f'<details type="tool_calls" done="false" id="{call_id}" name="{name}" arguments="{html.escape(json.dumps(arguments))}">\n<summary>Executing...</summary>\n</details>'
                )

        elif item_type == 'function_call_output':
            # Already handled inline with function_call above
            pass

        elif item_type in _OPENAI_TOOL_DISPLAY_NAMES:
            status = item.get('status', 'in_progress')
            done = status in ('completed', 'failed', 'incomplete') or idx != len(output) - 1
            parts.append(_render_openai_tool_call_handler(item, done).rstrip('\n'))

        elif item_type == 'reasoning':
            def collect_reasoning_text(parts: list) -> str:
                reasoning_parts = []
                for content_part in parts or []:
                    text = content_part.get('text', '')
                    if text and text.strip():
                        reasoning_parts.append(text.strip())
                return '\n\n'.join(reasoning_parts).strip()

            summary_reasoning = collect_reasoning_text(item.get('summary', []))
            body_reasoning = collect_reasoning_text(item.get('content', []))
            reasoning_content = summary_reasoning or body_reasoning
            has_visible_reasoning = bool(reasoning_content)

            duration = item.get('duration')
            status = item.get('status', 'in_progress')

            is_last_item = idx == len(output) - 1
            should_render_completed = has_visible_reasoning and (
                status == 'completed' or duration is not None or not is_last_item
            )
            should_render_placeholder = status != 'completed'

            if not should_render_completed and not should_render_placeholder:
                continue

            display = html.escape(
                '\n'.join(
                    (f'> {line}' if not line.startswith('>') else line) for line in reasoning_content.splitlines()
                )
            )

            if should_render_completed:
                summary_text = (
                    'Thought for less than a second'
                    if duration is not None and duration < 1
                    else f'Thought for {duration or 0} seconds'
                )
                parts.append(
                    f'<details type="reasoning" done="true" duration="{duration or 0}">\n<summary>{summary_text}</summary>\n{display}\n</details>'
                )
            else:
                parts.append(
                    f'<details type="reasoning" done="false">\n<summary>Thinking…</summary>\n{display}\n</details>'
                )

        elif item_type == 'open_webui:code_interpreter':
            # Code interpreter needs to inspect/mutate prior accumulated content
            # to strip trailing unclosed code fences — materialize only here.
            content = '\n'.join(parts)
            content_stripped, original_whitespace = split_content_and_whitespace(content)
            if is_opening_code_block(content_stripped):
                content = content_stripped.rstrip('`').rstrip() + original_whitespace
            else:
                content = content_stripped + original_whitespace

            # Re-split back into parts list after mutation
            parts = [content] if content else []

            # Render the code_interpreter item as a <details> block
            # so the frontend Collapsible renders "Analyzing..."/"Analyzed".
            code = item.get('code', '').strip()
            lang = item.get('lang', 'python')
            status = item.get('status', 'in_progress')
            duration = item.get('duration')
            is_last_item = idx == len(output) - 1

            # Build inner content: code block
            display = ''
            if code:
                display = f'```{lang}\n{code}\n```'

            # Build output attribute as HTML-escaped JSON for CodeBlock.svelte
            ci_output = item.get('output')
            output_attr = ''
            if ci_output:
                if isinstance(ci_output, dict):
                    output_json = json.dumps(ci_output, ensure_ascii=False)
                else:
                    output_json = json.dumps({'result': str(ci_output)}, ensure_ascii=False)
                output_attr = f' output="{html.escape(output_json)}"'

            if status == 'completed' or duration is not None or not is_last_item:
                parts.append(
                    f'<details type="code_interpreter" done="true" duration="{duration or 0}"{output_attr}>\n<summary>Analyzed</summary>\n{display}\n</details>'
                )
            else:
                parts.append(
                    f'<details type="code_interpreter" done="false"{output_attr}>\n<summary>Analyzing…</summary>\n{display}\n</details>'
                )

    return '\n'.join(parts).strip()


def normalize_responses_output_item(item: dict, previous_item: dict | None = None) -> dict:
    normalized_item = copy.deepcopy(item)

    if normalized_item.get("type") != "reasoning":
        return normalized_item

    normalized_item.pop("encrypted_content", None)
    normalized_item.setdefault("status", "in_progress")

    started_at = None
    if previous_item and isinstance(previous_item, dict):
        started_at = previous_item.get("started_at")
        if started_at is not None:
            normalized_item.setdefault("started_at", started_at)

        if previous_item.get("summary") and not normalized_item.get("summary"):
            normalized_item["summary"] = copy.deepcopy(previous_item["summary"])

        if previous_item.get("content") and not normalized_item.get("content"):
            normalized_item["content"] = copy.deepcopy(previous_item["content"])

    normalized_item.setdefault("started_at", time.time())
    return keep_reasoning_item_in_progress(normalized_item)


def is_transient_reasoning_placeholder(item: dict | None) -> bool:
    return (
        isinstance(item, dict)
        and item.get("type") == "reasoning"
        and item.get("_placeholder") is True
    )


def build_responses_reasoning_placeholder() -> dict:
    return {
        "type": "reasoning",
        "id": output_id("rs"),
        "status": "in_progress",
        "summary": [],
        "content": [],
        "started_at": time.time(),
        "_placeholder": True,
    }


def keep_reasoning_item_in_progress(item: dict) -> dict:
    if not isinstance(item, dict) or item.get("type") != "reasoning":
        return item

    item.pop("encrypted_content", None)
    item["status"] = "in_progress"
    item.pop("ended_at", None)
    item.pop("duration", None)
    return item


def finalize_output_items(output: list, ended_at: float | None = None) -> list:
    completed_at = ended_at or time.time()

    for item in output:
        if not isinstance(item, dict):
            continue

        item.pop("_placeholder", None)

        if item.get("type") == "reasoning":
            item.pop("encrypted_content", None)
            item["status"] = "completed"

            started_at = item.get("started_at")
            item["ended_at"] = item.get("ended_at", completed_at)
            item["duration"] = (
                max(0, int(item["ended_at"] - started_at))
                if started_at is not None
                else 0
            )
        elif item.get("status") == "in_progress":
            item["status"] = "completed"

    return output


def merge_responses_output_runtime_metadata(
    final_output: list,
    current_output: list,
) -> list:
    """Preserve local streaming metadata when Responses final output arrives.

    The Responses API final response.output contains the authoritative item
    content, but it does not include Open WebUI runtime fields such as
    started_at. Without merging those fields back in, completed reasoning
    blocks render as a zero-second thought even when the stream ran for minutes.
    """
    previous_by_id = {
        item.get("id"): item
        for item in current_output or []
        if isinstance(item, dict) and item.get("id")
    }

    merged_output = []
    for idx, item in enumerate(final_output or []):
        if not isinstance(item, dict):
            merged_output.append(item)
            continue

        merged_item = copy.deepcopy(item)
        previous_item = previous_by_id.get(item.get("id"))
        if previous_item is None and idx < len(current_output or []):
            candidate = current_output[idx]
            if isinstance(candidate, dict) and candidate.get("type") == item.get("type"):
                previous_item = candidate

        if isinstance(previous_item, dict):
            for key in ("started_at", "ended_at", "duration"):
                if merged_item.get(key) is None and previous_item.get(key) is not None:
                    merged_item[key] = previous_item[key]

            if merged_item.get("type") == "reasoning":
                for key in ("summary", "content"):
                    if not merged_item.get(key) and previous_item.get(key):
                        merged_item[key] = copy.deepcopy(previous_item[key])

        merged_output.append(merged_item)

    return merged_output


RESPONSES_TOOL_EVENT_TYPES = {
    'web_search_call',
    'file_search_call',
    'computer_call',
    'code_interpreter_call',
    'image_generation_call',
}


def handle_responses_tool_status_event(
    data: dict,
    current_output: list,
) -> tuple[list, dict | None]:
    event_type = data.get('type', '')
    event_parts = event_type.split('.')
    if len(event_parts) != 3:
        return current_output, None

    tool_type = event_parts[1]
    status = event_parts[2]
    if tool_type not in RESPONSES_TOOL_EVENT_TYPES:
        return current_output, None

    output_index = data.get('output_index')
    item_id = data.get('item_id')

    target_index = None
    if isinstance(output_index, int) and 0 <= output_index < len(current_output):
        target_index = output_index
    elif item_id:
        for idx, item in enumerate(current_output):
            if isinstance(item, dict) and item.get('id') == item_id:
                target_index = idx
                break

    status_map = {
        'in_progress': 'in_progress',
        'searching': 'in_progress',
        'completed': 'completed',
        'failed': 'failed',
        'incomplete': 'incomplete',
    }
    item_status = status_map.get(status)
    if item_status is None:
        return current_output, None

    new_output = list(current_output)
    incoming_item = data.get('item') if isinstance(data.get('item'), dict) else None

    if target_index is not None:
        item = new_output[target_index].copy()
        if incoming_item:
            item = deep_merge(item, incoming_item)
        item.setdefault('type', tool_type)
        if item_id:
            item.setdefault('id', item_id)
        item['status'] = item_status
        new_output[target_index] = item
    elif item_id:
        item = {
            'type': tool_type,
            'id': item_id,
            'status': item_status,
        }
        if incoming_item:
            item = deep_merge(item, incoming_item)
        new_output.append(item)
    else:
        return current_output, None

    return new_output, {}


def build_responses_web_search_status(item: dict) -> dict | None:
    if not isinstance(item, dict) or item.get('type') != 'web_search_call':
        return None

    action = item.get('action') if isinstance(item.get('action'), dict) else {}
    status = item.get('status', 'in_progress')
    done = status in ('completed', 'failed', 'incomplete')

    sources = action.get('sources') if isinstance(action.get('sources'), list) else []
    urls = []
    items = []

    def add_source(source):
        if isinstance(source, str):
            if source and source not in urls:
                urls.append(source)
            return

        if not isinstance(source, dict):
            return

        url = source.get('url') or source.get('source') or source.get('link') or ''
        title = source.get('title') or source.get('name') or url
        if not url:
            return

        if title and title != url:
            item_entry = {'link': url, 'title': title}
            if item_entry not in items:
                items.append(item_entry)
        elif url not in urls:
            urls.append(url)

    for source in sources:
        add_source(source)

    page_url = action.get('url')
    if page_url:
        add_source({'url': page_url, 'title': action.get('title') or page_url})

    queries = action.get('queries') if isinstance(action.get('queries'), list) else []
    query = action.get('query') or (str(queries[0]) if queries else '') or page_url or ''

    action_type = action.get('type', '')
    if status == 'failed':
        description = 'Web search failed'
    elif done and (items or urls):
        description = 'Searched {{count}} sites'
    elif query:
        description = 'Searching "{{searchQuery}}"'
    elif action_type == 'open_page':
        description = 'Opening web page'
    elif action_type == 'find_in_page':
        description = 'Searching within web page'
    else:
        description = 'Searching the web'

    status_data = {
        'action': 'web_search',
        'description': description,
        'done': done,
    }

    if item.get('id'):
        status_data['id'] = item.get('id')
    if query:
        status_data['query'] = query
    if items:
        status_data['items'] = items
    elif urls:
        status_data['urls'] = urls

    return status_data


def deep_merge(target, source):
    """
    Merge source into target recursively (returning new structure).
    - Dicts: Recursive merge.
    - Strings: Concatenation.
    - Others: Overwrite.
    """
    if isinstance(target, dict) and isinstance(source, dict):
        new_target = target.copy()
        for k, v in source.items():
            if k in new_target:
                new_target[k] = deep_merge(new_target[k], v)
            else:
                new_target[k] = v
        return new_target
    elif isinstance(target, str) and isinstance(source, str):
        return target + source
    else:
        return source


def extract_first_json_object(text: str) -> Optional[dict]:
    if not isinstance(text, str):
        return None

    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue

        try:
            value, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue

        if isinstance(value, dict):
            return value

    return None


RESPONSE_COMPLETION_RESPONSE_FIELDS = ('error', 'id', 'output', 'usage')


def get_response_completion_event_data(event: dict) -> dict:
    """Build the data payload for response:completion events."""
    response = event.get('response')
    if not isinstance(response, dict):
        return event

    response_data = {key: response[key] for key in RESPONSE_COMPLETION_RESPONSE_FIELDS if key in response}

    return {
        **event,
        'response': response_data,
    }


def handle_responses_streaming_event(
    data: dict,
    current_output: list,
) -> tuple[list, dict | None]:
    """
    Handle Responses API streaming events in a pure functional way.

    Args:
        data: The event data
        current_output: List of output items (treated as immutable)

    Returns:
        tuple[list, dict | None]: (new_output, metadata)
        - new_output: The updated output list.
        - metadata: Metadata to emit (e.g. usage), {} if update occurred, None if skip.
    """
    # Default: no change
    # Note: treating current_output as immutable, but avoiding full deepcopy for perf.
    # We will shallow copy only if we need to modify the list structure or items.

    event_type = data.get('type', '')

    if event_type == 'response.output_item.added':
        item = data.get('item', {})
        output_index = data.get('output_index', len(current_output))
        if item:
            new_output = list(current_output)
            output_index = data.get('output_index', len(new_output))
            existing_index = next(
                (
                    idx
                    for idx, existing in enumerate(new_output)
                    if (item.get('id') and existing.get('id') == item.get('id'))
                    or (item.get('call_id') and existing.get('call_id') == item.get('call_id'))
                ),
                None,
            )
            if existing_index is not None:
                new_output[existing_index] = normalize_responses_output_item(item, new_output[existing_index])
            elif isinstance(output_index, int) and 0 <= output_index < len(new_output):
                if is_transient_reasoning_placeholder(new_output[output_index]):
                    new_output[output_index] = normalize_responses_output_item(item, new_output[output_index])
                else:
                    new_output.insert(output_index, normalize_responses_output_item(item))
            else:
                new_output.append(normalize_responses_output_item(item))
            return new_output, None
        return current_output, None

    elif event_type == 'response.content_part.added':
        part = data.get('part', {})
        output_index = data.get('output_index', len(current_output) - 1)

        if current_output and 0 <= output_index < len(current_output):
            new_output = list(current_output)
            # Copy the item to mutate it
            item = new_output[output_index].copy()
            new_output[output_index] = item

            if 'content' not in item:
                item['content'] = []
            else:
                # Copy content list
                item['content'] = list(item['content'])

            if item.get('type') == 'reasoning':
                # Reasoning items should not have content parts
                pass
            else:
                item['content'].append(part)
            return new_output, None
        return current_output, None

    elif event_type == 'response.reasoning_summary_part.added':
        part = data.get('part', {})
        output_index = data.get('output_index', len(current_output) - 1)

        if current_output and 0 <= output_index < len(current_output):
            new_output = list(current_output)
            item = new_output[output_index].copy()
            new_output[output_index] = item

            if 'summary' not in item:
                item['summary'] = []
            else:
                item['summary'] = list(item['summary'])

            item['summary'].append(part)
            return new_output, None
        return current_output, None

    elif event_type.startswith('response.') and event_type.endswith('.delta'):
        # Generic Delta Handling
        parts = event_type.split('.')
        if len(parts) >= 3:
            delta_type = parts[1]
            delta = data.get('delta', '')

            output_index = data.get('output_index', len(current_output) - 1)

            if current_output and 0 <= output_index < len(current_output):
                new_output = list(current_output)
                item = new_output[output_index].copy()
                new_output[output_index] = item
                item_type = item.get('type', '')

                # Determine target field and object based on delta_type and item_type
                if delta_type == 'function_call_arguments':
                    key = 'arguments'
                    if item_type == 'function_call':
                        # Function call args are usually strings
                        item[key] = item.get(key, '') + str(delta)
                else:
                    # Generic handling, refined by item type below
                    pass

                    if item_type == 'message':
                        # Message items: "text"/"output_text" -> "text"
                        # "reasoning_text" -> Skipped (should use reasoning item)
                        if delta_type in ['text', 'output_text']:
                            key = 'text'
                        elif delta_type in ['reasoning_text', 'reasoning_summary_text']:
                            # Skip reasoning updates for message items
                            return new_output, None
                        else:
                            key = delta_type

                        content_index = data.get('content_index', 0)
                        if 'content' not in item:
                            item['content'] = []
                        else:
                            item['content'] = list(item['content'])
                        content_list = item['content']

                        while len(content_list) <= content_index:
                            content_list.append({'type': 'text', 'text': ''})

                        # Copy the part to mutate it
                        part = content_list[content_index].copy()
                        content_list[content_index] = part

                        current_val = part.get(key)
                        if current_val is None:
                            # Initialize based on delta type
                            current_val = {} if isinstance(delta, dict) else ''

                        part[key] = deep_merge(current_val, delta)

                    elif item_type == 'reasoning':
                        # Reasoning items: "reasoning_text"/"reasoning_summary_text" -> "text"
                        # "text"/"output_text" -> Skipped (should use message item)
                        if delta_type == 'reasoning_summary_text':
                            # Summary updates -> item['summary']
                            key = 'text'
                            summary_index = data.get('summary_index', 0)
                            if 'summary' not in item:
                                item['summary'] = []
                            else:
                                item['summary'] = list(item['summary'])
                            summary_list = item['summary']

                            while len(summary_list) <= summary_index:
                                summary_list.append({'type': 'summary_text', 'text': ''})

                            part = summary_list[summary_index].copy()
                            summary_list[summary_index] = part

                            target_val = part.get(key, '')
                            part[key] = deep_merge(target_val, delta)

                        elif delta_type == 'reasoning_text':
                            # Reasoning body updates -> item['content']
                            key = 'text'
                            content_index = data.get('content_index', 0)
                            if 'content' not in item:
                                item['content'] = []
                            else:
                                item['content'] = list(item['content'])
                            content_list = item['content']

                            while len(content_list) <= content_index:
                                # Reasoning content parts default to text
                                content_list.append({'type': 'text', 'text': ''})

                            part = content_list[content_index].copy()
                            content_list[content_index] = part

                            target_val = part.get(key, '')
                            part[key] = deep_merge(target_val, delta)

                        elif delta_type in ['text', 'output_text']:
                            return new_output, None
                        else:
                            # Fallback just in case other deltas target reasoning?
                            pass

                    else:
                        # Fallback for other item types
                        if delta_type in ['text', 'output_text']:
                            key = 'text'
                        else:
                            key = delta_type

                        current_val = item.get(key)
                        if current_val is None:
                            current_val = {} if isinstance(delta, dict) else ''
                        item[key] = deep_merge(current_val, delta)

                return new_output, None

        return current_output, None

    elif (
        event_type.startswith('response.')
        and event_type.endswith('.done')
        and event_type != 'response.output_item.done'
    ):
        # Delta Events: response.content_part.done, response.text.done, etc.
        parts = event_type.split('.')
        if len(parts) >= 3:
            type_name = parts[1]

            # 1. Handle specific Delta "done" signals
            if type_name == 'content_part':
                # "Signaling that no further changes will occur to a content part"
                # If payloads contains the full part, we could update it.
                # Usually purely signaling in standard implementation, but we check payload.
                part = data.get('part')
                output_index = data.get('output_index', len(current_output) - 1)

                if part and current_output and 0 <= output_index < len(current_output):
                    new_output = list(current_output)
                    item = new_output[output_index].copy()
                    new_output[output_index] = item

                    if 'content' in item:
                        item['content'] = list(item['content'])
                        content_index = data.get('content_index', len(item['content']) - 1)
                        if 0 <= content_index < len(item['content']):
                            item['content'][content_index] = part
                            return new_output, {}
                return current_output, None

            elif type_name == 'reasoning_summary_part':
                part = data.get('part')
                output_index = data.get('output_index', len(current_output) - 1)

                if part and current_output and 0 <= output_index < len(current_output):
                    new_output = list(current_output)
                    item = new_output[output_index].copy()
                    new_output[output_index] = item

                    if 'summary' in item:
                        item['summary'] = list(item['summary'])
                        summary_index = data.get('summary_index', len(item['summary']) - 1)
                        if 0 <= summary_index < len(item['summary']):
                            item['summary'][summary_index] = part
                            return new_output, {}
                return current_output, None

            # 2. Generic Field Done (text.done, audio.done)
            if type_name not in ['completed', 'failed']:
                output_index = data.get('output_index', len(current_output) - 1)
                if current_output and 0 <= output_index < len(current_output):
                    key = (
                        'text'
                        if type_name
                        in [
                            'text',
                            'output_text',
                            'reasoning_text',
                            'reasoning_summary_text',
                        ]
                        else type_name
                    )
                    if type_name == 'function_call_arguments':
                        key = 'arguments'

                    if key in data:
                        final_value = data[key]
                        new_output = list(current_output)
                        item = new_output[output_index].copy()
                        new_output[output_index] = item
                        item_type = item.get('type', '')

                        if type_name == 'function_call_arguments':
                            if item_type == 'function_call':
                                item['arguments'] = final_value
                        elif item_type == 'message':
                            content_index = data.get('content_index', 0)
                            if 'content' in item:
                                item['content'] = list(item['content'])
                                if len(item['content']) > content_index:
                                    part = item['content'][content_index].copy()
                                    item['content'][content_index] = part
                                    part[key] = final_value
                        elif item_type == 'reasoning':
                            item['status'] = 'completed'
                        else:
                            item[key] = final_value

                        return new_output, {}

        return current_output, None

    elif event_type == 'response.output_item.done':
        # Delta Event: Output item complete
        item = data.get('item')
        output_index = data.get('output_index', len(current_output) - 1)

        new_output = list(current_output)
        if item and 0 <= output_index < len(current_output):
            normalized_item = normalize_responses_output_item(
                item, current_output[output_index]
            )
            if normalized_item.get("type") == "reasoning":
                normalized_item = keep_reasoning_item_in_progress(normalized_item)
            new_output[output_index] = normalized_item
        elif item:
            normalized_item = normalize_responses_output_item(item)
            if normalized_item.get("type") == "reasoning":
                normalized_item = keep_reasoning_item_in_progress(normalized_item)
            new_output.append(normalized_item)
        return new_output, {}
    elif event_type == 'response.completed':
        # State Machine Event: Completed
        response_data = data.get('response', {})
        final_output = response_data.get('output')

        if isinstance(final_output, list) and final_output:
            new_output = merge_responses_output_runtime_metadata(final_output, current_output)
        else:
            new_output = copy.deepcopy(current_output)

        new_output = finalize_output_items(new_output)

        return new_output, {
            'usage': response_data.get('usage'),
            'done': True,
            'response_id': response_data.get('id'),
        }

    elif event_type == 'response.in_progress':
        # State Machine Event: In Progress
        # We could extract metadata if needed, but for now just acknowledge iteration
        return current_output, None

    elif event_type.startswith('response.') and any(
        event_type.startswith(f'response.{tool_type}.') for tool_type in RESPONSES_TOOL_EVENT_TYPES
    ):
        return handle_responses_tool_status_event(data, current_output)

    elif event_type == 'response.failed':
        # State Machine Event: Failed
        response_data = data.get('response') if isinstance(data.get('response'), dict) else {}
        error = response_data.get('error') or data.get('error')
        if not error:
            error = {
                'code': 'response_failed',
                'message': 'Responses API emitted response.failed.',
                'type': 'upstream_error',
            }
        if isinstance(error, dict):
            error.setdefault('last_event_type', 'response.failed')
            response_id = response_data.get('id') or data.get('response_id')
            if response_id:
                error.setdefault('response_id', response_id)
            sequence_number = data.get('sequence_number')
            if isinstance(sequence_number, int):
                error.setdefault('last_sequence_number', sequence_number)
        return current_output, {'error': error}

    else:
        return current_output, None


def get_source_context(sources: list, source_ids: dict = None, include_content: bool = True) -> str:
    """
    Build <source> tag context string from citation sources.
    """
    context_string = ''
    if source_ids is None:
        source_ids = {}
    for source in sources:
        for doc, meta in zip(source.get('document', []), source.get('metadata', [])):
            source_id = meta.get('source') or source.get('source', {}).get('id') or 'N/A'
            if source_id not in source_ids:
                source_ids[source_id] = len(source_ids) + 1
            src_name = source.get('source', {}).get('name')
            src_type = source.get('source', {}).get('type')
            src_rid = source.get('source', {}).get('id')
            body = doc if include_content else ''
            context_string += (
                f'<source id="{source_ids[source_id]}"'
                + (f' name="{src_name}"' if src_name else '')
                + (f' resource-type="{src_type}"' if src_type else '')
                + (f' resource-id="{src_rid}"' if src_rid else '')
                + f'>{body}</source>\n'
            )
    return context_string


async def apply_source_context_to_messages(
    request: Request,
    messages: list,
    sources: list,
    user_message: str,
    include_content: bool = True,
) -> list:
    """
    Build source context from citation sources and apply to messages.
    Uses RAG template to format context for model consumption.

    When include_content is False, emit <source> tags with id/name but no
    document body — useful when the content is already present elsewhere
    (e.g. in a tool result message) and only citation markers are needed.
    """
    if not sources or not user_message:
        return messages

    context = get_source_context(sources, include_content=include_content)

    context = context.strip()
    if not context:
        return messages

    if RAG_SYSTEM_CONTEXT:
        return add_or_update_system_message(
            await rag_template(await Config.get('rag.template'), context, user_message),
            messages,
            append=True,
        )
    else:
        return add_or_update_user_message(
            await rag_template(await Config.get('rag.template'), context, user_message),
            messages,
            append=False,
        )


async def process_tool_result(
    request,
    tool_function_name,
    tool_result,
    tool_type,
    direct_tool=False,
    metadata=None,
    user=None,
):
    tool_result_embeds = []
    EXTERNAL_TOOL_TYPES = ('external', 'action', 'terminal')

    # Support (HTMLResponse, result_context) tuples: the optional second
    # element lets tool authors provide the LLM with actionable context
    # about the generated embed instead of the generic fallback message.
    result_context = None
    if isinstance(tool_result, tuple) and len(tool_result) == 2 and isinstance(tool_result[0], HTMLResponse):
        tool_result, result_context = tool_result

    if isinstance(tool_result, HTMLResponse):
        content_disposition = tool_result.headers.get('Content-Disposition', '')
        if 'inline' in content_disposition:
            content = tool_result.body.decode('utf-8', 'replace')
            tool_result_embeds.append(content)

            if 200 <= tool_result.status_code < 300:
                if result_context is not None and isinstance(result_context, (str, dict, list)):
                    tool_result = result_context
                else:
                    tool_result = {
                        'status': 'success',
                        'code': 'ui_component',
                        'message': f'{tool_function_name}: Embedded UI result is active and visible to the user.',
                    }
            elif 400 <= tool_result.status_code < 500:
                tool_result = {
                    'status': 'error',
                    'code': 'ui_component',
                    'message': f'{tool_function_name}: Client error {tool_result.status_code} from embedded UI result.',
                }
            elif 500 <= tool_result.status_code < 600:
                tool_result = {
                    'status': 'error',
                    'code': 'ui_component',
                    'message': f'{tool_function_name}: Server error {tool_result.status_code} from embedded UI result.',
                }
            else:
                tool_result = {
                    'status': 'error',
                    'code': 'ui_component',
                    'message': f'{tool_function_name}: Unexpected status code {tool_result.status_code} from embedded UI result.',
                }
        else:
            tool_result = tool_result.body.decode('utf-8', 'replace')

    elif (tool_type in EXTERNAL_TOOL_TYPES and isinstance(tool_result, tuple)) or (
        direct_tool and isinstance(tool_result, list) and len(tool_result) == 2
    ):
        tool_result, tool_response_headers = tool_result

        try:
            if not isinstance(tool_response_headers, dict):
                tool_response_headers = dict(tool_response_headers)
        except Exception as e:
            tool_response_headers = {}
            log.debug(e)

        if tool_response_headers and isinstance(tool_response_headers, dict):
            content_disposition = tool_response_headers.get(
                'Content-Disposition',
                tool_response_headers.get('content-disposition', ''),
            )

            if 'inline' in content_disposition:
                content_type = tool_response_headers.get(
                    'Content-Type',
                    tool_response_headers.get('content-type', ''),
                )
                location = tool_response_headers.get(
                    'Location',
                    tool_response_headers.get('location', ''),
                )

                if 'text/html' in content_type:
                    # Support (html_content, result_context) nested tuple
                    result_context = None
                    html_content = tool_result
                    if isinstance(tool_result, (tuple, list)) and len(tool_result) == 2:
                        html_content, result_context = tool_result

                    # Display as iframe embed
                    tool_result_embeds.append(html_content)
                    if result_context is not None and isinstance(result_context, (str, dict, list)):
                        tool_result = result_context
                    else:
                        tool_result = {
                            'status': 'success',
                            'code': 'ui_component',
                            'message': f'{tool_function_name}: Embedded UI result is active and visible to the user.',
                        }
                elif location:
                    # Support (html_content, result_context) nested tuple for location embeds
                    result_context = None
                    if isinstance(tool_result, (tuple, list)) and len(tool_result) == 2:
                        _, result_context = tool_result

                    tool_result_embeds.append(location)
                    if result_context is not None and isinstance(result_context, (str, dict, list)):
                        tool_result = result_context
                    else:
                        tool_result = {
                            'status': 'success',
                            'code': 'ui_component',
                            'message': f'{tool_function_name}: Embedded UI result is active and visible to the user.',
                        }

    tool_result_files = []

    # Detect base64 image data URIs from tool results (e.g. binary image
    # responses from execute_tool_server).  Move the data URI to
    # tool_result_files and replace tool_result with a text summary.
    if isinstance(tool_result, str) and tool_result.startswith('data:image/'):
        tool_result_files.append({'type': 'image', 'url': tool_result})
        tool_result = f'{tool_function_name}: Image file read successfully.'

    if isinstance(tool_result, list):
        if tool_type == 'mcp':  # MCP
            tool_response = []
            for item in tool_result:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        text = item.get('text', '')
                        if isinstance(text, str):
                            try:
                                text = JSONCodec.loads(text)
                            except JSONCodec.JSONDecodeError:
                                pass
                        tool_response.append(text)
                    elif item.get('type') in ['image', 'audio']:
                        file_url = await get_file_url_from_base64(
                            request,
                            f'data:{item.get("mimeType")};base64,{item.get("data", item.get("blob", ""))}',
                            {
                                'chat_id': metadata.get('chat_id', None),
                                'message_id': metadata.get('message_id', None),
                                'session_id': metadata.get('session_id', None),
                                'result': item,
                            },
                            user,
                        )

                        tool_result_files.append(
                            {
                                'type': item.get('type', 'data'),
                                'url': file_url,
                            }
                        )
                    elif item.get('type') == 'resource':
                        resource = item.get('resource', {})
                        text = resource.get('text', '')
                        if isinstance(text, str) and text:
                            try:
                                text = JSONCodec.loads(text)
                            except JSONCodec.JSONDecodeError:
                                pass
                            tool_response.append(text)
                        elif resource.get('blob'):
                            resource_mime_type = resource.get('mimeType') or 'application/octet-stream'
                            resource_blob = resource.get('blob', '')
                            if resource_mime_type.startswith('image/'):
                                tool_result_files.append(
                                    {
                                        'type': 'image',
                                        'url': f'data:{resource_mime_type};base64,{resource_blob}',
                                    }
                                )
                            else:
                                resource_uri = resource.get('uri', 'resource')
                                tool_response.append(
                                    f'[Resource: {resource_uri}] (binary data, mimeType: {resource_mime_type})'
                                )
                        elif resource.get('uri'):
                            tool_response.append(resource.get('uri'))
            tool_result = tool_response[0] if len(tool_response) == 1 else tool_response
        else:  # OpenAPI
            for item in tool_result:
                if isinstance(item, str) and item.startswith('data:'):
                    tool_result_files.append(
                        {
                            'type': 'data',
                            'content': item,
                        }
                    )
                    tool_result.remove(item)

    if isinstance(tool_result, list):
        tool_result = {'results': tool_result}

    if isinstance(tool_result, dict) or isinstance(tool_result, list):
        tool_result = json.dumps(tool_result, indent=2, ensure_ascii=False)

    # Safety: ensure tool_result is always a string (or None) to prevent
    # downstream TypeError when concatenating (e.g. if an upstream callable
    # returned a tuple that was not unpacked by the branches above).
    if tool_result is not None and not isinstance(tool_result, str):
        if isinstance(tool_result, tuple):
            # execute_tool_server returns (data, headers); unpack the data part
            tool_result = json.dumps(tool_result[0], indent=2, ensure_ascii=False) if len(tool_result) > 0 else ''
        else:
            tool_result = str(tool_result)

    return tool_result, tool_result_files, tool_result_embeds


async def terminal_event_handler(
    tool_function_name: str,
    tool_function_params: dict,
    tool_result,
    event_emitter,
):
    """Emit terminal:* events for Open Terminal tools.

    - display_file  → emits 'terminal:display_file' to open the file preview.
    - write_file / replace_file_content → emits 'terminal:write_file' to refresh.
    - run_command → emits 'terminal:run_command' with cwd to refresh if relevant.
    """
    if not event_emitter:
        return

    if tool_function_name == 'display_file':
        if tool_function_params.get('inline') is True:
            return
        path = tool_function_params.get('path', '')
        if not path:
            return
        # Only emit if the file actually exists
        parsed = tool_result
        if isinstance(parsed, str):
            try:
                parsed = JSONCodec.loads(parsed)
            except (JSONCodec.JSONDecodeError, TypeError):
                pass
        if isinstance(parsed, dict) and parsed.get('exists') is False:
            return
        page = tool_function_params.get('page')

        await event_emitter(
            {
                'type': f'terminal:{tool_function_name}',
                'data': {
                    'path': path,
                    **({'page': page} if page else {}),
                },
            }
        )
    elif tool_function_name in ('write_file', 'replace_file_content'):
        path = tool_function_params.get('path', '')
        if not path:
            return
        await event_emitter(
            {
                'type': f'terminal:{tool_function_name}',
                'data': {'path': path},
            }
        )
    elif tool_function_name == 'run_command':
        await event_emitter(
            {
                'type': 'terminal:run_command',
                'data': {},
            }
        )


async def chat_completion_tools_handler(
    request: Request, body: dict, extra_params: dict, user: UserModel, models, tools
) -> tuple[dict, dict]:
    async def get_content_from_response(response) -> Optional[str]:
        content = None
        if hasattr(response, 'body_iterator'):
            async for chunk in response.body_iterator:
                data = JSONCodec.loads(chunk.decode('utf-8', 'replace'))
                content = data['choices'][0]['message']['content']

            # Cleanup any remaining background tasks if necessary
            if response.background is not None:
                await response.background()
        else:
            content = response['choices'][0]['message']['content']
        return content

    def get_tools_function_calling_payload(messages, task_model_id, content):
        user_message = get_last_user_message(messages)

        if user_message and messages and messages[-1]['role'] == 'user':
            # Remove the last user message to avoid duplication
            messages = messages[:-1]

        recent_messages = messages[-4:] if len(messages) > 4 else messages
        chat_history = '\n'.join(
            f'{message["role"].upper()}: """{get_content_from_message(message)}"""' for message in recent_messages
        )

        prompt = f'History:\n{chat_history}\nQuery: {user_message}' if chat_history else f'Query: {user_message}'

        return {
            'model': task_model_id,
            'messages': [
                {'role': 'system', 'content': content},
                {'role': 'user', 'content': prompt},
            ],
            'stream': False,
            'metadata': {'task': str(TASKS.FUNCTION_CALLING)},
        }

    event_caller = extra_params['__event_call__']
    event_emitter = extra_params['__event_emitter__']
    metadata = extra_params['__metadata__']

    # One batched SELECT instead of four sequential round trips.
    task_config = await Config.get_many(
        'task.model.default',
        'task.model.external',
        'task.tools.prompt_template',
    )
    task_model_id = get_task_model_id(
        body['model'],
        task_config.get('task.model.default'),
        task_config.get('task.model.external'),
        models,
    )

    skip_files = False
    sources = []

    specs = [tool['spec'] for tool in tools.values()]
    tools_specs = JSONCodec.dumps(specs, ensure_ascii=False)

    tools_prompt_template = task_config.get('task.tools.prompt_template')
    if tools_prompt_template != '':
        template = tools_prompt_template
    else:
        template = DEFAULT_TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE

    tools_function_calling_prompt = tools_function_calling_generation_template(template, tools_specs)
    payload = get_tools_function_calling_payload(body['messages'], task_model_id, tools_function_calling_prompt)

    try:
        response = await generate_chat_completion(request, form_data=payload, user=user)
        log.debug('response=%r', response)
        content = await get_content_from_response(response)
        log.debug('content=%r', content)

        if not content:
            return body, {}

        try:
            content = content[content.find('{') : content.rfind('}') + 1]
            if not content:
                raise Exception('No JSON object found in the response')

            result = JSONCodec.loads(content)

            async def tool_call_handler(tool_call):
                nonlocal skip_files

                log.debug('tool_call=%r', tool_call)

                tool_function_name = tool_call.get('name', None)
                if tool_function_name not in tools:
                    log.warning(f'Tool "{tool_function_name}" not found')
                    return

                tool_function_params = tool_call.get('parameters', {})

                tool = None
                tool_type = ''
                direct_tool = False

                try:
                    tool = tools[tool_function_name]
                    tool_type = tool.get('type', '')
                    direct_tool = tool.get('direct', False)

                    spec = tool.get('spec', {})
                    allowed_params = spec.get('parameters', {}).get('properties', {}).keys()
                    tool_function_params = {k: v for k, v in tool_function_params.items() if k in allowed_params}

                    if tool.get('direct', False):
                        tool_result = await event_caller(
                            {
                                'type': 'execute:tool',
                                'data': {
                                    'id': str(uuid4()),
                                    'name': tool_function_name,
                                    'params': tool_function_params,
                                    'server': tool.get('server', {}),
                                    'session_id': metadata.get('session_id', None),
                                },
                            }
                        )
                    else:
                        tool_function = tool['callable']
                        tool_result = await tool_function(**tool_function_params)

                except Exception as e:
                    tool_result = {'error': str(e)}

                tool_result, tool_result_files, tool_result_embeds = await process_tool_result(
                    request,
                    tool_function_name,
                    tool_result,
                    tool_type,
                    direct_tool,
                    metadata,
                    user,
                )

                if event_emitter:
                    await terminal_event_handler(
                        tool_function_name,
                        tool_function_params,
                        tool_result,
                        event_emitter,
                    )

                    if tool_result_files:
                        await event_emitter(
                            {
                                'type': 'files',
                                'data': {
                                    'files': tool_result_files,
                                },
                            }
                        )

                    if tool_result_embeds:
                        await event_emitter(
                            {
                                'type': 'embeds',
                                'data': {
                                    'embeds': tool_result_embeds,
                                },
                            }
                        )

                if tool_result:
                    tool = tools[tool_function_name]
                    tool_id = tool.get('tool_id', '')

                    tool_name = f'{tool_id}/{tool_function_name}' if tool_id else f'{tool_function_name}'

                    # Citation is enabled for this tool
                    sources.append(
                        {
                            'source': {
                                'name': (f'{tool_name}'),
                            },
                            'document': [str(tool_result)],
                            'metadata': [
                                {
                                    'source': (f'{tool_name}'),
                                    'parameters': tool_function_params,
                                }
                            ],
                            'tool_result': True,
                        }
                    )

                    if tools[tool_function_name].get('metadata', {}).get('file_handler', False):
                        skip_files = True

            # check if "tool_calls" in result
            if result.get('tool_calls'):
                for tool_call in result.get('tool_calls'):
                    await tool_call_handler(tool_call)
            else:
                await tool_call_handler(result)

        except Exception as e:
            log.debug('Error: %s', e)
            content = None
    except Exception as e:
        log.debug('Error: %s', e)
        content = None

    log.debug('tool_contexts: %s', sources)

    if skip_files and 'files' in body.get('metadata', {}):
        del body['metadata']['files']

    return body, {'sources': sources}


async def chat_web_search_handler(request: Request, form_data: dict, extra_params: dict, user):
    event_emitter = extra_params['__event_emitter__']
    await event_emitter(
        {
            'type': 'status',
            'data': {
                'action': 'web_search',
                'description': 'Searching the web',
                'done': False,
            },
        }
    )

    messages = form_data['messages']
    user_message = get_last_user_message(messages)

    queries = []
    try:
        res = await generate_queries(
            request,
            {
                'model': form_data['model'],
                'messages': messages,
                'prompt': user_message,
                'type': 'web_search',
                'chat_id': extra_params.get('__chat_id__'),
            },
            user,
        )

        # generate_queries returns a JSONResponse on error (e.g. model not
        # found, chat completion failure).  Extract the error detail and
        # re-raise so the outer except block falls back to using the raw
        # user message as the search query.
        if isinstance(res, JSONResponse):
            try:
                error_body = JSONCodec.loads(res.body)
                detail = error_body.get('detail', 'Query generation failed')
            except Exception:
                detail = 'Query generation failed'
            raise Exception(detail)

        response = res['choices'][0]['message']['content']

        try:
            bracket_start = response.rfind('{')
            bracket_end = response.rfind('}') + 1

            if bracket_start == -1 or bracket_end == -1:
                raise Exception('No JSON object found in the response')

            response = response[bracket_start:bracket_end]
            queries = JSONCodec.loads(response)
            queries = queries.get('queries', [])
        except Exception as e:
            queries = [response]

        if ENABLE_QUERIES_CACHE:
            request.state.cached_queries = queries

    except Exception as e:
        log.exception(e)
        queries = [user_message or '']

    # Check if generated queries are empty
    if len(queries) == 1 and queries[0].strip() == '':
        queries = [user_message or '']

    # Check if queries are not found
    if len(queries) == 0:
        await event_emitter(
            {
                'type': 'status',
                'data': {
                    'action': 'web_search',
                    'description': 'No search query generated',
                    'done': True,
                },
            }
        )
        return form_data

    await event_emitter(
        {
            'type': 'status',
            'data': {
                'action': 'web_search_queries_generated',
                'queries': queries,
                'done': False,
            },
        }
    )

    try:
        results = await process_web_search(
            request,
            SearchForm(queries=queries),
            user=user,
        )

        if results:
            files = form_data.get('files', [])

            if results.get('collection_names'):
                for col_idx, collection_name in enumerate(results.get('collection_names')):
                    files.append(
                        {
                            'collection_name': collection_name,
                            'name': ', '.join(queries),
                            'type': 'web_search',
                            'urls': results['filenames'],
                            'queries': queries,
                        }
                    )
            elif results.get('docs'):
                # Invoked when bypass embedding and retrieval is set to True
                docs = results['docs']
                files.append(
                    {
                        'docs': docs,
                        'name': ', '.join(queries),
                        'type': 'web_search',
                        'urls': results['filenames'],
                        'queries': queries,
                    }
                )

            form_data['files'] = files

            await event_emitter(
                {
                    'type': 'status',
                    'data': {
                        'action': 'web_search',
                        'description': 'Searched {{count}} sites',
                        'urls': results['filenames'],
                        'items': results.get('items', []),
                        'done': True,
                    },
                }
            )
        else:
            await event_emitter(
                {
                    'type': 'status',
                    'data': {
                        'action': 'web_search',
                        'description': 'No search results found',
                        'done': True,
                        'error': True,
                    },
                }
            )

    except Exception as e:
        log.exception(e)
        detail = e.detail if isinstance(e, HTTPException) else None
        await event_emitter(
            {
                'type': 'status',
                'data': {
                    'action': 'web_search',
                    'description': (str(detail) if detail else 'An error occurred while searching the web'),
                    'queries': queries,
                    'done': True,
                    'error': True,
                },
            }
        )

    return form_data


def get_images_from_messages(message_list):
    images = []

    for message in reversed(message_list):
        message_images = []
        for file in message.get('files', []):
            if file.get('type') == 'image':
                message_images.append(file.get('url'))
            elif file.get('content_type', '').startswith('image/'):
                message_images.append(file.get('url'))

        if message_images:
            images.append(message_images)

    return images


async def get_image_urls(delta_images, request, metadata, user) -> list[str]:
    if not isinstance(delta_images, list):
        return []

    image_urls = []
    for img in delta_images:
        if not isinstance(img, dict) or img.get('type') != 'image_url':
            continue

        url = img.get('image_url', {}).get('url')
        if not url:
            continue

        if url.startswith('data:image/png;base64'):
            url = await get_image_url_from_base64(request, url, metadata, user)

        image_urls.append(url)

    return image_urls


def has_attached_file_context(messages: list) -> bool:
    for message in messages or []:
        content = message.get('content', '') if isinstance(message, dict) else ''
        if isinstance(content, str):
            if '<attached_files>' in content:
                return True
        elif isinstance(content, list):
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get('type') == 'text'
                    and '<attached_files>' in part.get('text', '')
                ):
                    return True
    return False


async def add_file_context(messages: list, chat_id: str, user) -> list:
    """
    Add file URLs to messages for native function calling.
    """
    if not is_saved_chat_id(chat_id):
        return messages

    chat = await Chats.get_chat_by_id_and_user_id(chat_id, user.id)
    if not chat:
        return messages

    history = chat.chat.get('history', {})
    stored_messages = get_message_list(history.get('messages', {}), history.get('currentId'))

    def format_file_tag(file):
        # Every file reaching here has a url or a chat id, so id is always set.
        attrs = f'type="{file.get("type", "file")}" id="{file.get("id") or file.get("url")}"'
        if file.get('url'):
            attrs += f' url="{file["url"]}"'
        if file.get('content_type'):
            attrs += f' content_type="{file["content_type"]}"'
        if file.get('name'):
            attrs += f' name="{file["name"]}"'
        return f'<file {attrs}/>'

    # Pair only user-role messages from both lists to avoid misalignment.
    # After process_messages_with_output(), assistant messages with tool calls
    # are expanded into multiple messages (assistant + tool results), making
    # the payload message list longer than the stored message list. A naive
    # positional zip() would pair user messages with wrong stored messages,
    # causing later images to lose their file context (see #21878).
    user_messages = [m for m in messages if m.get('role') == 'user']
    stored_user_messages = [m for m in stored_messages if m.get('role') == 'user']

    for message, stored_message in zip(user_messages, stored_user_messages):
        # Chat references carry no url - they are addressed by id via view_chat.
        attached_files = [
            file
            for file in stored_message.get('files', [])
            if (file.get('url') and not file.get('url').startswith('data:'))
            or (file.get('type') == 'chat' and file.get('id'))
        ]
        if not attached_files:
            continue

        file_tags = [format_file_tag(file) for file in attached_files]
        file_context = '<attached_files>\n' + '\n'.join(file_tags) + '\n</attached_files>\n\n'

        content = message.get('content', '')
        if isinstance(content, list):
            message['content'] = [{'type': 'text', 'text': file_context}] + content
        else:
            message['content'] = file_context + content

    return messages


async def chat_image_generation_handler(request: Request, form_data: dict, extra_params: dict, user):
    metadata = extra_params.get('__metadata__', {})
    chat_id = metadata.get('chat_id', None)
    __event_emitter__ = extra_params.get('__event_emitter__', None)

    if not chat_id or not isinstance(chat_id, str) or not __event_emitter__:
        return form_data

    is_channel_chat = chat_id.startswith('channel:')
    image_metadata = {
        'message_id': metadata.get('message_id', None),
        **({'channel_id': chat_id.removeprefix('channel:')} if is_channel_chat else {'chat_id': chat_id}),
    }

    if not is_saved_chat_id(chat_id):
        message_list = form_data.get('messages', [])
    else:
        chat = await Chats.get_chat_by_id_and_user_id(chat_id, user.id)

        messages_map = chat.chat.get('history', {}).get('messages', {})
        message_id = chat.chat.get('history', {}).get('currentId')
        message_list = get_message_list(messages_map, message_id)

    user_message = get_last_user_message(message_list)

    prompt = user_message
    message_images = get_images_from_messages(message_list)

    # Limit to first 2 sets of images
    # We may want to change this in the future to allow more images
    input_images = []
    for idx, images in enumerate(message_images):
        if idx >= 2:
            break
        for image in images:
            input_images.append(image)

    # Called directly, bypassing the /images routes that enforce these switches.
    editing = len(input_images) > 0 and await Config.get('images.edit.enable')
    if not editing and not await Config.get('image_generation.enable'):
        return form_data

    if is_saved_chat_id(chat_id):
        await __event_emitter__(
            {
                'type': 'status',
                'data': {'description': 'Creating image', 'done': False},
            }
        )

    system_message_content = ''

    if editing:
        # Edit image(s)
        try:
            images = await image_edits(
                request=request,
                form_data=EditImageForm(**{'prompt': prompt, 'image': input_images}),
                metadata=image_metadata,
                user=user,
            )

            await __event_emitter__(
                {
                    'type': 'status',
                    'data': {'description': 'Image created', 'done': True},
                }
            )

            await __event_emitter__(
                {
                    'type': 'files',
                    'data': {
                        'files': [
                            {
                                'type': 'image',
                                **image,
                            }
                            for image in images
                        ]
                    },
                }
            )

            system_message_content = '<context>The requested image has been edited and created and is now being shown to the user. Let them know that it has been generated.</context>'
        except Exception as e:
            log.debug(e)

            error_message = ''
            if isinstance(e, HTTPException):
                if e.detail and isinstance(e.detail, dict):
                    error_message = e.detail.get('message', str(e.detail))
                else:
                    error_message = str(e.detail)

            await __event_emitter__(
                {
                    'type': 'status',
                    'data': {
                        'description': f'An error occurred while generating an image',
                        'done': True,
                    },
                }
            )

            system_message_content = f'<context>Image generation was attempted but failed. The system is currently unable to generate the image. Tell the user that the following error occurred: {error_message}</context>'

    elif not await Config.get('image_generation.enable'):
        await __event_emitter__(
            {
                'type': 'status',
                'data': {
                    'description': 'Image generation is disabled',
                    'done': True,
                },
            }
        )

        system_message_content = '<context>Image generation was requested but the feature is currently disabled by the administrator, so no image was created. Let the user know that image generation is currently unavailable.</context>'

    else:
        # Create image(s)
        if await Config.get('image_generation.prompt.enable'):
            try:
                res = await generate_image_prompt(
                    request,
                    {
                        'model': form_data['model'],
                        'messages': form_data['messages'],
                        'chat_id': metadata.get('chat_id'),
                    },
                    user,
                )

                # Handle JSONResponse from error paths
                if isinstance(res, JSONResponse):
                    try:
                        error_body = JSONCodec.loads(res.body)
                        detail = error_body.get('detail', 'Image prompt generation failed')
                    except Exception:
                        detail = 'Image prompt generation failed'
                    raise Exception(detail)

                response = res['choices'][0]['message']['content']

                try:
                    bracket_start = response.rfind('{')
                    bracket_end = response.rfind('}') + 1

                    if bracket_start == -1 or bracket_end == -1:
                        raise Exception('No JSON object found in the response')

                    response = response[bracket_start:bracket_end]
                    response = JSONCodec.loads(response)
                    prompt = response.get('prompt', [])
                except Exception as e:
                    prompt = user_message

            except Exception as e:
                log.exception(e)
                prompt = user_message

        try:
            images = await image_generations(
                request=request,
                form_data=CreateImageForm(**{'prompt': prompt}),
                metadata=image_metadata,
                user=user,
            )

            await __event_emitter__(
                {
                    'type': 'status',
                    'data': {'description': 'Image created', 'done': True},
                }
            )

            await __event_emitter__(
                {
                    'type': 'files',
                    'data': {
                        'files': [
                            {
                                'type': 'image',
                                **image,
                            }
                            for image in images
                        ]
                    },
                }
            )

            system_message_content = '<context>The requested image has been created by the system successfully and is now being shown to the user. Let the user know that the image they requested has been generated and is now shown in the chat.</context>'
        except Exception as e:
            log.debug(e)

            error_message = ''
            if isinstance(e, HTTPException):
                if e.detail and isinstance(e.detail, dict):
                    error_message = e.detail.get('message', str(e.detail))
                else:
                    error_message = str(e.detail)

            await __event_emitter__(
                {
                    'type': 'status',
                    'data': {
                        'description': f'An error occurred while generating an image',
                        'done': True,
                    },
                }
            )

            system_message_content = f'<context>Image generation was attempted but failed because of an error. The system is currently unable to generate the image. Tell the user that the following error occurred: {error_message}</context>'

    if system_message_content:
        form_data['messages'] = add_or_update_system_message(system_message_content, form_data['messages'])

    return form_data


async def chat_completion_files_handler(
    request: Request, body: dict, extra_params: dict, user: UserModel
) -> tuple[dict, dict[str, list]]:
    __event_emitter__ = extra_params['__event_emitter__']
    sources = []

    files = [item for item in (body.get('metadata', {}).get('files', None) or []) if item.get('type') != 'filesystem']
    if files:
        # Check if all files are in full context mode
        all_full_context = all(item.get('context') == 'full' for item in files)

        queries = []
        if not all_full_context:
            try:
                queries_response = await generate_queries(
                    request,
                    {
                        'model': body['model'],
                        'messages': body['messages'],
                        'type': 'retrieval',
                        'chat_id': body.get('metadata', {}).get('chat_id'),
                    },
                    user,
                )
                queries_response = queries_response['choices'][0]['message']['content']

                try:
                    bracket_start = queries_response.rfind('{')
                    bracket_end = queries_response.rfind('}') + 1

                    if bracket_start == -1 or bracket_end == -1:
                        raise Exception('No JSON object found in the response')

                    queries_response = queries_response[bracket_start:bracket_end]
                    queries_response = JSONCodec.loads(queries_response)
                except Exception as e:
                    queries_response = {'queries': [queries_response]}

                queries = queries_response.get('queries', [])
            except Exception:
                pass

            await __event_emitter__(
                {
                    'type': 'status',
                    'data': {
                        'action': 'queries_generated',
                        'queries': queries,
                        'done': False,
                    },
                }
            )

        if len(queries) == 0:
            queries = [get_last_user_message(body['messages']) or '']

        try:
            # One batched SELECT instead of six sequential round trips.
            rag_config = await Config.get_many(
                'rag.top_k',
                'rag.top_k_reranker',
                'rag.relevance_threshold',
                'rag.hybrid_bm25_weight',
                'rag.enable_hybrid_search',
                'rag.full_context',
            )
            # Directly await async get_sources_from_items (no thread needed - fully async now)
            sources = await get_sources_from_items(
                request=request,
                items=files,
                queries=queries,
                embedding_function=lambda query, prefix: request.app.state.EMBEDDING_FUNCTION(
                    query, prefix=prefix, user=user
                ),
                k=rag_config.get('rag.top_k'),
                reranking_function=(
                    (lambda query, documents: request.app.state.RERANKING_FUNCTION(query, documents, user=user))
                    if request.app.state.RERANKING_FUNCTION
                    else None
                ),
                k_reranker=rag_config.get('rag.top_k_reranker'),
                r=rag_config.get('rag.relevance_threshold'),
                hybrid_bm25_weight=rag_config.get('rag.hybrid_bm25_weight'),
                hybrid_search=rag_config.get('rag.enable_hybrid_search'),
                full_context=all_full_context or rag_config.get('rag.full_context'),
                user=user,
            )
        except Exception as e:
            log.exception(e)

        log.debug('rag_contexts:sources: %s', sources)

        unique_ids = set()
        for source in sources or []:
            if not source or len(source.keys()) == 0:
                continue

            documents = source.get('document') or []
            metadatas = source.get('metadata') or []
            src_info = source.get('source') or {}

            for index, _ in enumerate(documents):
                metadata = metadatas[index] if index < len(metadatas) else None
                _id = (metadata or {}).get('source') or (src_info or {}).get('id') or 'N/A'
                unique_ids.add(_id)

        sources_count = len(unique_ids)
        await __event_emitter__(
            {
                'type': 'status',
                'data': {
                    'action': 'sources_retrieved',
                    'count': sources_count,
                    'done': True,
                },
            }
        )

    return body, {'sources': sources}


async def convert_url_images_to_base64(form_data, user=None):
    messages = form_data.get('messages', [])

    for message in messages:
        content = message.get('content')
        if not isinstance(content, list):
            continue

        new_content = []

        for item in content:
            if not isinstance(item, dict) or item.get('type') != 'image_url':
                new_content.append(item)
                continue

            image_url_data = item.get('image_url', {})
            if isinstance(image_url_data, dict):
                image_url = image_url_data.get('url') or ''
            elif isinstance(image_url_data, str):
                image_url = image_url_data
            else:
                image_url = ''
            if image_url.startswith('data:image/'):
                new_content.append(item)
                continue

            try:
                base64_data = await get_image_base64_from_url(image_url, user=user)
                if base64_data:
                    image_url_payload = {'url': base64_data}
                    if isinstance(image_url_data, dict) and image_url_data.get('detail'):
                        image_url_payload['detail'] = image_url_data['detail']
                    new_content.append(
                        {
                            'type': 'image_url',
                            'image_url': image_url_payload,
                        }
                    )
                else:
                    new_content.append(item)
            except Exception as e:
                log.debug('Error converting image URL to base64: %s', e)
                new_content.append(item)

        message['content'] = new_content

    return form_data


MESSAGE_REPLAY_KEYS = ('id', 'role', 'content', 'output', 'files', 'contextSummary', 'usage', 'model')


async def load_messages_from_db(chat_id: str, message_id: str) -> Optional[list[dict]]:
    """
    Load the message chain from DB up to message_id,
    keeping only fields needed to rebuild the LLM payload.
    """
    messages_map = await Chats.get_messages_map_by_chat_id(chat_id)
    if not messages_map:
        return None

    db_messages = get_message_list(messages_map, message_id)
    if not db_messages:
        return None

    return [{k: v for k, v in msg.items() if k in MESSAGE_REPLAY_KEYS} for msg in db_messages]


def get_reasoning_format(model: dict) -> str | None:
    """
    Determine how reasoning should be included in reconstructed messages.

    Returns:
        'thinking': Ollama expects reasoning in the native thinking field.
        'think_tags': wrap reasoning in <think> tags inside content.
        'reasoning_content': llama.cpp supports reasoning_content as a top-level field.
        None: skip reasoning (safe default for strict providers).
    """
    provider = model.get('provider', '')
    if model.get('owned_by') == 'ollama':
        return 'thinking'
    if provider == 'llama.cpp':
        return 'reasoning_content'
    return None


def strip_reasoning_details(output: list) -> list:
    return [
        {key: value for key, value in item.items() if key != 'reasoning_details'} if isinstance(item, dict) else item
        for item in output
    ]


def process_messages_with_output(
    messages: list[dict],
    reasoning_format: str | None = None,
) -> list[dict]:
    """
    Process messages with OR-aligned output items for LLM consumption.

    For assistant messages with 'output' field, produces properly formatted
    OpenAI-style messages (tool_calls + tool results). Strips 'output' before LLM.
    """
    processed = []

    for message in messages:
        if message.get('role') == 'assistant' and message.get('output'):
            # Use output items for clean OpenAI-format messages
            output_messages = convert_output_to_messages(
                message['output'],
                raw=True,
                reasoning_format=reasoning_format,
                flatten_tool_images=True,
            )
            web_search_resume_message = convert_web_search_output_to_resume_message(message['output'])
            if web_search_resume_message:
                output_messages.append(web_search_resume_message)

            if output_messages:
                processed.extend(output_messages)
                continue

        clean_message = dict(message)
        for key in ('id', 'files', 'output', 'model', 'contextSummary', 'context_summary', 'usage'):
            clean_message.pop(key, None)
        processed.append(clean_message)

    return processed


def sanitize_tool_pairs(messages: list[dict]) -> list[dict]:
    tool_result_ids = {
        message.get('tool_call_id')
        for message in messages
        if message.get('role') == 'tool' and message.get('tool_call_id')
    }

    tool_call_ids = {
        tool_call.get('id')
        for message in messages
        for tool_call in (message.get('tool_calls') or [])
        if message.get('role') == 'assistant' and tool_call.get('id')
    }

    sanitized = []
    for message in messages:
        if message.get('role') == 'assistant' and message.get('tool_calls'):
            kept = [
                tool_call for tool_call in message.get('tool_calls') or [] if tool_call.get('id') in tool_result_ids
            ]
            if kept:
                sanitized.append({**message, 'tool_calls': kept})
            else:
                clean = dict(message)
                clean.pop('tool_calls', None)
                clean.pop('reasoning_items', None)
                if clean.get('content'):
                    sanitized.append(clean)
        elif message.get('role') != 'tool' or message.get('tool_call_id') in tool_call_ids:
            sanitized.append(message)

    return sanitized


# Ids are validated as [a-z0-9_-]+ on create; matching that keeps ordinary "<$..." text intact.
SKILL_MENTION_RE = re.compile(r'<(?:\$([a-z0-9_-]+)(?:\|[^>]*)?|/([a-z0-9_-]+)\|[^>]*)>')


def _get_text_parts(message: dict) -> list[str]:
    """Return all text segments from a message's content."""
    content = message.get('content')
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [p.get('text', '') for p in content if isinstance(p, dict) and p.get('type') == 'text']
    return []


def extract_skill_ids_from_messages(messages: list[dict]) -> set[str]:
    """Extract skill IDs from <$skillId|label> and </skillId|label> mention tags."""
    ids: set[str] = set()
    for message in messages:
        for text in _get_text_parts(message):
            ids.update(m.group(1) or m.group(2) for m in SKILL_MENTION_RE.finditer(text))
    return ids


SKILL_MENTION_STRIP_RE = re.compile(r'<(?:\$[a-z0-9_-]+(?:\|([^>]*))?|/[a-z0-9_-]+\|([^>]*))>')


def strip_skill_mentions(messages: list[dict]) -> None:
    """Replace <$skillId|label> and </skillId|label> mention tags with the label in-place."""

    def label(match):
        return match.group(1) or match.group(2) or ''

    for message in messages:
        content = message.get('content')
        if isinstance(content, str) and SKILL_MENTION_STRIP_RE.search(content):
            message['content'] = SKILL_MENTION_STRIP_RE.sub(label, content).strip()
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get('type') == 'text':
                    text = part.get('text', '')
                    if SKILL_MENTION_STRIP_RE.search(text):
                        part['text'] = SKILL_MENTION_STRIP_RE.sub(label, text).strip()


async def connect_mcp_server(
    request,
    server_id: str,
    user,
    metadata: dict,
    extra_params: dict,
) -> tuple[MCPClient, list[dict]] | None:
    """Resolve an MCP server connection, authenticate, and return (client, tool_specs).

    Returns None if the server is not found or access is denied.
    """
    mcp_server_connection = None
    for server_connection in await Config.get('tool_server.connections', []):
        if server_connection.get('type', '') == 'mcp' and (server_connection.get('info') or {}).get('id') == server_id:
            mcp_server_connection = server_connection
            break

    if not mcp_server_connection:
        log.error(f'MCP server with id {server_id} not found')
        return None

    if not await has_connection_access(user, mcp_server_connection):
        log.warning(f'Access denied to MCP server {server_id} for user {user.id}')
        return None

    headers, _ = await build_tool_server_headers(
        mcp_server_connection,
        request,
        user,
        server_id=server_id,
        metadata=metadata,
        extra_params=extra_params,
    )

    client = MCPClient()
    await client.connect(
        url=mcp_server_connection.get('url', ''),
        headers=headers if headers else None,
    )

    function_name_filter_list = mcp_server_connection.get('config', {}).get('function_name_filter_list', '')
    if isinstance(function_name_filter_list, str):
        function_name_filter_list = function_name_filter_list.split(',')

    tool_specs = await client.list_tool_specs()
    if function_name_filter_list:
        tool_specs = [spec for spec in tool_specs if is_string_allowed(spec['name'], function_name_filter_list)]

    return client, tool_specs


async def process_chat_payload(request, form_data, user, metadata, model):
    # Ensure chat_id is always a string — external API clients may omit it.
    if not isinstance(metadata.get('chat_id'), str):
        metadata['chat_id'] = ''

    # Pipeline Inlet -> Filter Inlet -> Chat Memory -> Chat Web Search -> Chat Image Generation
    # -> Chat Code Interpreter (Form Data Update) -> (Default) Chat Tools Function Calling
    # -> Chat Files

    # Arena model resolution — pick the sub-model now so all downstream
    # processing (knowledge, capabilities, tools, params) uses its settings
    # instead of the empty arena wrapper.
    if model.get('owned_by') == 'arena':
        arena_model_ids = model.get('info', {}).get('meta', {}).get('model_ids')
        arena_filter_mode = model.get('info', {}).get('meta', {}).get('filter_mode')
        if arena_model_ids and arena_filter_mode == 'exclude':
            arena_model_ids = [
                available_model['id']
                for available_model in request.app.state.MODELS.values()
                if available_model.get('owned_by') != 'arena' and available_model['id'] not in arena_model_ids
            ]

        if isinstance(arena_model_ids, list) and arena_model_ids:
            selected_model_id = random.choice(arena_model_ids)
        else:
            arena_model_ids = [
                available_model['id']
                for available_model in request.app.state.MODELS.values()
                if available_model.get('owned_by') != 'arena'
            ]
            selected_model_id = random.choice(arena_model_ids)

        selected_model = request.app.state.MODELS.get(selected_model_id)
        if selected_model:
            model = selected_model
            form_data['model'] = selected_model_id
            metadata['selected_model_id'] = selected_model_id

    # Captured before apply_params_to_form_data pops 'params'; populates metadata['system_prompt'] below
    model_system_prompt = (form_data.get('params') or {}).get('system')

    form_data = apply_params_to_form_data(form_data, model)
    log.debug('form_data: %s', form_data)

    # Guided regeneration: extract before it reaches the LLM provider
    regeneration_prompt = form_data.pop('regeneration_prompt', None)

    # Load messages from DB when available — DB preserves structured 'output' items
    # which the frontend strips, causing tool calls to be merged into content.
    chat_id = metadata.get('chat_id')
    user_message_id = metadata.get('user_message_id')

    if is_saved_chat_id(chat_id) and user_message_id:
        db_messages = await load_messages_from_db(chat_id, user_message_id)
        if db_messages:
            # Continue: frontend sends assistant_message_id when continuing
            # an existing response. Load its content so the LLM sees prior output.
            assistant_message_id = metadata.get('assistant_message_id')
            if assistant_message_id:
                assistant_message = await Chats.get_message_by_id_and_message_id(chat_id, assistant_message_id)
                if assistant_message and (assistant_message.get('content') or assistant_message.get('output')):
                    db_messages.append({k: v for k, v in assistant_message.items() if k in MESSAGE_REPLAY_KEYS})

            system_message = get_system_message(form_data.get('messages', []))
            form_data['messages'] = [system_message, *db_messages] if system_message else db_messages

            # Inject image files into content as image_url parts (mirrors frontend logic)
            for message in form_data['messages']:
                image_files = [
                    f
                    for f in message.get('files', [])
                    if f.get('type') == 'image' or (f.get('content_type') or '').startswith('image/')
                ]
                if message.get('role') == 'user' and image_files:
                    text_content = message.get('content', '')
                    if isinstance(text_content, str):
                        message['content'] = [
                            {'type': 'text', 'text': text_content},
                            *[
                                {
                                    'type': 'image_url',
                                    'image_url': {'url': f['url']},
                                }
                                for f in image_files
                                if f.get('url')
                            ],
                        ]
                # Strip files field — it's been incorporated into content
                message.pop('files', None)

    if regeneration_prompt:
        form_data['messages'].append({'role': 'user', 'content': regeneration_prompt})

    if is_saved_chat_id(chat_id) and user_message_id:
        if getattr(request.state, 'direct', False) and hasattr(request.state, 'model'):
            compaction_models = {
                **dict(request.app.state.MODELS.items()),
                request.state.model['id']: request.state.model,
            }
        else:
            compaction_models = request.app.state.MODELS

        system_message = get_system_message(form_data.get('messages', []))
        system_prompt = get_content_from_message(system_message) if system_message else ''

        try:
            form_data['messages'], context_summary, _ = await compact_messages_for_request(
                request,
                user,
                form_data.get('messages', []),
                metadata,
                form_data.get('model'),
                compaction_models,
                system_prompt,
            )
            if context_summary:
                form_data['messages'] = add_or_update_system_message(
                    f'[CONVERSATION SUMMARY]\n{context_summary}',
                    form_data['messages'],
                    append=True,
                )
        except Exception:
            log.exception('Context compaction failed; continuing with full chat history')

    # Process messages with OR-aligned output items for clean LLM messages
    for message in form_data.get('messages', []):
        output = message.get('output')
        # reasoning_details can be model/provider-bound, so only replay them
        # for output produced by the same model.
        if message.get('role') == 'assistant' and message.get('model') != model['id'] and isinstance(output, list):
            message['output'] = strip_reasoning_details(output)

    form_data['messages'] = process_messages_with_output(
        form_data.get('messages', []),
        reasoning_format=get_reasoning_format(model),
    )
    form_data['messages'] = sanitize_tool_pairs(form_data['messages'])

    system_message = get_system_message(form_data.get('messages', []))
    if system_message:  # Chat Controls/User Settings
        try:
            form_data = await apply_system_prompt_to_body(
                system_message.get('content'), form_data, metadata, user, replace=True
            )  # Required to handle system prompt variables
        except Exception:
            pass

    form_data = await convert_url_images_to_base64(form_data, user=user)

    event_emitter = await get_event_emitter(metadata)
    event_caller = await get_event_call(metadata)

    extra_params = {
        '__event_emitter__': event_emitter,
        '__event_call__': event_caller,
        '__user__': user.model_dump() if isinstance(user, UserModel) else {},
        '__metadata__': metadata,
        '__oauth_token__': await get_system_oauth_token(request, user),
        '__request__': request,
        '__model__': model,
        '__chat_id__': metadata.get('chat_id'),
        '__message_id__': metadata.get('message_id'),
    }
    # Initialize events to store additional event to be sent to the client
    # Initialize contexts and citation
    if getattr(request.state, 'direct', False) and hasattr(request.state, 'model'):
        models = {
            request.state.model['id']: request.state.model,
        }
    else:
        models = request.app.state.MODELS

    task_model_id = get_task_model_id(
        form_data['model'],
        await Config.get('task.model.default'),
        await Config.get('task.model.external'),
        models,
    )

    events = []
    sources = []

    # Folder "Project" handling
    # Check if the request has chat_id and is inside of a folder
    # Uses lightweight column query — only fetches folder_id, not the full chat JSON blob
    chat_id = metadata.get('chat_id', None)
    folder_id = None
    if user and is_saved_chat_id(chat_id):
        folder_id = await Chats.get_chat_folder_id(chat_id, user.id)

    # Fallback: use folder_id from metadata (temporary chats have no DB record)
    if not folder_id:
        folder_id = metadata.get('folder_id', None)

    if folder_id and user:
        folder = await Folders.get_folder_by_id(folder_id)
        if folder and user.role != 'admin' and not await has_folder_access(user.id, folder, 'read', db=None):
            folder = None

        if folder and folder.data:
            if 'system_prompt' in folder.data:
                form_data = await apply_system_prompt_to_body(folder.data['system_prompt'], form_data, metadata, user)
            if 'files' in folder.data:
                if metadata.get('params', {}).get('function_calling') == 'legacy':
                    form_data['files'] = [
                        {'type': 'folder', 'id': folder.id},
                        *form_data.get('files', []),
                    ]
                else:
                    # Native FC: skip RAG injection, builtin tools
                    # will read folder knowledge from metadata.
                    metadata['folder_knowledge'] = await get_owner_accessible_folder_files(folder)

    # Model "Knowledge" handling
    user_message = get_last_user_message(form_data['messages'])
    model_knowledge = model.get('info', {}).get('meta', {}).get('knowledge', False)

    if model_knowledge and metadata.get('params', {}).get('function_calling') == 'legacy':
        await event_emitter(
            {
                'type': 'status',
                'data': {
                    'action': 'knowledge_search',
                    'query': user_message,
                    'done': False,
                },
            }
        )

        knowledge_files = []
        for item in model_knowledge:
            if item.get('collection_name'):
                knowledge_files.append(
                    {
                        'id': item.get('collection_name'),
                        'name': item.get('name'),
                        'legacy': True,
                    }
                )
            elif item.get('collection_names'):
                knowledge_files.append(
                    {
                        'name': item.get('name'),
                        'type': 'collection',
                        'collection_names': item.get('collection_names'),
                        'legacy': True,
                    }
                )
            else:
                knowledge_files.append(item)

        files = form_data.get('files', [])
        files.extend(knowledge_files)
        form_data['files'] = files

    variables = form_data.pop('variables', None)
    payload_tools = form_data.get('tools', None)  # snapshot before filters

    # Process the form_data through the pipeline
    try:
        form_data = await process_pipeline_inlet_filter(request, form_data, user, models)
    except Exception as e:
        raise e

    filter_functions = []
    filter_context = get_filter_context(request) if ENABLE_PLUGINS else None
    if ENABLE_PLUGINS:
        try:
            filter_functions = await get_filter_functions(request, model, metadata.get('filter_ids', []))

            form_data, flags = await process_filter_functions(
                request=request,
                filter_context=filter_context,
                filter_functions=filter_functions,
                filter_type='inlet',
                form_data=form_data,
                extra_params=extra_params,
            )
        except Exception as e:
            raise Exception(f'{e}')

    features = form_data.pop('features', None) or {}
    extra_params['__features__'] = features
    if features:
        if 'voice' in features and features['voice']:
            if await Config.get('task.voice.prompt.enable'):
                template = await Config.get('task.voice.prompt_template')
                if not template:
                    template = DEFAULT_VOICE_MODE_PROMPT_TEMPLATE

                form_data['messages'] = add_or_update_system_message(
                    template,
                    form_data['messages'],
                )

        if 'memory' in features and features['memory'] and await Config.get('memories.system_context.enable'):
            # features is client-supplied; re-check the permission the native FC path enforces.
            if getattr(user, 'role', None) == 'admin' or await has_permission(
                getattr(user, 'id', ''),
                'features.memories',
                await Config.get('user.permissions'),
            ):
                form_data = await add_memory_context(
                    request,
                    form_data,
                    user,
                    model,
                    event_emitter=extra_params.get('__event_emitter__'),
                )

        if 'web_search' in features and features['web_search'] and await Config.get('web.search.enable'):
            # features is client-supplied; re-check the permission the native FC path enforces.
            if getattr(user, 'role', None) == 'admin' or await has_permission(
                getattr(user, 'id', ''),
                'features.web_search',
                await Config.get('user.permissions'),
            ):
                # Skip forced RAG web search when native FC is enabled - model can use web_search tool
                if metadata.get('params', {}).get('function_calling') == 'legacy':
                    form_data = await chat_web_search_handler(request, form_data, extra_params, user)

        if 'image_generation' in features and features['image_generation']:
            # features is client-supplied; re-check the permission the direct /images routes enforce.
            if getattr(user, 'role', None) == 'admin' or await has_permission(
                getattr(user, 'id', ''),
                'features.image_generation',
                await Config.get('user.permissions'),
            ):
                # Skip forced image generation when native FC is enabled - model can use generate_image tool
                if metadata.get('params', {}).get('function_calling') == 'legacy':
                    form_data = await chat_image_generation_handler(request, form_data, extra_params, user)

        if 'code_interpreter' in features and features['code_interpreter']:
            engine = await Config.get('code_interpreter.engine', 'pyodide')

            # Skip XML-tag prompt injection when native FC is enabled —
            # execute_code will be injected as a builtin tool instead
            if metadata.get('params', {}).get('function_calling') == 'legacy':
                ci_prompt_template = await Config.get('code_interpreter.prompt_template')
                prompt = ci_prompt_template if ci_prompt_template != '' else DEFAULT_CODE_INTERPRETER_PROMPT

                # Append filesystem awareness only for pyodide engine
                if engine != 'jupyter':
                    prompt += CODE_INTERPRETER_PYODIDE_PROMPT

                form_data['messages'] = add_or_update_user_message(
                    prompt,
                    form_data['messages'],
                )
            else:
                # Native FC: tool docstring can't be dynamic, so inject
                # filesystem context into the system message for pyodide
                # engine.  Appending to the system prompt (instead of the
                # user message) keeps it in the stable cached prefix so
                # providers with prefix caching don't re-bill the full
                # conversation on every turn.
                if engine != 'jupyter':
                    form_data['messages'] = add_or_update_system_message(
                        CODE_INTERPRETER_PYODIDE_PROMPT,
                        form_data['messages'],
                        append=True,
                    )

    tool_ids = form_data.pop('tool_ids', None)
    terminal_id = form_data.pop('terminal_id', None)
    files = form_data.pop('files', None)
    form_data.pop('folder_id', None)

    # If the original caller provided tools, use them as-is (skip resolution).
    # Otherwise, save any tools that filter inlets added for merging later.
    inlet_filter_tools = None if payload_tools is not None else form_data.get('tools', None)

    # Mentioned skills get full content; selected/default skills can be loaded through view_skill.
    mentioned_skill_ids = extract_skill_ids_from_messages(form_data.get('messages', []))
    skill_ids = sorted(
        set(form_data.pop('skill_ids', None) or [])
        | set(model.get('info', {}).get('meta', {}).get('skillIds', []))
        | mentioned_skill_ids
    )
    available_skills = []
    view_skill_ids = []
    chat = None
    if is_saved_chat_id(metadata.get('chat_id')):
        chat = await Chats.get_chat_by_id(metadata['chat_id'])

    is_note_chat = bool(chat and (chat.meta or {}).get('internal') is True and (chat.meta or {}).get('type') == 'note')

    if is_note_chat:
        note_id = (chat.meta or {}).get('note_id')
        note = await Notes.get_note_by_id(note_id) if note_id else None
        if note and (
            user.role == 'admin'
            or note.user_id == user.id
            or await AccessGrants.has_access(
                user_id=user.id,
                resource_type='note',
                resource_id=note.id,
                permission='read',
            )
        ):
            note_files = [
                file
                for file in ((note.data or {}).get('files') or [])
                if isinstance(file, dict)
                and file.get('type') != 'image'
                and not (file.get('content_type') or '').startswith('image/')
            ]
            if note_files:
                files = [*(files or []), *note_files]

    use_builtin_tools = is_note_chat or (
        bool(metadata.get('session_id'))
        and metadata.get('params', {}).get('function_calling') != 'legacy'
        and (model.get('info', {}).get('meta', {}).get('capabilities') or {}).get('builtin_tools', True)
    )

    if skill_ids:
        from open_webui.models.skills import Skills as SkillsModel

        accessible_skills = {s.id: s for s in await SkillsModel.get_skills(user_id=user.id, ids=skill_ids)}
        for sid in skill_ids:
            s = accessible_skills.get(sid)
            if s and s.is_active:
                available_skills.append(s)

        skill_manifest = ''
        for skill in available_skills:
            if skill.id in mentioned_skill_ids or not use_builtin_tools:
                form_data['messages'] = add_or_update_system_message(
                    f'<skill name="{skill.name}">\n{skill.content}\n</skill>',
                    form_data['messages'],
                    append=True,
                )
            else:
                view_skill_ids.append(skill.id)
                skill_manifest += (
                    f'<skill>\n<id>{skill.id}</id>\n<name>{skill.name}</name>\n'
                    f'<description>{skill.description or ""}</description>\n</skill>\n'
                )

        if skill_manifest:
            form_data['messages'] = add_or_update_system_message(
                f'<available_skills>\n{skill_manifest}</available_skills>',
                form_data['messages'],
                append=True,
            )

    # Strip <$skillId|label> mention tags so the model doesn't see raw markup.
    strip_skill_mentions(form_data.get('messages', []))

    prompt = get_last_user_message(form_data['messages'])

    # Guard against empty user message after skill mention stripping.
    # When a user selects a skill ($skill-name) without typing additional text,
    # the stripped result is an empty string which causes 400 errors on providers
    # that reject empty content blocks (e.g. AWS Bedrock ConverseStream).
    if not prompt or not prompt.strip():
        fallback = ', '.join(s.name for s in available_skills)
        if fallback:
            set_last_user_message_content(fallback, form_data['messages'])
            prompt = fallback
    # TODO: re-enable URL extraction from prompt
    # urls = []
    # if prompt and len(prompt or "") < 500 and (not files or len(files) == 0):
    #     urls = extract_urls(prompt)

    if files:
        # files = [*files, *[{"type": "url", "url": url, "name": url} for url in urls]]
        # Remove duplicate files based on their content
        files = list({json.dumps(f, sort_keys=True): f for f in files}.values())

    metadata.update(
        {
            'model_id': form_data.get('model'),
            'tool_ids': tool_ids,
            'skill_ids': skill_ids,
            'terminal_id': terminal_id,
            'files': files,
            'features': features,
        }
    )
    form_data['metadata'] = metadata

    # When the caller provides an explicit `tools` key in the request body,
    # skip all server-side tool resolution and pass the caller's tools through
    # unchanged.  Sending `tools: []` explicitly opts out of builtin injection.
    if payload_tools is None:
        # Server side tools
        tool_ids = metadata.get('tool_ids', None)
        # Client side tools
        direct_tool_servers = metadata.get('tool_servers', None)

        log.debug('tool_ids=%r', tool_ids)
        log.debug('direct_tool_servers=%r', direct_tool_servers)

        tools_dict = {}

        mcp_clients = {}
        mcp_tools_dict = {}

        if tool_ids:
            db_tool_ids = []
            for tool_id in tool_ids:
                if tool_id.startswith('server:mcp:'):
                    try:
                        server_id = tool_id[len('server:mcp:') :]

                        result = await connect_mcp_server(
                            request,
                            server_id,
                            user,
                            metadata,
                            extra_params,
                        )
                        if result is None:
                            continue

                        client, tool_specs = result
                        mcp_clients[server_id] = client

                        for tool_spec in tool_specs:

                            async def make_tool_function(client, function_name):
                                async def tool_function(**kwargs):
                                    return await client.call_tool(
                                        function_name,
                                        function_args=kwargs,
                                    )

                                return tool_function

                            tool_function = await make_tool_function(client, tool_spec['name'])

                            mcp_tools_dict[f'{server_id}_{tool_spec["name"]}'] = {
                                'spec': {
                                    **tool_spec,
                                    'name': f'{server_id}_{tool_spec["name"]}',
                                },
                                'callable': tool_function,
                                'type': 'mcp',
                                'client': client,
                                'direct': False,
                            }
                    except Exception as e:
                        log.debug(e)
                        if event_emitter:
                            await event_emitter(
                                {
                                    'type': 'chat:message:error',
                                    'data': {'error': {'content': f"Failed to connect to MCP server '{server_id}'"}},
                                }
                            )
                        continue
                elif ENABLE_PLUGINS:
                    db_tool_ids.append(tool_id)

            if db_tool_ids:
                tools_dict = await get_tools(
                    request,
                    db_tool_ids,
                    user,
                    {
                        **extra_params,
                        '__model__': models[task_model_id],
                        '__messages__': form_data['messages'],
                        '__files__': metadata.get('files', []),
                    },
                )

            if mcp_tools_dict:
                tools_dict = {**tools_dict, **mcp_tools_dict}

        # Resolve terminal tools if terminal_id is set (outside tool_ids check
        # so system terminals work even when no other tools are selected)
        terminal_capability = (model.get('info', {}).get('meta', {}).get('capabilities') or {}).get('terminal', True)
        if terminal_id and terminal_capability:
            try:
                terminal_result = await get_terminal_tools(
                    request,
                    terminal_id,
                    user,
                    extra_params,
                )
                if isinstance(terminal_result, tuple):
                    terminal_tools, system_prompt = terminal_result
                else:
                    terminal_tools = terminal_result
                    system_prompt = None
                if terminal_tools:
                    tools_dict = {**tools_dict, **terminal_tools}
                if system_prompt:
                    form_data['messages'] = add_or_update_system_message(
                        system_prompt,
                        form_data['messages'],
                        append=True,
                    )
            except Exception as e:
                log.exception(e)
                raise HTTPException(status_code=503, detail=f'Terminal unavailable: {e}') from e

        if direct_tool_servers:
            for tool_server in direct_tool_servers:
                system_prompt = tool_server.pop('system_prompt', None)
                if system_prompt:
                    form_data['messages'] = add_or_update_system_message(
                        system_prompt,
                        form_data['messages'],
                        append=True,
                    )

                tool_specs = tool_server.pop('specs', [])

                for tool in tool_specs:
                    tools_dict[tool['name']] = {
                        'spec': tool,
                        'direct': True,
                        'server': tool_server,
                    }

        if mcp_clients:
            metadata['mcp_clients'] = mcp_clients

        # Inject builtin tools for native function calling based on enabled features and model capability.
        # Only inject when the request originates from the UI (identified by session_id).
        # API callers don't expect hidden tools; they can explicitly request tools via tool_ids.
        if use_builtin_tools:
            # Add file context to user messages
            chat_id = metadata.get('chat_id')
            form_data['messages'] = await add_file_context(form_data.get('messages', []), chat_id, user)
            metadata['has_attached_files'] = has_attached_file_context(form_data.get('messages', []))

            if (model.get('info', {}).get('meta', {}).get('builtinTools') or {}).get('knowledge', True):
                from html import escape

                knowledge_tags = []
                for item in get_attached_knowledge(model, metadata):
                    if not item.get('id') or not item.get('type'):
                        continue
                    attrs = f'type="{escape(str(item["type"]), quote=True)}" id="{escape(str(item["id"]), quote=True)}"'
                    if item.get('name'):
                        attrs += f' name="{escape(str(item["name"]), quote=True)}"'
                    if item.get('source'):
                        attrs += f' source="{escape(str(item["source"]), quote=True)}"'
                    knowledge_tags.append(f'<knowledge {attrs}/>')

                if knowledge_tags:
                    form_data['messages'] = add_or_update_system_message(
                        '<attached_knowledge>\n' + '\n'.join(knowledge_tags) + '\n</attached_knowledge>',
                        form_data['messages'],
                        append=True,
                    )

            builtin_tools = await get_builtin_tools(
                request,
                {
                    **extra_params,
                    '__event_emitter__': event_emitter,
                    '__skill_ids__': view_skill_ids,
                },
                features,
                model,
                is_note_chat=is_note_chat,
            )
            for name, tool_dict in builtin_tools.items():
                if name not in tools_dict:
                    tools_dict[name] = tool_dict

        if tools_dict:
            # Always store resolved tools in metadata so downstream consumers
            # (e.g. pipe functions) can access all tools including MCP and builtins.
            metadata['tools'] = tools_dict

            if metadata.get('params', {}).get('function_calling') != 'legacy':
                # If the function calling is native, then call the tools function calling handler
                form_data['tools'] = [
                    {'type': 'function', 'function': tool.get('spec', {})} for tool in tools_dict.values()
                ]
                if inlet_filter_tools:
                    form_data['tools'].extend(inlet_filter_tools)
            else:
                # If the function calling is not native, then call the tools function calling handler
                try:
                    form_data, flags = await chat_completion_tools_handler(
                        request, form_data, extra_params, user, models, tools_dict
                    )
                    sources.extend(flags.get('sources', []))
                except Exception as e:
                    log.exception(e)

    # Check if file context extraction is enabled for this model (default True).
    # OpenAI-backed models must not use Open WebUI's local RAG/file-context path:
    # file attachments are handled later by the OpenAI router via direct message
    # context or official Responses file uploads.
    file_context_enabled = (model.get('info', {}).get('meta', {}).get('capabilities') or {}).get('file_context', True)
    local_file_context_allowed = file_context_enabled and model.get('owned_by') != 'openai'

    if local_file_context_allowed:
        try:
            form_data, flags = await chat_completion_files_handler(request, form_data, extra_params, user)
            sources.extend(flags.get('sources', []))
        except Exception as e:
            log.exception(e)

    # Save the pre-RAG message state so the native tool call loop can
    # restore to the true original (before file-source injection) rather
    # than a snapshot that already has the RAG template baked in.
    system_message = get_system_message(form_data['messages'])
    system_content = get_content_from_message(system_message) if system_message else ''
    resolved_model_system_prompt = await resolve_system_prompt(
        model_system_prompt,
        metadata,
        user,
    )
    if resolved_model_system_prompt:
        system_content = (
            f'{resolved_model_system_prompt}\n{system_content}' if system_content else resolved_model_system_prompt
        )
    metadata['system_prompt'] = system_content or None
    metadata['user_prompt'] = get_last_user_message(form_data['messages'])
    metadata['sources'] = sources[:] if sources else []

    # If context is not empty, insert it into the messages
    if sources and prompt:
        form_data['messages'] = await apply_source_context_to_messages(request, form_data['messages'], sources, prompt)

    # If there are citations, add them to the data_items
    sources = [
        source
        for source in sources
        if source.get('source', {}).get('name', '') or source.get('source', {}).get('id', '')
    ]

    if len(sources) > 0:
        events.append({'sources': sources})

    if model_knowledge:
        await event_emitter(
            {
                'type': 'status',
                'data': {
                    'action': 'knowledge_search',
                    'query': user_message,
                    'done': True,
                    'hidden': True,
                },
            }
        )

    if ENABLE_PLUGINS:
        try:
            form_data, _ = await process_filter_functions(
                request=request,
                filter_context=filter_context,
                filter_functions=filter_functions,
                filter_type='request',
                form_data=form_data,
                extra_params=extra_params,
            )
        except Exception as e:
            raise Exception(f'{e}')

    form_data = normalize_messages_for_model(form_data)

    return form_data, metadata, events


async def get_event_emitter_and_caller(metadata):
    event_emitter = None
    event_caller = None

    # event_emitter only needs user_id + chat_id + message_id.
    # It broadcasts to user:{user_id} room AND persists to DB,
    # so it works for backend-initiated calls (automations, API).
    if metadata.get('chat_id') and metadata.get('message_id'):
        event_emitter = await get_event_emitter(metadata)

    # event_caller needs session_id — it calls back to a specific
    # websocket session (used by direct tools, pyodide code interpreter).
    if metadata.get('session_id') and metadata.get('chat_id') and metadata.get('message_id'):
        event_caller = await get_event_call(metadata)

    return event_emitter, event_caller


async def build_chat_response_context(request, form_data, user, model, metadata, tasks, events):
    event_emitter, event_caller = await get_event_emitter_and_caller(metadata)
    return {
        'request': request,
        'form_data': form_data,
        'user': user,
        'model': model,
        'metadata': metadata,
        'tasks': tasks,
        'events': events,
        'event_emitter': event_emitter,
        'event_caller': event_caller,
    }


async def execute_tool_call_for_output(request, form_data, user, metadata, event_caller, event_emitter, tool_call):
    tools = metadata.get('tools', {})
    name = tool_call.get('function', {}).get('name', '')
    tool_args = tool_call.get('function', {}).get('arguments', '{}')
    params = {}
    if tool_args and tool_args.strip():
        try:
            params = JSONCodec.loads(tool_args)
        except Exception:
            try:
                params = ast.literal_eval(tool_args)
            except Exception as e:
                log.debug(e)
                return {
                    'tool_call_id': tool_call.get('id', ''),
                    'content': (
                        'Error: Tool call arguments could not be parsed. '
                        'The model generated malformed or incomplete JSON.'
                    ),
                }
    tool_call.setdefault('function', {})['arguments'] = JSONCodec.dumps(params)

    tool = tools.get(name)
    if not tool:
        return {'tool_call_id': tool_call.get('id', ''), 'content': f'Error: Tool "{name}" not found.'}

    spec = tool.get('spec', {})
    tool_type = tool.get('type', '')
    direct_tool = tool.get('direct', False)
    allowed_params = spec.get('parameters', {}).get('properties', {}).keys()
    params = {key: value for key, value in params.items() if key in allowed_params}

    try:
        if direct_tool:
            if not event_caller:
                result = 'Error: Browser session is not connected for this direct tool.'
            else:
                result = await event_caller(
                    {
                        'type': 'execute:tool',
                        'data': {
                            'id': str(uuid4()),
                            'name': name,
                            'params': params,
                            'server': tool.get('server', {}),
                            'session_id': metadata.get('session_id'),
                        },
                    }
                )
        else:
            function = await get_updated_tool_function(
                function=tool['callable'],
                extra_params={
                    '__messages__': form_data.get('messages', []),
                    '__files__': metadata.get('files', []),
                },
            )
            result = await function(**params)
    except Exception as e:
        result = {'error': str(e)}

    terminal_file_result = build_terminal_file_tool_result(name, params, result, tool, metadata)
    if terminal_file_result:
        result = terminal_file_result

    result, files, embeds = await process_tool_result(
        request,
        name,
        result,
        tool_type,
        direct_tool,
        metadata,
        user,
    )

    await terminal_event_handler(name, params, result, event_emitter)

    return {
        'tool_call_id': tool_call.get('id', ''),
        'content': tool_result_content(result),
        **({'files': files} if files else {}),
        **({'embeds': embeds} if embeds else {}),
    }


async def drain_approved_tool_calls(request, form_data, user, model, metadata) -> bool:
    chat_id = metadata.get('chat_id')
    assistant_message_id = metadata.get('assistant_message_id')
    # Only a resume/continue payload re-enters an existing message; other paths mint a fresh id with nothing to drain.
    if not is_saved_chat_id(chat_id) or not assistant_message_id:
        return False

    message_id = metadata.get('message_id') or assistant_message_id
    message = await Chats.get_message_by_id_and_message_id(chat_id, message_id)
    output = message.get('output') if message else None
    if not isinstance(output, list):
        return False

    result_call_ids = {
        item.get('call_id') for item in output if item.get('type') == 'function_call_output' and item.get('call_id')
    }
    approved_calls = [
        item
        for item in output
        if item.get('type') == 'function_call'
        and item.get('call_id')
        and item.get('status') == 'queued'
        and item.get('approved') is True
        and item.get('call_id') not in result_call_ids
    ]
    if not approved_calls:
        if metadata.get('params', {}).get('tool_approval_mode', 'full') == 'ask' and any(
            item.get('type') == 'function_call'
            and item.get('name') != 'ask_user'
            and (item.get('call_id') or item.get('id'))
            and item.get('status') == 'queued'
            and item.get('approved') is not True
            and (item.get('call_id') or item.get('id')) not in result_call_ids
            for item in output
        ):
            event_emitter, _ = await get_event_emitter_and_caller(metadata)
            await pause_for_tool_approval(chat_id, message_id, output, form_data, metadata)
            if event_emitter:
                await event_emitter({'type': 'chat:completion', 'data': {'done': False, 'output': output}})
            return True
        return False

    event_emitter, event_caller = await get_event_emitter_and_caller(metadata)
    changed = False
    for item in approved_calls:
        if item.get('name') == 'ask_user':
            item['status'] = 'pending'
            item.pop('approved', None)
            changed = True
            continue

        tool_call = {
            'id': item.get('call_id', ''),
            'type': 'function',
            'function': {
                'name': item.get('name', ''),
                'arguments': item.get('arguments', '{}'),
            },
        }
        result = await execute_tool_call_for_output(
            request,
            form_data,
            user,
            metadata,
            event_caller,
            event_emitter,
            tool_call,
        )
        item['arguments'] = tool_call.get('function', {}).get('arguments', '{}')
        output_parts = [{'type': 'input_text', 'text': result.get('content', '')}]
        item['status'] = 'failed' if _is_tool_result_error(result.get('content', '')) else 'completed'
        display_files = []
        for file_item in result.get('files', []):
            if file_item.get('type') == 'image' and file_item.get('url', '').startswith('data:'):
                output_parts.append({'type': 'input_image', 'image_url': file_item['url']})
            else:
                display_files.append(file_item)

        output.append(
            {
                'type': 'function_call_output',
                'id': output_id('fco'),
                'call_id': result.get('tool_call_id', ''),
                'output': output_parts,
                'status': item['status'],
                **({'files': display_files} if display_files else {}),
                **({'embeds': result.get('embeds')} if result.get('embeds') else {}),
            }
        )
        changed = True

    if changed:
        result_call_ids = {
            item.get('call_id') for item in output if item.get('type') == 'function_call_output' and item.get('call_id')
        }
        if metadata.get('params', {}).get('tool_approval_mode', 'full') == 'ask' and any(
            item.get('type') == 'function_call'
            and item.get('name') != 'ask_user'
            and (item.get('call_id') or item.get('id'))
            and item.get('status') == 'queued'
            and item.get('approved') is not True
            and (item.get('call_id') or item.get('id')) not in result_call_ids
            for item in output
        ):
            await pause_for_tool_approval(chat_id, message_id, output, form_data, metadata)
            result_call_ids = {
                item.get('call_id')
                for item in output
                if item.get('type') == 'function_call_output' and item.get('call_id')
            }
        paused = any(
            item.get('type') == 'function_call'
            and item.get('call_id')
            and item.get('status') in {'pending', 'queued', 'requires_approval'}
            and item.get('call_id') not in result_call_ids
            for item in output
        )
        if not paused:
            output.append(
                {
                    'type': 'message',
                    'id': output_id('msg'),
                    'status': 'in_progress',
                    'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': ''}],
                }
            )

        await Chats.upsert_message_to_chat_by_id_and_message_id(
            chat_id,
            message_id,
            {'done': False, 'output': output},
            touch=False,
        )
        if event_emitter:
            await event_emitter(
                {
                    'type': 'chat:completion',
                    'data': {
                        'done': False,
                        'output': output,
                    },
                }
            )

        db_messages = await load_messages_from_db(chat_id, metadata.get('user_message_id'))
        if db_messages:
            assistant_message = await Chats.get_message_by_id_and_message_id(chat_id, message_id)
            if assistant_message:
                db_messages.append({k: v for k, v in assistant_message.items() if k in MESSAGE_REPLAY_KEYS})
            for message in db_messages:
                output = message.get('output')
                # reasoning_details can be model/provider-bound, so only replay them
                # for output produced by the same model.
                if (
                    message.get('role') == 'assistant'
                    and message.get('model') != model['id']
                    and isinstance(output, list)
                ):
                    message['output'] = strip_reasoning_details(output)

            form_data['messages'] = process_messages_with_output(
                db_messages,
                reasoning_format=get_reasoning_format(model),
            )
            form_data['messages'] = sanitize_tool_pairs(form_data['messages'])

        if not paused and ENABLE_PLUGINS:
            filter_functions = await get_filter_functions(request, model, metadata.get('filter_ids', []))
            if filter_functions:
                filtered_form_data, _ = await process_filter_functions(
                    request=request,
                    filter_context=get_filter_context(request),
                    filter_functions=filter_functions,
                    filter_type='request',
                    form_data=form_data,
                    extra_params={
                        '__event_emitter__': event_emitter,
                        '__event_call__': event_caller,
                        '__user__': user.model_dump() if isinstance(user, UserModel) else {},
                        '__metadata__': metadata,
                        '__oauth_token__': await get_system_oauth_token(request, user),
                        '__request__': request,
                        '__model__': model,
                        '__chat_id__': metadata.get('chat_id'),
                        '__message_id__': metadata.get('message_id'),
                    },
                )
                if filtered_form_data is not form_data:
                    form_data.clear()
                    form_data.update(filtered_form_data)

        if not paused:
            normalize_messages_for_model(form_data)

        return paused

    return False


async def pause_for_tool_approval(chat_id: str, message_id: str, output: list[dict], form_data: dict, metadata: dict):
    result_call_ids = {
        item.get('call_id') for item in output if item.get('type') == 'function_call_output' and item.get('call_id')
    }
    has_pending_approval = False
    for item in output:
        if item.get('type') == 'function_call' and not item.get('call_id') and item.get('id'):
            item['call_id'] = item['id']

        if (
            item.get('type') == 'function_call'
            and item.get('call_id')
            and item.get('call_id') not in result_call_ids
            and item.get('status') != 'rejected'
        ):
            if not has_pending_approval:
                item['status'] = 'pending'
                has_pending_approval = True
            elif item.get('status') == 'in_progress':
                item['status'] = 'queued'

    await Chats.upsert_message_to_chat_by_id_and_message_id(
        chat_id,
        message_id,
        {
            'done': False,
            'output': output,
            'meta': {
                **(metadata.get('tool_approval') or {}),
                'session_id': metadata.get('session_id'),
                'tool_ids': metadata.get('tool_ids') or [],
                'skill_ids': metadata.get('skill_ids') or [],
                'terminal_id': metadata.get('terminal_id'),
                'tool_servers': metadata.get('tool_servers'),
                'filter_ids': metadata.get('filter_ids') or [],
                'features': metadata.get('features') or {},
                'variables': metadata.get('variables') or {},
                'files': metadata.get('files') or [],
                'params': metadata.get('params') or {},
            },
        },
        touch=False,
    )


def get_response_data(response):
    if isinstance(response, list) and len(response) == 1:
        # If the response is a single-item list, unwrap it #17213
        response = response[0]

    if isinstance(response, JSONResponse):
        if isinstance(response.body, bytes):
            try:
                response_data = JSONCodec.loads(response.body.decode('utf-8', 'replace'))
            except JSONCodec.JSONDecodeError:
                response_data = {'error': {'detail': 'Invalid JSON response'}}
        else:
            response_data = response
    elif isinstance(response, dict):
        response_data = response
    else:
        response_data = None

    return response, response_data


def merge_events_into_response(response_data, events):
    if events and isinstance(events, list):
        extra_response = {}
        for event in events:
            if isinstance(event, dict):
                extra_response.update(event)
            else:
                extra_response[event] = True

        return {
            **extra_response,
            **response_data,
        }
    return response_data


def build_response_object(response, response_data):
    if isinstance(response, dict):
        return response_data
    if isinstance(response, JSONResponse):
        return JSONResponse(
            content=response_data,
            headers=response.headers,
            status_code=response.status_code,
        )
    return response


def update_assistant_message_from_stream(assistant_message, raw):
    line = raw.decode('utf-8', 'replace') if isinstance(raw, bytes) else raw
    if not isinstance(line, str):
        return

    def append_output_text(item, text):
        parts = item.setdefault('content', [])
        if parts and parts[-1].get('type') == 'output_text':
            parts[-1]['text'] += text
        else:
            parts.append({'type': 'output_text', 'text': text})

    for raw_part in line.splitlines():
        part = raw_part.removeprefix('data:').strip()
        if not part or part == '[DONE]':
            continue

        try:
            data = JSONCodec.loads(part)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        if data.get('type', '').startswith('response.'):
            output, meta = handle_responses_streaming_event(data, assistant_message.get('output', []))
            if output:
                assistant_message['output'] = output
            if meta and meta.get('usage'):
                assistant_message['usage'] = merge_usage(assistant_message.get('usage'), meta['usage'])
            continue

        raw_usage = data.get('usage', {}) or {}
        raw_usage.update(data.get('timings', {}))
        if raw_usage:
            assistant_message['usage'] = merge_usage(assistant_message.get('usage'), raw_usage)

        for choice in data.get('choices', []):
            delta = choice.get('delta', {}) or {}
            content = delta.get('content')
            reasoning_content = delta.get('reasoning_content') or delta.get('reasoning') or delta.get('thinking')

            if reasoning_content:
                output = assistant_message.setdefault('output', [])
                if not output or output[-1].get('type') != 'reasoning':
                    output.append(
                        {
                            'type': 'reasoning',
                            'id': output_id('r'),
                            'status': 'in_progress',
                            'start_tag': '<think>',
                            'end_tag': '</think>',
                            'attributes': {'type': 'reasoning_content'},
                            'content': [],
                            'summary': None,
                            'started_at': time.time(),
                        }
                    )

                append_output_text(output[-1], reasoning_content)

            if content:
                output = assistant_message.get('output')
                if output:
                    if output[-1].get('type') == 'reasoning':
                        output[-1]['status'] = 'completed'
                        output[-1]['ended_at'] = time.time()
                        output[-1]['duration'] = int(output[-1]['ended_at'] - output[-1]['started_at'])

                    if not output or output[-1].get('type') != 'message':
                        output.append(
                            {
                                'type': 'message',
                                'id': output_id('msg'),
                                'status': 'in_progress',
                                'role': 'assistant',
                                'content': [],
                            }
                        )

                    append_output_text(output[-1], content)

                assistant_message['content'] = assistant_message.get('content', '') + content


async def get_system_oauth_token(request, user):
    """Get the system OAuth token for a user.

    Primary path: use the oauth_session_id cookie (browser requests).
    Fallback: look up the user's most recent OAuth session from the DB
    (covers automations, API calls, and other cookie-less contexts).
    """
    oauth_token = None
    try:
        oauth_session_id = request.cookies.get('oauth_session_id', None)
        if oauth_session_id:
            oauth_token = await request.app.state.oauth_manager.get_oauth_token(
                user.id,
                oauth_session_id,
            )

        # Fallback: no cookie (automation, API key, etc.) — use most recent session
        if oauth_token is None:
            from open_webui.models.oauth_sessions import OAuthSessions

            sessions = await OAuthSessions.get_sessions_by_user_id(user.id)
            # Filter out MCP-provider sessions — their token refresh is handled
            # separately by oauth_client_manager.  Passing them to the SSO
            # oauth_manager causes a failed refresh and session deletion (#24618).
            sessions = [s for s in sessions if not (s.provider or '').startswith('mcp:')]
            if sessions:
                best = max(sessions, key=lambda s: s.updated_at)
                oauth_token = await request.app.state.oauth_manager.get_oauth_token(
                    user.id,
                    best.id,
                )
    except Exception as e:
        log.error(f'Error getting OAuth token: {e}')
    return oauth_token


async def background_tasks_handler(ctx):
    request = ctx['request']
    form_data = ctx['form_data']
    user = ctx['user']
    metadata = ctx['metadata']
    tasks = ctx['tasks']
    event_emitter = ctx['event_emitter']

    message = None
    messages = []

    if is_saved_chat_id(metadata.get('chat_id')):
        messages_map = await Chats.get_messages_map_by_chat_id(metadata['chat_id'])
        if not messages_map:
            # Chat was deleted while the response was streaming — skip background tasks
            return
        message = messages_map.get(metadata['message_id'])

        message_list = get_message_list(messages_map, metadata['message_id'])

        # Remove details tags and files from the messages.
        # as get_message_list creates a new list, it does not affect
        # the original messages outside of this handler

        messages = []
        for message in message_list:
            content = message.get('content', '')
            if isinstance(content, list):
                for item in content:
                    if item.get('type') == 'text':
                        content = item['text']
                        break

            if isinstance(content, str):
                content = re.sub(
                    r'<details\b[^>]*>.*?<\/details>|!\[.*?\]\(.*?\)',
                    '',
                    content,
                    flags=re.S | re.I,
                ).strip()

            messages.append(
                {
                    **message,
                    'role': message.get('role', 'assistant'),  # Safe fallback for missing role
                    'content': content,
                }
            )
    else:
        # Local temp chat, get the model and message from the form_data
        message = get_last_user_message_item(form_data.get('messages', []))
        messages = form_data.get('messages', [])
        if message:
            message['model'] = form_data.get('model')

    if message and 'model' in message:
        if tasks and messages:
            if TASKS.FOLLOW_UP_GENERATION in tasks and tasks[TASKS.FOLLOW_UP_GENERATION]:
                res = await generate_follow_ups(
                    request,
                    {
                        'model': message['model'],
                        'messages': messages,
                        'message_id': metadata['message_id'],
                        'chat_id': metadata['chat_id'],
                    },
                    user,
                )

                if res and isinstance(res, dict):
                    if len(res.get('choices', [])) == 1:
                        response_message = res.get('choices', [])[0].get('message', {})

                        follow_ups_string = response_message.get('content') or response_message.get(
                            'reasoning_content', ''
                        )
                    else:
                        follow_ups_string = ''

                    follow_ups_string = follow_ups_string[
                        follow_ups_string.find('{') : follow_ups_string.rfind('}') + 1
                    ]

                    try:
                        follow_ups = JSONCodec.loads(follow_ups_string).get('follow_ups', [])
                        await event_emitter(
                            {
                                'type': 'chat:message:follow_ups',
                                'data': {
                                    'follow_ups': follow_ups,
                                },
                            }
                        )

                        if is_saved_chat_id(metadata.get('chat_id')):
                            await Chats.upsert_message_to_chat_by_id_and_message_id(
                                metadata['chat_id'],
                                metadata['message_id'],
                                {
                                    'followUps': follow_ups,
                                },
                                touch=False,
                            )

                    except Exception as e:
                        pass

            if is_saved_chat_id(metadata.get('chat_id')):  # Only update titles and tags for saved chats
                if TASKS.TITLE_GENERATION in tasks:
                    user_message = get_last_user_message(messages)
                    if user_message and len(user_message) > 100:
                        user_message = user_message[:100] + '...'

                    title = None
                    if tasks[TASKS.TITLE_GENERATION]:
                        res = await generate_title(
                            request,
                            {
                                'model': message['model'],
                                'messages': messages,
                                'chat_id': metadata['chat_id'],
                            },
                            user,
                        )

                        if res and isinstance(res, dict):
                            if len(res.get('choices', [])) == 1:
                                response_message = res.get('choices', [])[0].get('message', {})

                                title_string = (
                                    response_message.get('content')
                                    or response_message.get(
                                        'reasoning_content',
                                    )
                                    or message.get('content', user_message)
                                )
                            else:
                                title_string = ''

                            title_string = title_string[title_string.find('{') : title_string.rfind('}') + 1]

                            try:
                                title = JSONCodec.loads(title_string).get('title', user_message)
                            except Exception as e:
                                title = ''

                            if not title:
                                title = messages[0].get('content', user_message)

                            await Chats.update_chat_title_by_id(metadata['chat_id'], title)

                            await event_emitter(
                                {
                                    'type': 'chat:title',
                                    'data': title,
                                }
                            )

                    if title == None and len(messages) == 2 and (not messages_map or len(messages_map) <= 2):
                        title = messages[0].get('content', user_message)

                        await Chats.update_chat_title_by_id(metadata['chat_id'], title)

                        await event_emitter(
                            {
                                'type': 'chat:title',
                                'data': message.get('content', user_message),
                            }
                        )

                if TASKS.TAGS_GENERATION in tasks and tasks[TASKS.TAGS_GENERATION]:
                    res = await generate_chat_tags(
                        request,
                        {
                            'model': message['model'],
                            'messages': messages,
                            'chat_id': metadata['chat_id'],
                        },
                        user,
                    )

                    if res and isinstance(res, dict):
                        if len(res.get('choices', [])) == 1:
                            response_message = res.get('choices', [])[0].get('message', {})

                            tags_string = response_message.get('content') or response_message.get(
                                'reasoning_content', ''
                            )
                        else:
                            tags_string = ''

                        tags_string = tags_string[tags_string.find('{') : tags_string.rfind('}') + 1]

                        try:
                            tags = JSONCodec.loads(tags_string).get('tags', [])
                            await Chats.update_chat_tags_by_id(metadata['chat_id'], tags, user)

                            await event_emitter(
                                {
                                    'type': 'chat:tags',
                                    'data': tags,
                                }
                            )
                        except Exception as e:
                            pass

        try:
            await run_post_chat_memory_extractor(
                request=request,
                user=user,
                model=ctx.get('model') or {},
                metadata=metadata,
                messages=messages,
            )
        except Exception as e:
            log.debug(f'Post-chat memory extractor error: {e}')

        if messages:
            await review_memory_after_turn(
                request=request,
                user=user,
                model=ctx['model'],
                metadata=metadata,
                form_data=form_data,
                assistant_message=ctx.get('assistant_message') or {},
                messages=messages,
            )


async def outlet_filter_handler(ctx):
    """Run outlet filters inline after chat completion.

    Replaces the separate POST /api/chat/completed round-trip.
    Persists outlet-modified content to DB and emits a chat:outlet event
    so the frontend can sync its in-memory state. Returns immediately when
    the model has no filters.

    For temp/API chats, messages are built from form_data plus ctx['assistant_message'].
    """
    request = ctx['request']
    user = ctx['user']
    model = ctx['model']
    metadata = ctx['metadata']
    event_emitter = ctx.get('event_emitter')
    event_caller = ctx.get('event_caller')

    chat_id = metadata.get('chat_id', '')
    message_id = metadata.get('message_id')

    if not chat_id and not ctx.get('assistant_message'):
        return

    if not message_id:
        message_id = output_id('msg')

    is_unsaved_chat = not is_saved_chat_id(chat_id)
    try:
        filter_functions = (
            await get_filter_functions(request, model, metadata.get('filter_ids', [])) if ENABLE_PLUGINS else []
        )
        model_id = model.get('id') if isinstance(model, dict) else model
        models = request.app.state.MODELS
        has_pipeline_outlet_filters = bool(
            (isinstance(model, dict) and 'pipeline' in model) or get_sorted_filters(model_id, models)
        )
        if not filter_functions and not has_pipeline_outlet_filters:
            return

        messages_map = None

        if is_unsaved_chat:
            form_messages = ctx.get('form_data', {}).get('messages', [])
            assistant_message = ctx.get('assistant_message', {})

            message_list = [
                {
                    'role': m.get('role'),
                    'content': m.get('content') or get_output_text(m.get('output')),
                }
                for m in form_messages
            ]

            if assistant_message:
                message_list.append(
                    {
                        'id': message_id,
                        'role': 'assistant',
                        **assistant_message,
                    }
                )

            if not message_list:
                return
        else:
            messages_map = await Chats.get_messages_map_by_chat_id(chat_id)
            if not messages_map:
                return

            message_list = get_message_list(messages_map, message_id)
            if not message_list:
                return

        outlet_data = {
            'model': model_id,
            'messages': [
                {
                    'id': m.get('id'),
                    'role': m.get('role'),
                    'content': m.get('content') or get_output_text(m.get('output')),
                    'info': m.get('info'),
                    'timestamp': m.get('timestamp'),
                    # Deepcopy so in-place filter mutations do not alias messages_map's baseline
                    **({'output': copy.deepcopy(m['output'])} if m.get('output') else {}),
                    **({'usage': m['usage']} if m.get('usage') else {}),
                    **({'sources': m['sources']} if m.get('sources') else {}),
                }
                for m in message_list
            ],
            'filter_ids': metadata.get('filter_ids', []),
            'chat_id': chat_id,
            'session_id': metadata.get('session_id'),
            'id': message_id,
        }

        # Pipeline outlet filters
        try:
            outlet_data = await process_pipeline_outlet_filter(request, outlet_data, user, models)
        except Exception as e:
            log.debug('Pipeline outlet filter error: %s', e)

        # Function outlet filters
        extra_params = {
            '__event_emitter__': event_emitter,
            '__event_call__': event_caller,
            '__user__': user.model_dump() if isinstance(user, UserModel) else {},
            '__metadata__': metadata,
            '__request__': request,
            '__model__': model,
        }

        if filter_functions:
            outlet_result, _ = await process_filter_functions(
                request=request,
                filter_context=None,
                filter_functions=filter_functions,
                filter_type='outlet',
                form_data=outlet_data,
                extra_params=extra_params,
            )
        else:
            outlet_result = outlet_data

        if outlet_result and outlet_result.get('messages'):
            if not is_unsaved_chat and messages_map:
                for message in outlet_result['messages']:
                    outlet_message_id = message.get('id')
                    if outlet_message_id and outlet_message_id in messages_map:
                        original_message = messages_map[outlet_message_id]
                        original_content = original_message.get('content') or get_output_text(
                            original_message.get('output')
                        )
                        message_content = message.get('content') or get_output_text(message.get('output'))
                        content_changed = original_content != message_content
                        output_changed = message.get('output') and message.get('output') != original_message.get(
                            'output'
                        )
                        if content_changed or output_changed:
                            message_update = {
                                'originalContent': original_content,
                                **({'output': message['output']} if output_changed else {}),
                            }
                            if content_changed:
                                message_update['content'] = message_content or ''
                            await Chats.upsert_message_to_chat_by_id_and_message_id(
                                chat_id,
                                outlet_message_id,
                                message_update,
                            )

            if event_emitter:
                await event_emitter(
                    {
                        'type': 'chat:outlet',
                        'data': {'messages': outlet_result['messages']},
                    }
                )
    except Exception as e:
        log.debug('Error running outlet filters: %s', e)


async def non_streaming_chat_response_handler(response, ctx):
    request = ctx['request']

    user = ctx['user']
    metadata = ctx['metadata']
    events = ctx['events']

    event_emitter = ctx['event_emitter']

    response, response_data = get_response_data(response)
    if response_data is None:
        return response

    chat_id = metadata.get('chat_id') or ''
    save_to_chat = is_saved_chat_id(chat_id)

    if event_emitter:
        try:
            if 'error' in response_data:
                error = response_data.get('error')

                if isinstance(error, dict):
                    error = error.get('detail', error)
                else:
                    error = str(error)

                log.error('Provider returned error (non-streaming): %s', error)

                if save_to_chat:
                    await Chats.upsert_message_to_chat_by_id_and_message_id(
                        metadata['chat_id'],
                        metadata['message_id'],
                        {
                            'error': {'content': error},
                        },
                    )
                if isinstance(error, str) or isinstance(error, dict):
                    await event_emitter(
                        {
                            'type': 'chat:message:error',
                            'data': {'error': {'content': error}},
                        }
                    )

            if 'selected_model_id' in response_data and save_to_chat:
                await Chats.upsert_message_to_chat_by_id_and_message_id(
                    metadata['chat_id'],
                    metadata['message_id'],
                    {
                        'selectedModelId': response_data['selected_model_id'],
                    },
                    touch=False,
                )

            choices = response_data.get('choices', [])
            response_output = response_data.get('output')
            content = choices[0].get('message', {}).get('content') if choices else ''

            if choices and (content or response_output):
                if content or response_output:
                    await event_emitter(
                        {
                            'type': 'chat:completion',
                            'data': response_data,
                        }
                    )

                    title = await Chats.get_chat_title_by_id(metadata['chat_id']) if save_to_chat else ''

                    # Use output from backend if provided (OR-compliant backends),
                    # otherwise generate from response content
                    if not response_output:
                        choice_message = choices[0].get('message', {})
                        reasoning_content = choice_message.get('reasoning_content') or choice_message.get('reasoning')
                        reasoning_details = get_reasoning_details(choice_message)
                        response_output = []
                        if reasoning_content or reasoning_details:
                            reasoning_item = {
                                'type': 'reasoning',
                                'id': output_id('r'),
                                'status': 'completed',
                                'start_tag': '<think>',
                                'end_tag': '</think>',
                                'attributes': {'type': 'reasoning_content'},
                                'content': (
                                    [{'type': 'output_text', 'text': reasoning_content}] if reasoning_content else []
                                ),
                                'summary': None,
                            }
                            if reasoning_details:
                                reasoning_item['reasoning_details'] = (
                                    reasoning_details if isinstance(reasoning_details, list) else [reasoning_details]
                                )
                            response_output.append(reasoning_item)
                        response_output.append(
                            {
                                'type': 'message',
                                'id': output_id('msg'),
                                'status': 'completed',
                                'role': 'assistant',
                                'content': [{'type': 'output_text', 'text': content}],
                            }
                        )

                    await event_emitter(
                        {
                            'type': 'chat:completion',
                            'data': {
                                'done': True,
                                'output': response_output,
                                'title': title,
                            },
                        }
                    )

                    # Save message in the database
                    usage = normalize_usage(response_data.get('usage', {}) or {})

                    if save_to_chat:
                        await Chats.upsert_message_to_chat_by_id_and_message_id(
                            metadata['chat_id'],
                            metadata['message_id'],
                            {
                                'done': True,
                                'role': 'assistant',
                                'output': response_output,
                                **({'usage': usage} if usage else {}),
                            },
                        )

                    await publish_chat_finished_event(request, user, metadata, title, content, response_output)

                    ctx['assistant_message'] = {
                        'content': content,
                        'output': response_output,
                        **({'usage': usage} if usage else {}),
                    }
                    await outlet_filter_handler(ctx)
                    await background_tasks_handler(ctx)

            response = build_response_object(response, merge_events_into_response(response_data, events))
        except Exception as e:
            log.debug('Error occurred while processing request: %s', e)
            chat_id = metadata.get('chat_id')
            if getattr(request.state, 'internal', False) is not True and chat_id and is_saved_chat_id(chat_id):
                webui_url = await Config.get('webui.url')
                await publish_event(
                    request,
                    EVENTS.CHAT_FAILED,
                    actor=user,
                    subject_id=chat_id,
                    subject_type='chat',
                    data={
                        'user_id': user.id,
                        'chat_id': chat_id,
                        'message_id': metadata.get('message_id'),
                        'model_id': metadata.get('model_id'),
                        'url': f'{webui_url}/c/{chat_id}' if webui_url else f'/c/{chat_id}',
                        'message': str(e),
                    },
                    message='Chat failed',
                )
            pass

        return response

    choices = response_data.get('choices', [])
    output = response_data.get('output')
    content = choices[0].get('message', {}).get('content') if choices else ''
    if ENABLE_API_OUTLET_FILTERS and (content or output):
        usage = normalize_usage(response_data.get('usage', {}) or {})
        ctx['assistant_message'] = {
            **({'content': content} if content else {}),
            **({'output': output} if output else {}),
            **({'usage': usage} if usage else {}),
        }
        await outlet_filter_handler(ctx)

    if isinstance(response, dict):
        response = merge_events_into_response(response_data, events)

    return response


async def streaming_chat_response_handler(response, ctx):
    request = ctx['request']

    form_data = ctx['form_data']

    user = ctx['user']
    model = ctx['model']

    metadata = ctx['metadata']
    events = ctx['events']

    event_emitter = ctx['event_emitter']
    event_caller = ctx['event_caller']
    chat_id = metadata.get('chat_id') or ''
    save_to_chat = is_saved_chat_id(chat_id)

    extra_params = {
        '__event_emitter__': event_emitter,
        '__event_call__': event_caller,
        '__user__': user.model_dump() if isinstance(user, UserModel) else {},
        '__metadata__': metadata,
        '__oauth_token__': await get_system_oauth_token(request, user),
        '__request__': request,
        '__model__': model,
        '__chat_id__': metadata.get('chat_id'),
        '__message_id__': metadata.get('message_id'),
    }

    filter_functions = (
        await get_filter_functions(request, model, metadata.get('filter_ids', [])) if ENABLE_PLUGINS else []
    )

    # Standard streaming response handler
    # event_caller is optional — only needed for direct (client-side) tools
    # and pyodide code interpreter. Server-side tools work without it.
    if event_emitter:
        task_id = str(uuid4())  # Create a unique task ID.
        model_id = form_data.get('model', '')

        # Handle as a background task
        async def response_handler(response, events):
            filter_context = FilterContext()
            tag_scan_positions = {}
            tag_boundary_positions = {}
            response_stream_task_id = metadata.get('task_id') or metadata.get('message_id')

            def tag_output_handler(content_type, tags, output):
                """
                Detect special tags (reasoning, solution, code_interpreter) in streaming
                content and create corresponding OR-aligned output items directly.
                Operates on output items instead of content_blocks.

                Uses the text from the output items themselves for tag detection,
                eliminating state divergence between accumulated content and items.
                """
                end_flag = False

                def extract_attributes(tag_content):
                    """Extract attributes from a tag if they exist."""
                    attributes = {}
                    if not tag_content:
                        return attributes
                    matches = re.findall(r'(\w+)\s*=\s*"([^"]+)"', tag_content)
                    for key, value in matches:
                        attributes[key] = value
                    return attributes

                def get_last_text(out):
                    """Get text from last message item, or empty string."""
                    if out and out[-1].get('type') == 'message':
                        parts = out[-1].get('content', [])
                        if parts and parts[-1].get('type') == 'output_text':
                            return parts[-1].get('text', '')
                    return ''

                def set_last_text(out, text):
                    """Set text on last message item's output_text."""
                    if out and out[-1].get('type') == 'message':
                        parts = out[-1].get('content', [])
                        if parts and parts[-1].get('type') == 'output_text':
                            parts[-1]['text'] = text

                def get_scanned_length(item, text):
                    item_id = item.get('id')
                    if not item_id:
                        return 0

                    scanned_length = tag_scan_positions.get((item_id, content_type), 0)
                    return scanned_length if scanned_length <= len(text) else 0

                def save_scanned_length(item, text):
                    item_id = item.get('id')
                    if item_id:
                        tag_scan_positions[(item_id, content_type)] = len(text)

                def clear_scanned_length(item):
                    item_id = item.get('id')
                    if item_id:
                        tag_scan_positions.pop((item_id, content_type), None)
                        tag_boundary_positions.pop((item_id, content_type), None)

                def get_tag_boundaries(item, text, scanned_length):
                    """Index of the last '<', and of the last '>' or newline, before scanned_length."""
                    key = (item.get('id'), content_type)
                    scanned, last_open, last_boundary = tag_boundary_positions.get(key, (0, -1, -1))
                    if scanned > scanned_length:  # the item was rewritten, so the cached positions are stale
                        scanned, last_open, last_boundary = 0, -1, -1

                    if scanned < scanned_length:
                        # only text added since the last call can move either position
                        open_tag = text.rfind('<', scanned, scanned_length)
                        if open_tag != -1:
                            last_open = open_tag
                        boundary = max(
                            text.rfind('>', scanned, scanned_length),
                            text.rfind('\n', scanned, scanned_length),
                        )
                        if boundary != -1:
                            last_boundary = boundary
                        tag_boundary_positions[key] = (scanned_length, last_open, last_boundary)

                    return last_open, last_boundary

                # Map content_type to output item type
                output_type_map = {
                    'reasoning': 'reasoning',
                    'solution': 'message',  # solution tags just produce text
                    'code_interpreter': 'open_webui:code_interpreter',
                }
                output_item_type = output_type_map.get(content_type, content_type)

                last_type = output[-1].get('type', '') if output else ''

                if last_type == 'message':
                    # Use the output item's own text for tag detection
                    item = output[-1]
                    item_text = get_last_text(output)
                    scanned_length = get_scanned_length(item, item_text)
                    max_start_tag_length = max((len(start_tag) for start_tag, _ in tags), default=1)
                    search_start = max(0, scanned_length - max_start_tag_length + 1)

                    if scanned_length and any(
                        start_tag.startswith('<') and start_tag.endswith('>') for start_tag, _ in tags
                    ):
                        open_tag_start, last_tag_boundary = get_tag_boundaries(item, item_text, scanned_length)
                        if open_tag_start > last_tag_boundary:
                            search_start = min(search_start, open_tag_start)

                    for start_tag, end_tag in tags:
                        match = re.compile(_start_tag_pattern(start_tag)).search(item_text, search_start)
                        if match:
                            clear_scanned_length(item)
                            try:
                                attr_content = match.group(1) if match.group(1) else ''
                            except Exception:
                                attr_content = ''

                            attributes = extract_attributes(attr_content)

                            before_tag = item_text[: match.start()]
                            after_tag = item_text[match.end() :]

                            # Keep only text before the tag in the message
                            set_last_text(output, before_tag)

                            if not before_tag.strip():
                                # Remove empty message item
                                if output and output[-1].get('type') == 'message':
                                    output.pop()

                            # Append the new output item
                            if output_item_type == 'reasoning':
                                output.append(
                                    {
                                        'type': 'reasoning',
                                        'id': output_id('r'),
                                        'status': 'in_progress',
                                        'start_tag': start_tag,
                                        'end_tag': end_tag,
                                        'attributes': attributes,
                                        'content': [],
                                        'summary': None,
                                        'started_at': time.time(),
                                    }
                                )
                            elif output_item_type == 'open_webui:code_interpreter':
                                output.append(
                                    {
                                        'type': 'open_webui:code_interpreter',
                                        'id': output_id('ci'),
                                        'status': 'in_progress',
                                        'start_tag': start_tag,
                                        'end_tag': end_tag,
                                        'attributes': attributes,
                                        'lang': attributes.get('lang', 'python'),
                                        'code': '',
                                        'output': None,
                                        'started_at': time.time(),
                                    }
                                )
                            else:
                                # solution or other text-producing tag
                                output.append(
                                    {
                                        'type': 'message',
                                        'id': output_id('msg'),
                                        'status': 'in_progress',
                                        'role': 'assistant',
                                        'content': [{'type': 'output_text', 'text': ''}],
                                        '_tag_type': content_type,
                                        'start_tag': start_tag,
                                        'end_tag': end_tag,
                                        'attributes': attributes,
                                        'started_at': time.time(),
                                    }
                                )

                            if after_tag:
                                # Set the after_tag content on the new item
                                if output_item_type == 'reasoning':
                                    output[-1]['content'] = [{'type': 'output_text', 'text': after_tag}]
                                elif output_item_type == 'open_webui:code_interpreter':
                                    output[-1]['code'] = after_tag
                                else:
                                    set_last_text(output, after_tag)

                                _, recursive_end = tag_output_handler(content_type, tags, output)
                                if recursive_end:
                                    end_flag = True

                            break
                    else:
                        save_scanned_length(item, item_text)

                elif (
                    (last_type == 'reasoning' and content_type == 'reasoning')
                    or (last_type == 'open_webui:code_interpreter' and content_type == 'code_interpreter')
                    or (last_type == 'message' and output[-1].get('_tag_type') == content_type)
                ):
                    item = output[-1]
                    start_tag = item.get('start_tag', '')
                    end_tag = item.get('end_tag', '')

                    # Get the block content from the item itself
                    if last_type == 'reasoning':
                        parts = item.get('content', [])
                        block_content = ''
                        if parts and parts[-1].get('type') == 'output_text':
                            block_content = parts[-1].get('text', '')
                    elif last_type == 'open_webui:code_interpreter':
                        block_content = item.get('code', '')
                    else:
                        block_content = get_last_text(output)

                    scanned_length = get_scanned_length(item, block_content)
                    end_tag_search_start = max(0, scanned_length - max(len(end_tag), 1) + 1)

                    if block_content.find(end_tag, end_tag_search_start) != -1:
                        clear_scanned_length(item)
                        end_flag = True

                        # Strip start and end tags from content
                        start_tag_pattern = _start_tag_pattern(start_tag)
                        block_content = re.sub(start_tag_pattern, '', block_content).strip()

                        end_tag_pattern = rf'{re.escape(end_tag)}'
                        end_tag_regex = re.compile(end_tag_pattern, re.DOTALL)
                        split_content = end_tag_regex.split(block_content, maxsplit=1)

                        block_content = split_content[0].strip() if split_content else ''
                        leftover_content = split_content[1].strip() if len(split_content) > 1 else ''

                        if block_content:
                            # Update the item with final content
                            if last_type == 'reasoning':
                                item['content'] = [{'type': 'output_text', 'text': block_content}]
                                item['ended_at'] = time.time()
                                item['duration'] = int(item['ended_at'] - item['started_at'])
                                item['status'] = 'completed'
                            elif last_type == 'open_webui:code_interpreter':
                                item['code'] = block_content
                                item['ended_at'] = time.time()
                                item['duration'] = int(item['ended_at'] - item['started_at'])
                            else:
                                set_last_text(output, block_content)
                                item['ended_at'] = time.time()

                            # Reset by appending a new message item for leftover
                            output.append(
                                {
                                    'type': 'message',
                                    'id': output_id('msg'),
                                    'status': 'in_progress',
                                    'role': 'assistant',
                                    'content': [
                                        {
                                            'type': 'output_text',
                                            'text': leftover_content,
                                        }
                                    ],
                                }
                            )
                        else:
                            # Remove the block if content is empty
                            output.pop()
                            output.append(
                                {
                                    'type': 'message',
                                    'id': output_id('msg'),
                                    'status': 'in_progress',
                                    'role': 'assistant',
                                    'content': [
                                        {
                                            'type': 'output_text',
                                            'text': leftover_content,
                                        }
                                    ],
                                }
                            )
                    else:
                        save_scanned_length(item, block_content)

                return output, end_flag

            message = (
                await Chats.get_message_by_id_and_message_id(metadata['chat_id'], metadata['message_id'])
                if save_to_chat
                else None
            )

            tool_calls = []

            last_assistant_message = None
            try:
                if form_data['messages'][-1]['role'] == 'assistant':
                    last_assistant_message = get_last_assistant_message(form_data['messages'])
            except Exception as e:
                pass

            initial_content = (
                message.get('content', '') if message else last_assistant_message if last_assistant_message else ''
            )
            content_parts = [initial_content] if initial_content else []

            # Initialize output: use existing from message if continuing, else create new
            existing_output = message.get('output') if message else None
            prior_output = []
            if existing_output and metadata.get('assistant_message_id'):
                prior_output = list(existing_output)
                if (
                    prior_output
                    and prior_output[-1].get('type') == 'message'
                    and prior_output[-1].get('status') == 'in_progress'
                ):
                    msg_parts = prior_output[-1].get('content', [])
                    if not msg_parts or (len(msg_parts) == 1 and not msg_parts[0].get('text', '').strip()):
                        prior_output.pop()
                output = []
                content_parts = []
            elif existing_output:
                output = existing_output
            else:
                # Only create an initial message item if there is content to initialize with
                if initial_content:
                    output = [
                        {
                            'type': 'message',
                            'id': output_id('msg'),
                            'status': 'in_progress',
                            'role': 'assistant',
                            'content': [{'type': 'output_text', 'text': initial_content}],
                        }
                    ]
                else:
                    output = []

            usage = None
            last_response_id = None
            responses_web_search_status_signatures = {}
            latest_responses_stream_state = None
            last_responses_retry_cursor = {}

            def full_output():
                return prior_output + output if prior_output else output

            stream_response_backgrounds_completed = set()

            async def cleanup_stream_response(stream_response):
                if (
                    stream_response
                    and stream_response.background
                    and id(stream_response) not in stream_response_backgrounds_completed
                ):
                    stream_response_backgrounds_completed.add(id(stream_response))
                    await stream_response.background()

            def enrich_stream_error_with_cursor(error, include_saved_cursor=True, saved_cursor=None):
                if not isinstance(error, dict):
                    return error

                cursor_store = last_responses_retry_cursor if saved_cursor is None else saved_cursor
                cursor_payload = _responses_stream_cursor_from_error(
                    error,
                    latest_responses_stream_state,
                )

                if include_saved_cursor and cursor_store:
                    for key, value in cursor_store.items():
                        cursor_payload.setdefault(key, value)

                if (
                    cursor_payload.get('response_id')
                    and cursor_payload.get('response_sequence_number') is not None
                ):
                    cursor_store.clear()
                    cursor_store.update(cursor_payload)

                if cursor_payload.get('response_id') is not None and error.get('response_id') is None:
                    error['response_id'] = cursor_payload['response_id']

                if (
                    cursor_payload.get('response_sequence_number') is not None
                    and error.get('last_sequence_number') is None
                    and error.get('response_sequence_number') is None
                ):
                    error['last_sequence_number'] = cursor_payload['response_sequence_number']

                if cursor_payload.get('response_route_idx') is not None and error.get('response_route_idx') is None:
                    error['response_route_idx'] = cursor_payload['response_route_idx']

                if cursor_payload.get('response_route_url') is not None and error.get('response_route_url') is None:
                    error['response_route_url'] = cursor_payload['response_route_url']

                return error

            async def record_stream_error(error, saved_cursor=None, diagnostic_cursor=None):
                error = enrich_stream_error_with_cursor(error, saved_cursor=saved_cursor)
                log.error('Provider returned error (streaming): %s', error)
                current_output = full_output()
                cursor_payload = _responses_stream_cursor_from_error(
                    error,
                    latest_responses_stream_state,
                )
                cursor_store = last_responses_retry_cursor if saved_cursor is None else saved_cursor
                if cursor_store:
                    for key, value in cursor_store.items():
                        cursor_payload.setdefault(key, value)
                if diagnostic_cursor:
                    for key, value in diagnostic_cursor.items():
                        cursor_payload.setdefault(key, value)
                error_payload = {'content': error}
                message_update = {
                    'content': serialize_output(current_output),
                    'output': current_output,
                    'done': False,
                    'error': error_payload,
                }
                message_update.update(cursor_payload)
                try:
                    await Chats.upsert_message_to_chat_by_id_and_message_id(
                        metadata['chat_id'],
                        metadata['message_id'],
                        message_update,
                    )
                except Exception:
                    pass

                event_data = {
                    'content': serialize_output(full_output()),
                    'output': full_output(),
                    'done': False,
                    'error': error,
                }
                event_data.update(cursor_payload)
                await event_emitter(
                    {
                        'type': 'chat:completion',
                        'data': event_data,
                    }
                )

            async def try_resume_background_response(error, attempt):
                if not ENABLE_RESPONSES_API_BACKGROUND_RESUME:
                    return None

                if not isinstance(error, dict):
                    return None

                response_id = error.get('response_id')
                sequence_number = error.get('last_sequence_number')
                route_idx = error.get('response_route_idx')

                if not response_id or sequence_number is None:
                    return None

                if route_idx is None:
                    log.warning(
                        'Cannot resume background response without stored route '
                        '(chat_id=%s, message_id=%s, response_id=%s)',
                        metadata.get('chat_id'),
                        metadata.get('message_id'),
                        response_id,
                    )
                    return None

                model_id_for_resume = form_data.get('model')
                if not model_id_for_resume:
                    return None

                log.warning(
                    'Resuming background response stream '
                    '(attempt=%s, chat_id=%s, message_id=%s, response_id=%s, starting_after=%s, route_idx=%s)',
                    attempt,
                    metadata.get('chat_id'),
                    metadata.get('message_id'),
                    response_id,
                    sequence_number,
                    route_idx,
                )

                await event_emitter(
                    {
                        'type': 'status',
                        'data': {
                            'action': 'stream_resume',
                            'description': f'Upstream stream interrupted; resuming response {response_id}',
                            'done': False,
                            'hidden': True,
                        },
                    }
                )

                try:
                    return await resume_response_stream(
                        request=request,
                        model_id=model_id_for_resume,
                        response_id=response_id,
                        starting_after=sequence_number,
                        route_idx=route_idx,
                        user=user,
                    )
                except Exception as e:
                    log.warning(
                        'Background response resume setup failed; falling back '
                        '(chat_id=%s, message_id=%s, response_id=%s, route_idx=%s): %s',
                        metadata.get('chat_id'),
                        metadata.get('message_id'),
                        response_id,
                        route_idx,
                        e,
                        exc_info=True,
                    )
                    error['background_resume_error'] = {
                        'code': e.__class__.__name__,
                        'message': str(e),
                        'type': 'stream_resume_error',
                    }
                    return None

            async def prepare_contextual_stream_retry(error, attempt, max_attempts, base_form_data):
                nonlocal content_parts
                nonlocal usage
                nonlocal output
                nonlocal prior_output
                nonlocal last_response_id

                retry_output = _clean_output_for_contextual_retry(full_output())
                retry_messages = convert_output_to_messages(
                    retry_output,
                    raw=True,
                    reasoning_format=get_reasoning_format(model),
                )

                web_search_resume_message = convert_web_search_output_to_resume_message(retry_output)
                if web_search_resume_message:
                    retry_messages.append(web_search_resume_message)

                retry_messages.append(_build_stream_resume_instruction(error, retry_output))

                retry_form_data = copy.deepcopy(base_form_data)
                retry_form_data['stream'] = True
                retry_form_data['metadata'] = metadata
                retry_form_data.pop('previous_response_id', None)
                retry_form_data['messages'] = [
                    *base_form_data.get('messages', []),
                    *retry_messages,
                ]

                prior_output = retry_output
                content_parts[:] = [serialize_output(prior_output)] if prior_output else []
                usage = None
                output = []
                last_response_id = None
                responses_web_search_status_signatures.clear()
                tool_calls.clear()

                log.warning(
                    'Falling back to contextual response retry '
                    '(attempt %s/%s, chat_id=%s, message_id=%s): %s',
                    attempt,
                    max_attempts,
                    metadata.get('chat_id'),
                    metadata.get('message_id'),
                    error,
                )

                await event_emitter(
                    {
                        'type': 'status',
                        'data': {
                            'action': 'stream_retry',
                            'description': f'Could not resume upstream response; continuing from partial context ({attempt}/{max_attempts})',
                            'done': False,
                            'hidden': True,
                        },
                    }
                )

                current_output = full_output()
                if metadata.get('chat_id') and not metadata['chat_id'].startswith('channel:'):
                    try:
                        await Chats.upsert_message_to_chat_by_id_and_message_id(
                            metadata['chat_id'],
                            metadata['message_id'],
                            {
                                'content': serialize_output(current_output),
                                'output': current_output,
                                'error': None,
                            },
                        )
                    except Exception:
                        pass

                await event_emitter(
                    {
                        'type': 'chat:completion',
                        'data': {
                            'content': serialize_output(current_output),
                            'output': current_output,
                        },
                    }
                )

                return retry_form_data

            async def emit_responses_web_search_statuses(items: list):
                for item in items or []:
                    status_data = build_responses_web_search_status(item)
                    if not status_data:
                        continue

                    status_key = status_data.get('id') or item.get('id')
                    if not status_key:
                        continue

                    signature = json.dumps(status_data, sort_keys=True, ensure_ascii=False)
                    if responses_web_search_status_signatures.get(status_key) == signature:
                        continue

                    responses_web_search_status_signatures[status_key] = signature
                    await event_emitter(
                        {
                            'type': 'status',
                            'data': status_data,
                        }
                    )

            def get_message_error_content(error):
                if isinstance(error, HTTPException):
                    error = error.detail
                elif isinstance(error, dict):
                    error = error.get('detail', error)
                else:
                    error = str(error)

                return error if isinstance(error, (str, dict)) else str(error)

            async def emit_message_error(error_content):
                if save_to_chat:
                    await Chats.upsert_message_to_chat_by_id_and_message_id(
                        metadata['chat_id'],
                        metadata['message_id'],
                        {'error': {'content': error_content}},
                    )
                await event_emitter(
                    {
                        'type': 'chat:message:error',
                        'data': {'error': {'content': error_content}},
                    }
                )

            reasoning_tags_param = metadata.get('params', {}).get('reasoning_tags')
            DETECT_REASONING_TAGS = reasoning_tags_param is not False

            # Legacy tool-calling only: native FC gets execute_code as a builtin tool.
            # Same five authz gates as utils/tools.py get_builtin_tools.
            features = metadata.get('features', {}) or {}
            model_capabilities = model.get('info', {}).get('meta', {}).get('capabilities') or {}
            builtin_tools_meta = model.get('info', {}).get('meta', {}).get('builtinTools', {})
            DETECT_CODE_INTERPRETER = (
                metadata.get('params', {}).get('function_calling') == 'legacy'
                and bool(features.get('code_interpreter'))
                and builtin_tools_meta.get('code_interpreter', True)
                and await Config.get('code_interpreter.enable')
                and model_capabilities.get('code_interpreter', True)
                and (
                    getattr(user, 'role', None) == 'admin'
                    or await has_permission(
                        getattr(user, 'id', ''),
                        'features.code_interpreter',
                        await Config.get('user.permissions'),
                    )
                )
            )

            reasoning_tags = []
            if DETECT_REASONING_TAGS:
                if isinstance(reasoning_tags_param, list) and len(reasoning_tags_param) == 2:
                    reasoning_tags = [(reasoning_tags_param[0], reasoning_tags_param[1])]
                else:
                    reasoning_tags = DEFAULT_REASONING_TAGS

            try:
                for event in events:
                    await event_emitter(
                        {
                            'type': 'chat:completion',
                            'data': event,
                        }
                    )

                    # Save message in the database
                    if save_to_chat:
                        await Chats.upsert_message_to_chat_by_id_and_message_id(
                            metadata['chat_id'],
                            metadata['message_id'],
                            {
                                **event,
                            },
                        )

                if (
                    not output
                    and model_id.lower().startswith("gpt-5")
                    and not metadata.get("reasoning_placeholder_emitted")
                ):
                    output = [build_responses_reasoning_placeholder()]
                    await event_emitter(
                        {
                            "type": "chat:completion",
                            "data": {
                                "content": serialize_output(output),
                                "output": output,
                            },
                        }
                    )

                async def stream_body_handler(response, form_data):
                    nonlocal usage
                    nonlocal output
                    nonlocal prior_output
                    nonlocal last_response_id
                    nonlocal latest_responses_stream_state

                    response_tool_calls = []

                    delta_count = 0
                    delta_chunk_size = max(
                        CHAT_RESPONSE_STREAM_DELTA_CHUNK_SIZE,
                        int(metadata.get('params', {}).get('stream_delta_chunk_size') or 1),
                    )
                    last_delta_data = None
                    last_delta_type = None
                    last_delta_key = None

                    joined_content = ''
                    joined_part_count = 0

                    async def save_current_response_stream(stream_output: list | None = None):
                        nonlocal joined_content
                        nonlocal joined_part_count

                        if not chat_id or not metadata.get('message_id'):
                            return

                        # content_parts is append-only, so its length tells us when the join is stale
                        if joined_part_count != len(content_parts):
                            joined_content = ''.join(content_parts)
                            joined_part_count = len(content_parts)

                        current_stream_output = stream_output if stream_output is not None else full_output()
                        await save_response_stream(
                            request.app.state.redis,
                            response_stream_task_id,
                            chat_id,
                            metadata.get('message_id'),
                            joined_content or get_output_text(current_stream_output),
                            current_stream_output,
                        )

                    def get_response_delta_key(delta_data: dict):
                        event_type = delta_data.get('type', '')
                        if not event_type.startswith('response.') or not event_type.endswith('.delta'):
                            return None
                        return (
                            event_type,
                            delta_data.get('item_id'),
                            delta_data.get('output_index'),
                            delta_data.get('content_index'),
                            delta_data.get('summary_index'),
                        )

                    def get_response_data_with_full_output_index(response_data: dict):
                        if prior_output and isinstance(response_data.get('output_index'), int):
                            return {
                                **response_data,
                                'output_index': response_data['output_index'] + len(prior_output),
                            }
                        return response_data

                    response_route_idx = response.headers.get('x-openwebui-openai-url-idx') if response.headers else None
                    try:
                        response_route_idx = int(response_route_idx) if response_route_idx is not None else None
                    except ValueError:
                        response_route_idx = None

                    responses_stream_state = ResponsesStreamState(
                        route_idx=response_route_idx,
                        route_url=response.headers.get('x-openwebui-openai-base-url') if response.headers else None,
                    )
                    latest_responses_stream_state = responses_stream_state

                    async def flush_pending_delta_data(threshold: int = 0):
                        nonlocal delta_count
                        nonlocal last_delta_data
                        nonlocal last_delta_type
                        nonlocal last_delta_key

                        if delta_count >= threshold and last_delta_data:
                            await event_emitter(
                                {
                                    'type': 'response:completion',
                                    'data': last_delta_data,
                                }
                            )
                            await save_current_response_stream()
                            delta_count = 0
                            last_delta_data = None
                            last_delta_type = None
                            last_delta_key = None

                    async def queue_pending_delta_data(delta_data: dict, delta_type: str):
                        nonlocal delta_count
                        nonlocal last_delta_data
                        nonlocal last_delta_type
                        nonlocal last_delta_key

                        delta_data = get_response_data_with_full_output_index(delta_data)
                        delta_key = get_response_delta_key(delta_data)
                        if (
                            last_delta_data
                            and last_delta_key == delta_key
                            and isinstance(last_delta_data.get('delta'), str)
                            and isinstance(delta_data.get('delta'), str)
                        ):
                            last_delta_data['delta'] += delta_data['delta']
                            delta_count += 1
                        else:
                            if last_delta_data and (last_delta_type != delta_type or last_delta_key != delta_key):
                                await flush_pending_delta_data()

                            delta_count += 1
                            last_delta_data = delta_data
                            last_delta_type = delta_type
                            last_delta_key = delta_key

                        if delta_count >= delta_chunk_size:
                            await flush_pending_delta_data(delta_chunk_size)

                    async def emit_response_completion_event(response_data: dict, stream_output: list | None = None):
                        if response_data.get('type', '').endswith('.delta'):
                            await queue_pending_delta_data(
                                response_data,
                                response_data.get('type', 'response.delta'),
                            )
                            return

                        response_data = get_response_data_with_full_output_index(response_data)
                        await flush_pending_delta_data()
                        await event_emitter(
                            {
                                'type': 'response:completion',
                                'data': get_response_completion_event_data(response_data),
                            }
                        )
                        await save_current_response_stream(stream_output)

                    filter_extra_params = {'__body__': form_data, **extra_params} if filter_functions else None

                    async def iter_response_lines():
                        try:
                            async for line in response.body_iterator:
                                yield line
                        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                            stream_error = _responses_stream_error_from_exception(
                                e,
                                responses_stream_state,
                            )
                            if stream_error is None and responses_stream_state.completed:
                                log.warning(
                                    'Responses stream closed after response.completed; '
                                    'finalizing buffered output '
                                    '(chat_id=%s, message_id=%s, response_id=%s, sequence=%s)',
                                    metadata.get('chat_id'),
                                    metadata.get('message_id'),
                                    responses_stream_state.response_id,
                                    responses_stream_state.last_sequence_number,
                                )
                                return
                            raise

                    async for line in iter_response_lines():
                        line = line.decode('utf-8', 'replace') if isinstance(line, bytes) else line
                        data = line

                        # Skip empty lines
                        if not data or data.isspace():
                            continue

                        # "data:" is the prefix for each event
                        if not data.startswith('data:'):
                            # Some upstreams return plain JSON error lines in a streaming response
                            # (without SSE `data:` prefix). Try to normalize these into standard
                            # error events so frontend and DB paths still receive them.
                            try:
                                raw_obj = JSONCodec.loads(data)
                                raw_error = raw_obj.get('error') if isinstance(raw_obj, dict) else None
                                if raw_error:
                                    if save_to_chat:
                                        try:
                                            await Chats.upsert_message_to_chat_by_id_and_message_id(
                                                metadata['chat_id'],
                                                metadata['message_id'],
                                                {
                                                    'error': {'content': raw_error},
                                                },
                                            )
                                        except Exception:
                                            pass
                                    await event_emitter({'type': 'chat:completion', 'data': {'error': raw_error}})
                            except Exception:
                                pass
                            continue

                        # Remove the "data:" prefix
                        data = data[5:].strip()

                        try:
                            data = JSONCodec.loads(data)

                            if filter_functions:
                                data, _ = await process_filter_functions(
                                    request=request,
                                    filter_context=filter_context,
                                    filter_functions=filter_functions,
                                    filter_type='stream',
                                    form_data=data,
                                    extra_params=filter_extra_params,
                                )

                            if data:
                                if 'event' in data and not getattr(request.state, 'direct', False):
                                    await event_emitter(data.get('event', {}))

                                if 'selected_model_id' in data:
                                    model_id = data['selected_model_id']
                                    if save_to_chat:
                                        await Chats.upsert_message_to_chat_by_id_and_message_id(
                                            metadata['chat_id'],
                                            metadata['message_id'],
                                            {
                                                'selectedModelId': model_id,
                                            },
                                            touch=False,
                                        )
                                    await event_emitter(
                                        {
                                            'type': 'chat:completion',
                                            'data': data,
                                        }
                                    )
                                # Check for Responses API events (type field starts with "response.")
                                elif data.get('type', '').startswith('response.'):
                                    response_data_type = data.get('type', '')
                                    response_data_is_delta = response_data_type.endswith('.delta')
                                    output, response_metadata = handle_responses_streaming_event(data, output)
                                    responses_stream_state.observe(data, output)

                                    if not response_data_is_delta:
                                        await flush_pending_delta_data()

                                    # Emit citation sources from finalized output items
                                    # (mirrors Chat Completions annotation handling at delta level)
                                    if response_data_type == 'response.output_item.done':
                                        item = data.get('item', {})
                                        if item.get('type') == 'message':
                                            for part in item.get('content', []):
                                                for annotation in part.get('annotations', []):
                                                    if annotation.get('type') == 'url_citation':
                                                        # Handle both flat (Responses API) and nested (Chat Completions) formats
                                                        url_citation = annotation.get('url_citation', annotation)

                                                        url = url_citation.get('url', '')
                                                        title = url_citation.get('title', url)

                                                        if url:
                                                            await event_emitter(
                                                                {
                                                                    'type': 'source',
                                                                    'data': {
                                                                        'source': {
                                                                            'name': title,
                                                                            'url': url,
                                                                        },
                                                                        'document': [title],
                                                                        'metadata': [
                                                                            {
                                                                                'source': url,
                                                                                'name': title,
                                                                            }
                                                                        ],
                                                                    },
                                                                }
                                                            )

                                    await emit_responses_web_search_statuses(output)

                                    processed_data = {
                                        'output': full_output(),
                                        'content': serialize_output(full_output()),
                                        'response_id': responses_stream_state.response_id,
                                        'response_sequence_number': responses_stream_state.last_sequence_number,
                                        'response_route_idx': responses_stream_state.route_idx,
                                    }

                                    if (
                                        metadata.get('chat_id')
                                        and not metadata['chat_id'].startswith('channel:')
                                        and responses_stream_state.last_sequence_number is not None
                                        and data.get('type') in {
                                            'response.in_progress',
                                            'response.output_item.done',
                                            'response.completed',
                                            'response.failed',
                                        }
                                    ):
                                        try:
                                            await Chats.upsert_message_to_chat_by_id_and_message_id(
                                                metadata['chat_id'],
                                                metadata['message_id'],
                                                {
                                                    'response_id': responses_stream_state.response_id,
                                                    'response_sequence_number': responses_stream_state.last_sequence_number,
                                                    'response_route_idx': responses_stream_state.route_idx,
                                                    'response_route_url': responses_stream_state.route_url,
                                                },
                                            )
                                        except Exception:
                                            pass

                                    # print(data)
                                    # print(processed_data)
                                    # Merge any metadata (usage, etc.)
                                    # Strip 'done' — response.completed emits
                                    # it but we may still need to execute tool
                                    # calls. The outer middleware manages the
                                    # actual completion signal.
                                    if response_metadata:
                                        stream_error = response_metadata.get('error')
                                        if stream_error:
                                            if _is_retryable_stream_error(stream_error):
                                                raise RetryableStreamError(stream_error)
                                            raise StreamFatalError(stream_error)

                                        if ENABLE_RESPONSES_API_STATEFUL:
                                            response_id = response_metadata.pop('response_id', None)
                                            if response_id:
                                                last_response_id = response_id

                                        # Normalize and capture usage for DB persistence
                                        if response_metadata.get('usage'):
                                            usage = merge_usage(usage, response_metadata['usage'])
                                            response_metadata['usage'] = usage

                                        if response_metadata.get('error'):
                                            await event_emitter(
                                                {
                                                    'type': 'chat:completion',
                                                    'data': {'error': response_metadata['error']},
                                                }
                                            )

                                    await emit_response_completion_event(data)

                                    if not response_data_is_delta:
                                        await event_emitter({'type': 'chat:completion', 'data': processed_data})

                                    if response_metadata and response_metadata.get('usage'):
                                        await event_emitter(
                                            {
                                                'type': 'chat:completion',
                                                'data': {'usage': usage},
                                            }
                                        )
                                    continue
                                else:
                                    choices = data.get('choices', [])

                                    # Normalize usage data to standard format
                                    raw_usage = data.get('usage', {}) or {}
                                    raw_usage.update(data.get('timings', {}))  # llama.cpp
                                    if raw_usage:
                                        usage = merge_usage(usage, raw_usage)
                                        await event_emitter(
                                            {
                                                'type': 'chat:completion',
                                                'data': {
                                                    'usage': usage,
                                                },
                                            }
                                        )

                                    if not choices:
                                        error = data.get('error', {})
                                        if error:
                                            if _is_retryable_stream_error(error):
                                                raise RetryableStreamError(error)

                                            raise StreamFatalError(error)
                                        continue

                                    delta = choices[0].get('delta', {})
                                    delta_type = 'content'

                                    # Handle delta annotations
                                    annotations = delta.get('annotations')
                                    if annotations:
                                        for annotation in annotations:
                                            if (
                                                annotation.get('type') == 'url_citation'
                                                and 'url_citation' in annotation
                                            ):
                                                url_citation = annotation['url_citation']

                                                url = url_citation.get('url', '')
                                                title = url_citation.get('title', url)

                                                await event_emitter(
                                                    {
                                                        'type': 'source',
                                                        'data': {
                                                            'source': {
                                                                'name': title,
                                                                'url': url,
                                                            },
                                                            'document': [title],
                                                            'metadata': [
                                                                {
                                                                    'source': url,
                                                                    'name': title,
                                                                }
                                                            ],
                                                        },
                                                    }
                                                )

                                    delta_tool_calls = delta.get('tool_calls', None)
                                    if delta_tool_calls:
                                        for delta_tool_call in delta_tool_calls:
                                            tool_call_index = delta_tool_call.get('index')

                                            if tool_call_index is not None:
                                                # Check if the tool call already exists
                                                current_response_tool_call = None
                                                for response_tool_call in response_tool_calls:
                                                    if response_tool_call.get('index') == tool_call_index:
                                                        current_response_tool_call = response_tool_call
                                                        break

                                                if current_response_tool_call is None:
                                                    # Add the new tool call
                                                    delta_tool_call.setdefault('function', {})
                                                    delta_tool_call['function'].setdefault('name', '')
                                                    delta_tool_call['id'] = delta_tool_call.get('id') or output_id('fc')
                                                    delta_arguments = delta_tool_call['function'].get('arguments')
                                                    if not isinstance(delta_arguments, str):
                                                        delta_tool_call['function']['arguments'] = (
                                                            ''
                                                            if delta_arguments is None
                                                            else JSONCodec.dumps(delta_arguments)
                                                        )
                                                    response_tool_calls.append(delta_tool_call)
                                                else:
                                                    # Update the existing tool call
                                                    delta_name = delta_tool_call.get('function', {}).get('name')
                                                    delta_arguments = delta_tool_call.get('function', {}).get(
                                                        'arguments'
                                                    )

                                                    if delta_name:
                                                        current_response_tool_call['function']['name'] = delta_name

                                                    if delta_arguments is not None:
                                                        if not isinstance(delta_arguments, str):
                                                            delta_arguments = JSONCodec.dumps(delta_arguments)
                                                        current_response_tool_call.setdefault('function', {})
                                                        if not isinstance(
                                                            current_response_tool_call['function'].get('arguments'),
                                                            str,
                                                        ):
                                                            current_response_tool_call['function']['arguments'] = ''
                                                        current_response_tool_call['function']['arguments'] += (
                                                            delta_arguments
                                                        )

                                        # Emit pending tool calls in real-time as Responses events.
                                        if response_tool_calls:
                                            output_by_call_id = {
                                                item.get('call_id'): (idx, item)
                                                for idx, item in enumerate(output)
                                                if item.get('type') == 'function_call'
                                            }

                                            for tc in response_tool_calls:
                                                call_id = tc.get('id') or output_id('fc')
                                                tc['id'] = call_id
                                                func = tc.get('function', {})
                                                if call_id in output_by_call_id:
                                                    output_index, item = output_by_call_id[call_id]
                                                    item['name'] = func.get('name', item.get('name', ''))
                                                    item['arguments'] = func.get('arguments', item.get('arguments', ''))
                                                    item['status'] = 'in_progress'
                                                else:
                                                    output_index = len(output)
                                                    item = {
                                                        'type': 'function_call',
                                                        'id': call_id,
                                                        'call_id': call_id,
                                                        'name': func.get('name', ''),
                                                        'arguments': '',
                                                        'status': 'in_progress',
                                                    }
                                                    output.append(item)
                                                    output_by_call_id[call_id] = (output_index, item)
                                                    await emit_response_completion_event(
                                                        {
                                                            'type': 'response.output_item.added',
                                                            'output_index': output_index,
                                                            'item': item.copy(),
                                                        }
                                                    )
                                                    item['arguments'] = func.get('arguments', '')

                                            for delta_tool_call in delta_tool_calls:
                                                tool_call_index = delta_tool_call.get('index')
                                                current_response_tool_call = next(
                                                    (
                                                        tc
                                                        for tc in response_tool_calls
                                                        if tc.get('index') == tool_call_index
                                                    ),
                                                    None,
                                                )
                                                if not current_response_tool_call:
                                                    continue
                                                call_id = current_response_tool_call.get('id')
                                                output_index, _ = output_by_call_id.get(call_id, (len(output) - 1, {}))
                                                delta_arguments = delta_tool_call.get('function', {}).get('arguments')
                                                if delta_arguments is not None:
                                                    if not isinstance(delta_arguments, str):
                                                        delta_arguments = JSONCodec.dumps(delta_arguments)
                                                    await emit_response_completion_event(
                                                        {
                                                            'type': 'response.function_call_arguments.delta',
                                                            'item_id': call_id,
                                                            'output_index': output_index,
                                                            'delta': delta_arguments,
                                                        }
                                                    )

                                            await save_current_response_stream()
                                            data = None
                                            delta_type = 'tool_call'

                                    delta_images = delta.get('images')
                                    image_urls = (
                                        await get_image_urls(delta_images, request, metadata, user)
                                        if delta_images
                                        else []
                                    )
                                    if image_urls:
                                        image_file_list = [{'type': 'image', 'url': url} for url in image_urls]
                                        message_files = image_file_list
                                        if save_to_chat:
                                            message_files = await Chats.add_message_files_by_id_and_message_id(
                                                metadata['chat_id'],
                                                metadata['message_id'],
                                                image_file_list,
                                            )
                                            if message_files is None:
                                                message_files = image_file_list

                                        await event_emitter(
                                            {
                                                'type': 'files',
                                                'data': {'files': message_files},
                                            }
                                        )

                                    # content and reasoning deltas are raw JSON: a stream filter can make them any type
                                    value = delta.get('content')
                                    if value and not isinstance(value, str):
                                        value = f'{value}'

                                    reasoning_content = (
                                        delta.get('reasoning_content')
                                        or delta.get('reasoning')
                                        or delta.get('thinking')
                                    )
                                    if reasoning_content and not isinstance(reasoning_content, str):
                                        reasoning_content = f'{reasoning_content}'
                                    reasoning_details = get_reasoning_details(delta)
                                    reasoning_detail_items = (
                                        [item for item in reasoning_details if isinstance(item, dict)]
                                        if isinstance(reasoning_details, list)
                                        else [reasoning_details]
                                        if isinstance(reasoning_details, dict)
                                        else []
                                    )
                                    existing_reasoning_item = next(
                                        (item for item in reversed(output) if item.get('type') == 'reasoning'),
                                        None,
                                    )
                                    message_index = next(
                                        (i for i, item in enumerate(output) if item.get('type') == 'message'),
                                        None,
                                    )
                                    if reasoning_content or (
                                        reasoning_detail_items
                                        and (
                                            existing_reasoning_item
                                            or any(
                                                item.get('text') or item.get('summary') or item.get('data')
                                                for item in reasoning_detail_items
                                            )
                                        )
                                    ):
                                        reasoning_item = (
                                            existing_reasoning_item
                                            if (reasoning_detail_items and not reasoning_content)
                                            or message_index is not None
                                            else None
                                        )

                                        if reasoning_item is None:
                                            if not output or output[-1].get('type') != 'reasoning':
                                                reasoning_item = {
                                                    'type': 'reasoning',
                                                    'id': output_id('r'),
                                                    'status': 'in_progress',
                                                    'start_tag': '<think>',
                                                    'end_tag': '</think>',
                                                    'attributes': {'type': 'reasoning_content'},
                                                    'content': [],
                                                    'summary': None,
                                                    'started_at': time.time(),
                                                }
                                                if message_index is not None:
                                                    reasoning_item['ended_at'] = time.time()
                                                    reasoning_item['duration'] = 0
                                                    reasoning_item['status'] = 'completed'
                                                    output.insert(message_index, reasoning_item)
                                                else:
                                                    output.append(reasoning_item)
                                            else:
                                                reasoning_item = output[-1]

                                        if reasoning_content:
                                            # Append to reasoning content
                                            parts = reasoning_item.get('content', [])
                                            if parts and parts[-1].get('type') == 'output_text':
                                                parts[-1]['text'] += reasoning_content
                                            else:
                                                reasoning_item['content'] = [
                                                    {
                                                        'type': 'output_text',
                                                        'text': reasoning_content,
                                                    }
                                                ]

                                            reasoning_index = output.index(reasoning_item)
                                            data = {
                                                'type': 'response.reasoning_text.delta',
                                                'item_id': reasoning_item.get('id'),
                                                'output_index': reasoning_index,
                                                'content_index': max(
                                                    len(reasoning_item.get('content', [])) - 1,
                                                    0,
                                                ),
                                                'delta': reasoning_content,
                                            }
                                            delta_type = 'response.reasoning_text.delta'

                                        if reasoning_detail_items:
                                            merge_streamed_reasoning_details(
                                                reasoning_item.setdefault('reasoning_details', []),
                                                reasoning_detail_items,
                                            )
                                            await save_current_response_stream()
                                            # Providers such as OpenRouter send reasoning_details
                                            # alongside the reasoning text: only drop the event when
                                            # the details were all there was to report, otherwise the
                                            # reasoning delta never reaches the client.
                                            if not reasoning_content:
                                                data = None

                                    if value:
                                        if (
                                            output
                                            and output[-1].get('type') == 'reasoning'
                                            and output[-1].get('attributes', {}).get('type') == 'reasoning_content'
                                        ):
                                            reasoning_item = output[-1]
                                            reasoning_item['ended_at'] = time.time()
                                            reasoning_item['duration'] = int(
                                                reasoning_item['ended_at'] - reasoning_item['started_at']
                                            )
                                            reasoning_item['status'] = 'completed'

                                            output.append(
                                                {
                                                    'type': 'message',
                                                    'id': output_id('msg'),
                                                    'status': 'in_progress',
                                                    'role': 'assistant',
                                                    'content': [
                                                        {
                                                            'type': 'output_text',
                                                            'text': '',
                                                        }
                                                    ],
                                                }
                                            )

                                        if ENABLE_CHAT_RESPONSE_BASE64_IMAGE_URL_CONVERSION:
                                            value = await convert_markdown_base64_images(
                                                request,
                                                value,
                                                {
                                                    'chat_id': metadata.get('chat_id', None),
                                                    'message_id': metadata.get('message_id', None),
                                                },
                                                user,
                                            )

                                        # closure-cell str += recopies per chunk; append + join once at read is O(n)
                                        content_parts.append(value)

                                        # Check if we're inside a tag-based block
                                        # (reasoning, code_interpreter, or solution).
                                        # If so, append to the existing in-progress
                                        # item instead of creating a new message —
                                        # otherwise tag_output_handler re-detects the
                                        # start tag on every chunk and fragments the
                                        # output.
                                        last_item = output[-1] if output else None
                                        last_item_type = last_item.get('type', '') if last_item else ''
                                        inside_tag_block = (
                                            last_item is not None
                                            and last_item.get('status') == 'in_progress'
                                            and last_item.get('attributes', {}).get('type') != 'reasoning_content'
                                            and (
                                                last_item_type == 'reasoning'
                                                or last_item_type == 'open_webui:code_interpreter'
                                                or (
                                                    last_item_type == 'message'
                                                    and last_item.get('_tag_type') is not None
                                                )
                                            )
                                        )

                                        if inside_tag_block:
                                            # Append to the existing tag-based item
                                            if last_item_type == 'open_webui:code_interpreter':
                                                last_item['code'] = last_item.get('code', '') + value
                                            elif last_item_type == 'reasoning':
                                                parts = last_item.get('content', [])
                                                if parts and parts[-1].get('type') == 'output_text':
                                                    parts[-1]['text'] += value
                                                else:
                                                    last_item['content'] = [
                                                        {
                                                            'type': 'output_text',
                                                            'text': value,
                                                        }
                                                    ]
                                            else:
                                                # solution or other _tag_type message
                                                msg_parts = last_item.get('content', [])
                                                if msg_parts and msg_parts[-1].get('type') == 'output_text':
                                                    msg_parts[-1]['text'] += value
                                                else:
                                                    last_item['content'] = [
                                                        {
                                                            'type': 'output_text',
                                                            'text': value,
                                                        }
                                                    ]
                                        else:
                                            if not output or output[-1].get('type') != 'message':
                                                output.append(
                                                    {
                                                        'type': 'message',
                                                        'id': output_id('msg'),
                                                        'status': 'in_progress',
                                                        'role': 'assistant',
                                                        'content': [
                                                            {
                                                                'type': 'output_text',
                                                                'text': '',
                                                            }
                                                        ],
                                                    }
                                                )

                                            # Append value to last message item's text
                                            msg_parts = output[-1].get('content', [])
                                            if msg_parts and msg_parts[-1].get('type') == 'output_text':
                                                msg_parts[-1]['text'] += value
                                            else:
                                                output[-1]['content'] = [
                                                    {
                                                        'type': 'output_text',
                                                        'text': value,
                                                    }
                                                ]

                                        if DETECT_REASONING_TAGS:
                                            output, _ = tag_output_handler(
                                                'reasoning',
                                                reasoning_tags,
                                                output,
                                            )

                                            output, _ = tag_output_handler(
                                                'solution',
                                                DEFAULT_SOLUTION_TAGS,
                                                output,
                                            )

                                        if DETECT_CODE_INTERPRETER:
                                            output, end = tag_output_handler(
                                                'code_interpreter',
                                                DEFAULT_CODE_INTERPRETER_TAGS,
                                                output,
                                            )

                                            if end:
                                                break

                                        target_index = len(output) - 1
                                        target_item = output[target_index] if target_index >= 0 else {}
                                        target_content = target_item.get('content', [])
                                        content_index = max(len(target_content) - 1, 0)
                                        delta_event_type = (
                                            'response.reasoning_text.delta'
                                            if target_item.get('type') == 'reasoning'
                                            else 'response.output_text.delta'
                                        )
                                        data = {
                                            'type': delta_event_type,
                                            'item_id': target_item.get('id'),
                                            'output_index': target_index,
                                            'content_index': content_index,
                                            'delta': value,
                                        }
                                        delta_type = delta_event_type

                                if delta and data:
                                    await queue_pending_delta_data(data, delta_type)
                                elif data:
                                    await event_emitter(
                                        {
                                            'type': 'chat:completion',
                                            'data': data,
                                        }
                                    )
                        except RetryableStreamError:
                            raise
                        except StreamFatalError:
                            raise
                        except (asyncio.CancelledError, KeyboardInterrupt):
                            raise
                        except Exception as e:
                            done = 'data: [DONE]' in line
                            if done:
                                pass
                            else:
                                log.debug('Error: %s', e)
                                continue
                    await flush_pending_delta_data()

                    incomplete_error = responses_stream_state.incomplete_error()
                    if incomplete_error:
                        log.warning(
                            'Responses stream ended without response.completed '
                            '(chat_id=%s, message_id=%s, response_id=%s, sequence=%s, route_idx=%s, last_event_type=%s, duration=%s, idle=%s)',
                            metadata.get('chat_id'),
                            metadata.get('message_id'),
                            incomplete_error.get('response_id'),
                            incomplete_error.get('last_sequence_number'),
                            incomplete_error.get('response_route_idx'),
                            incomplete_error.get('last_event_type'),
                            incomplete_error.get('duration'),
                            incomplete_error.get('idle'),
                        )
                        raise RetryableStreamError(incomplete_error)

                    if output:
                        # Clean up the last message item
                        if output[-1].get('type') == 'message':
                            parts = output[-1].get('content', [])
                            if parts and parts[-1].get('type') == 'output_text':
                                parts[-1]['text'] = parts[-1]['text'].strip()

                                if not parts[-1]['text']:
                                    output.pop()

                                    if not output:
                                        output.append(
                                            {
                                                'type': 'message',
                                                'id': output_id('msg'),
                                                'status': 'in_progress',
                                                'role': 'assistant',
                                                'content': [{'type': 'output_text', 'text': ''}],
                                            }
                                        )

                        if output[-1].get('type') == 'reasoning':
                            reasoning_item = output[-1]
                            if reasoning_item.get('ended_at') is None:
                                reasoning_item['ended_at'] = time.time()
                                if reasoning_item.get('started_at') is not None:
                                    reasoning_item['duration'] = int(
                                        reasoning_item['ended_at'] - reasoning_item['started_at']
                                    )
                                reasoning_item['status'] = 'completed'

                    if response_tool_calls:
                        for tc in response_tool_calls:
                            call_id = tc.get('id', '')
                            arguments = tc.get('function', {}).get('arguments', '{}')
                            for output_index, item in enumerate(output):
                                if item.get('type') == 'function_call' and item.get('call_id') == call_id:
                                    item['arguments'] = arguments
                                    item['status'] = 'completed'
                                    await emit_response_completion_event(
                                        {
                                            'type': 'response.function_call_arguments.done',
                                            'item_id': item.get('id'),
                                            'output_index': output_index,
                                            'arguments': arguments,
                                        }
                                    )
                                    await emit_response_completion_event(
                                        {
                                            'type': 'response.output_item.done',
                                            'output_index': output_index,
                                            'item': item.copy(),
                                        }
                                    )
                                    break
                        tool_calls.append(_split_tool_calls(response_tool_calls))

                    # Responses API path: extract function_call items from output
                    if not response_tool_calls and output:
                        # Collect call_ids that already have results,
                        # including those from prior_output so we don't
                        # re-process tool calls from a previous turn.
                        handled_call_ids = {
                            item.get('call_id')
                            for item in (prior_output + output)
                            if item.get('type') == 'function_call_output'
                        }
                        responses_api_tool_calls = []
                        for item in output:
                            call_id = item.get('call_id') or item.get('id') or output_id('fc')
                            if item.get('type') == 'function_call' and call_id not in handled_call_ids:
                                arguments = item.get('arguments', '{}')
                                responses_api_tool_calls.append(
                                    {
                                        'id': call_id,
                                        'index': len(responses_api_tool_calls),
                                        'function': {
                                            'name': item.get('name', ''),
                                            'arguments': (
                                                arguments if isinstance(arguments, str) else JSONCodec.dumps(arguments)
                                            ),
                                        },
                                    }
                                )
                        if responses_api_tool_calls:
                            tool_calls.append(_split_tool_calls(responses_api_tool_calls))

                async def stream_with_retries(
                    current_response,
                    current_form_data,
                    *,
                    bypass_system_prompt=False,
                    contextual_base_messages=None,
                ) -> bool:
                    nonlocal output
                    nonlocal prior_output

                    base_form_data = copy.deepcopy(current_form_data)
                    if contextual_base_messages is not None:
                        base_form_data['messages'] = copy.deepcopy(contextual_base_messages)
                    retry_attempts = max(0, CHAT_RESPONSE_STREAM_RETRY_ATTEMPTS)
                    background_attempts_by_response_id = {}
                    contextual_attempt = 0
                    saved_retry_cursor = {}
                    diagnostic_retry_cursor = {}

                    while True:
                        retry_error = None

                        try:
                            await stream_body_handler(current_response, current_form_data)
                            if prior_output:
                                output[:0] = prior_output
                                prior_output = []
                            return True
                        except RetryableStreamError as e:
                            retry_error = e.error
                        except StreamFatalError as e:
                            await record_stream_error(e.error, saved_retry_cursor, diagnostic_retry_cursor)
                            return False
                        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                            retry_error = _responses_stream_error_from_exception(
                                e,
                                latest_responses_stream_state,
                            )
                            if retry_error is None:
                                retry_error = {
                                    'code': 'stream_completed_finalize_interrupted',
                                    'message': (
                                        'Responses API stream closed after response.completed '
                                        'before handler finalization.'
                                    ),
                                    'type': 'upstream_error',
                                }
                                retry_error.update(
                                    _responses_stream_cursor_from_error(
                                        retry_error,
                                        latest_responses_stream_state,
                                    )
                                )
                                await record_stream_error(
                                    retry_error,
                                    saved_retry_cursor,
                                    diagnostic_retry_cursor,
                                )
                                return False
                        finally:
                            await cleanup_stream_response(current_response)

                        retry_error = enrich_stream_error_with_cursor(
                            retry_error,
                            saved_cursor=saved_retry_cursor,
                        )

                        background_attempt = None
                        if ENABLE_RESPONSES_API_BACKGROUND_RESUME:
                            background_attempt = _next_response_background_resume_attempt(
                                background_attempts_by_response_id,
                                retry_error,
                                max(0, RESPONSES_API_BACKGROUND_RESUME_ATTEMPTS),
                            )

                        if background_attempt is not None:
                            retry_response = await try_resume_background_response(
                                retry_error,
                                background_attempt,
                            )

                            if isinstance(retry_response, StreamingResponse):
                                current_response = retry_response
                                continue

                            if retry_response is not None:
                                log.warning(
                                    'Background response resume did not return a stream; falling back '
                                    '(chat_id=%s, message_id=%s)',
                                    metadata.get('chat_id'),
                                    metadata.get('message_id'),
                                )

                        if contextual_attempt >= retry_attempts or not _stream_retryable(full_output()):
                            await record_stream_error(
                                retry_error,
                                saved_retry_cursor,
                                diagnostic_retry_cursor,
                            )
                            return False

                        contextual_attempt += 1
                        retry_form_data = await prepare_contextual_stream_retry(
                            retry_error,
                            contextual_attempt,
                            retry_attempts,
                            base_form_data,
                        )
                        # From here on, a no-cursor failure belongs to the new
                        # contextual response. Keep the original cursor for
                        # diagnostics/final persistence, but never use it as
                        # a background-resume target for the replacement response.
                        if saved_retry_cursor:
                            diagnostic_retry_cursor = dict(saved_retry_cursor)
                            saved_retry_cursor.clear()

                        if CHAT_RESPONSE_STREAM_RETRY_DELAY > 0:
                            await asyncio.sleep(CHAT_RESPONSE_STREAM_RETRY_DELAY)

                        retry_response = await generate_chat_completion(
                            request,
                            retry_form_data,
                            user,
                            bypass_system_prompt=bypass_system_prompt,
                        )

                        if not isinstance(retry_response, StreamingResponse):
                            await record_stream_error(
                                {
                                    'code': 'stream_retry_non_streaming_response',
                                    'message': 'Upstream retry returned a non-streaming response.',
                                    'type': 'upstream_error',
                                },
                                saved_retry_cursor,
                                diagnostic_retry_cursor,
                            )
                            return False

                        current_response = retry_response
                        current_form_data = retry_form_data

                stream_completed = await stream_with_retries(response, form_data)
                if not stream_completed:
                    return

                tool_call_iterations = 0
                max_tool_call_iterations = getattr(
                    request.state,
                    'max_tool_call_iterations',
                    CHAT_RESPONSE_MAX_TOOL_CALL_ITERATIONS,
                )
                tool_call_sources = []  # Track citation sources from tool results
                all_tool_call_sources = []  # Accumulated sources across all iterations
                user_message = get_last_user_message(form_data['messages'])

                # Check if citations are enabled for this model
                citations_enabled = (model.get('info', {}).get('meta', {}).get('capabilities') or {}).get(
                    'citations', True
                )

                # Use the pre-RAG system content captured before the
                # initial file-source injection in process_chat_payload.
                # This ensures restore truly undoes the RAG template.
                original_system_content = metadata.get('system_prompt')
                if original_system_content is None:
                    original_system_message = get_system_message(form_data['messages'])
                    original_system_content = (
                        get_content_from_message(original_system_message) if original_system_message else None
                    )

                while tool_calls and (
                    max_tool_call_iterations is None or tool_call_iterations < max_tool_call_iterations
                ):
                    tool_call_iterations += 1

                    response_tool_calls = tool_calls.pop(0)
                    ask_user_staged, ask_user_error = stage_ask_user_tool_calls(response_tool_calls, output, output_id)
                    if ask_user_error:
                        response_tool_calls = [
                            tool_call
                            for tool_call in response_tool_calls
                            if tool_call.get('function', {}).get('name') != 'ask_user'
                        ]
                    elif ask_user_staged:
                        if is_saved_chat_id(metadata.get('chat_id')) and metadata.get('message_id'):
                            await pause_for_tool_approval(
                                metadata['chat_id'],
                                metadata['message_id'],
                                full_output(),
                                form_data,
                                metadata,
                            )
                        await event_emitter({'type': 'chat:completion', 'data': {'output': full_output()}})
                        return

                    # Append function_call items for each tool call
                    # (Responses API already has them from streaming, so skip duplicates)
                    existing_call_ids = {item.get('call_id') for item in output if item.get('type') == 'function_call'}
                    for tc in response_tool_calls:
                        call_id = tc.get('id', '')
                        if call_id not in existing_call_ids:
                            func = tc.get('function', {})
                            output.append(
                                {
                                    'type': 'function_call',
                                    'id': call_id or output_id('fc'),
                                    'call_id': call_id,
                                    'name': func.get('name', ''),
                                    'arguments': func.get('arguments', '{}'),
                                    'status': 'in_progress',
                                }
                            )

                    tool_approval_mode = metadata.get('params', {}).get('tool_approval_mode', 'full')
                    if (
                        response_tool_calls
                        and tool_approval_mode == 'ask'
                        and is_saved_chat_id(metadata.get('chat_id'))
                        and metadata.get('message_id')
                    ):
                        await pause_for_tool_approval(
                            metadata['chat_id'],
                            metadata['message_id'],
                            full_output(),
                            form_data,
                            metadata,
                        )
                        await event_emitter(
                            {
                                'type': 'chat:completion',
                                'data': {
                                    'output': full_output(),
                                },
                            }
                        )
                        return

                    await event_emitter(
                        {
                            'type': 'chat:completion',
                            'data': {
                                'output': full_output(),
                            },
                        }
                    )

                    tools = metadata.get('tools', {})

                    results = []

                    def parse_tool_params(tool_call):
                        tool_args = tool_call.get('function', {}).get('arguments', '{}')
                        params = {}
                        if tool_args and tool_args.strip():
                            try:
                                params = JSONCodec.loads(tool_args)
                            except Exception:
                                try:
                                    params = ast.literal_eval(tool_args)
                                except Exception as e:
                                    log.debug(e)
                                    return None
                        tool_call.setdefault('function', {})['arguments'] = JSONCodec.dumps(params)
                        return params

                    async def execute_tool_call(tool_call):
                        name = tool_call.get('function', {}).get('name', '')
                        params = parse_tool_params(tool_call)
                        if params is None:
                            return {}, None, None, None, False
                        tool = tools.get(name)
                        if not tool:
                            return params, f'Error: Tool "{name}" not found.', None, None, False
                        spec = tool.get('spec', {})
                        tool_type = tool.get('type', '')
                        direct_tool = tool.get('direct', False)
                        allowed_params = spec.get('parameters', {}).get('properties', {}).keys()
                        params = {key: value for key, value in params.items() if key in allowed_params}
                        try:
                            if direct_tool:
                                if not event_caller:
                                    raise RuntimeError(
                                        'Direct tool execution requires an active websocket session.'
                                    )
                                result = await event_caller(
                                    {
                                        'type': 'execute:tool',
                                        'data': {
                                            'id': str(uuid4()),
                                            'name': name,
                                            'params': params,
                                            'server': tool.get('server', {}),
                                            'session_id': metadata.get('session_id'),
                                        },
                                    }
                                )
                            else:
                                function = await get_updated_tool_function(
                                    function=tool['callable'],
                                    extra_params={
                                        '__messages__': form_data.get('messages', []),
                                        '__files__': metadata.get('files', []),
                                    },
                                )
                                result = await function(**params)
                        except Exception as e:
                            result = {'error': str(e)}
                        return params, result, tool, tool_type, direct_tool

                    delegate_calls = [
                        tool_call
                        for tool_call in response_tool_calls
                        if tool_call.get('function', {}).get('name') == 'delegate_task'
                    ]
                    tool_results = {}
                    for tool_call in response_tool_calls:
                        if tool_call.get('function', {}).get('name') != 'delegate_task':
                            tool_results[id(tool_call)] = await execute_tool_call(tool_call)
                    tool_results.update(
                        zip(
                            [id(tool_call) for tool_call in delegate_calls],
                            await asyncio.gather(*(execute_tool_call(tool_call) for tool_call in delegate_calls)),
                        )
                    )

                    for tool_call in response_tool_calls:
                        tool_call_id = tool_call.get('id', '')
                        tool_function_name = tool_call.get('function', {}).get('name', '')
                        tool_function_params, tool_result, tool, tool_type, direct_tool = tool_results[id(tool_call)]
                        if tool_result is None:
                            results.append(
                                {
                                    'tool_call_id': tool_call_id,
                                    'content': (
                                        'Error: Tool call arguments could not be parsed. The model generated '
                                        f'malformed or incomplete JSON for `{tool_function_name}`. Please try again.'
                                    ),
                                }
                            )
                            continue

                        terminal_file_result = build_terminal_file_tool_result(
                            tool_function_name,
                            tool_function_params,
                            tool_result,
                            tool,
                            metadata,
                        )
                        if terminal_file_result:
                            tool_result = terminal_file_result

                        tool_result, tool_result_files, tool_result_embeds = await process_tool_result(
                            request,
                            tool_function_name,
                            tool_result,
                            tool_type,
                            direct_tool,
                            metadata,
                            user,
                        )

                        await terminal_event_handler(
                            tool_function_name,
                            tool_function_params,
                            tool_result,
                            event_emitter,
                        )

                        # Extract citation sources from tool results
                        if (
                            citations_enabled
                            and tool_function_name
                            in [
                                'search_web',
                                'fetch_url',
                                'view_file',
                                'view_knowledge_file',
                                'query_knowledge_files',
                                'query_chat_files',
                            ]
                            and tool_result
                        ):
                            try:
                                citation_sources = get_citation_source_from_tool_result(
                                    tool_name=tool_function_name,
                                    tool_params=tool_function_params,
                                    tool_result=tool_result,
                                    tool_id=tool.get('tool_id', '') if tool else '',
                                )
                                tool_call_sources.extend(citation_sources)
                            except Exception as e:
                                log.exception(f'Error extracting citation source: {e}')

                        results.append(
                            {
                                'tool_call_id': tool_call_id,
                                'content': tool_result_content(tool_result),
                                **({'files': tool_result_files} if tool_result_files else {}),
                                **({'embeds': tool_result_embeds} if tool_result_embeds else {}),
                            }
                        )

                    result_status_by_call_id = {}
                    for result in results:
                        output_parts = [{'type': 'input_text', 'text': result.get('content', '')}]
                        local_output_status = (
                            'failed' if _is_tool_result_error(result.get('content', '')) else 'completed'
                        )
                        result_status_by_call_id[result.get('tool_call_id', '')] = local_output_status

                        # Separate image data URIs (for LLM via input_image) from
                        # other files (for frontend display via files attribute).
                        display_files = []
                        for file_item in result.get('files', []):
                            if file_item.get('type') == 'image' and file_item.get('url', '').startswith('data:'):
                                # LLM-only: add as input_image part, not frontend display output.
                                output_parts.append({'type': 'input_image', 'image_url': file_item['url']})
                            else:
                                # Frontend display (MCP images, audio, etc.)
                                display_files.append(file_item)

                        output.append(
                            {
                                'type': 'function_call_output',
                                'id': output_id('fco'),
                                'call_id': result.get('tool_call_id', ''),
                                'output': output_parts,
                                'status': local_output_status,
                                **({'files': display_files} if display_files else {}),
                                **({'embeds': result.get('embeds')} if result.get('embeds') else {}),
                            }
                        )

                    # Update function_call statuses and parsed/sanitized arguments.
                    for tc in response_tool_calls:
                        call_id = tc.get('id', '')
                        for item in output:
                            if item.get('type') == 'function_call' and item.get('call_id') == call_id:
                                item['status'] = result_status_by_call_id.get(call_id, 'completed')
                                item['arguments'] = tc.get('function', {}).get('arguments', '{}')
                                break

                    # Emit citation sources to the frontend for display
                    if citations_enabled:
                        for source in tool_call_sources:
                            await event_emitter({'type': 'source', 'data': source})

                        # Apply tool source context to messages for the model.
                        # Restoring to pre-RAG original prevents duplicating
                        # the RAG template across file and tool sources.
                        all_tool_call_sources.extend(tool_call_sources)
                        if all_tool_call_sources and user_message:
                            # Restore pre-RAG message state before re-applying
                            # to prevent RAG template duplication.
                            original_user_message = metadata.get('user_prompt') or user_message
                            set_last_user_message_content(
                                original_user_message,
                                form_data['messages'],
                            )
                            if original_system_content is not None:
                                if get_system_message(form_data['messages']):
                                    replace_system_message_content(
                                        original_system_content,
                                        form_data['messages'],
                                    )
                                else:
                                    form_data['messages'] = add_or_update_system_message(
                                        original_system_content,
                                        form_data['messages'],
                                    )
                            else:
                                replace_system_message_content('', form_data['messages'])

                            # Build context: file sources with content,
                            # tool sources as citation markers only.
                            source_ids = {}
                            source_context = get_source_context(
                                metadata.get('sources', []), source_ids
                            ) + get_source_context(
                                all_tool_call_sources,
                                source_ids,
                                include_content=False,
                            )
                            source_context = source_context.strip()
                            if source_context:
                                rag_content = await rag_template(
                                    await Config.get('rag.template'),
                                    source_context,
                                    user_message,
                                )
                                if RAG_SYSTEM_CONTEXT:
                                    form_data['messages'] = add_or_update_system_message(
                                        rag_content,
                                        form_data['messages'],
                                        append=True,
                                    )
                                else:
                                    form_data['messages'] = add_or_update_user_message(
                                        rag_content,
                                        form_data['messages'],
                                        append=False,
                                    )
                        tool_call_sources.clear()

                    # Strip input_image parts (large base64 data URIs) from the
                    # output sent to the frontend — they're only for LLM consumption
                    # via convert_output_to_messages.
                    frontend_output = []
                    for item in full_output():
                        if item.get('type') == 'function_call_output':
                            parts = item.get('output', [])
                            if any(p.get('type') == 'input_image' for p in parts):
                                item = {**item, 'output': [p for p in parts if p.get('type') != 'input_image']}
                        frontend_output.append(item)

                    await event_emitter(
                        {
                            'type': 'chat:completion',
                            'data': {
                                'output': frontend_output,
                            },
                        }
                    )

                    try:
                        new_form_data = {
                            **form_data,
                            'model': model_id,
                            'stream': True,
                            'metadata': metadata,
                        }

                        if ENABLE_RESPONSES_API_STATEFUL and last_response_id:
                            system_message = get_system_message(form_data['messages'])
                            new_form_data['messages'] = (
                                [system_message] if system_message else []
                            ) + convert_output_to_messages(
                                output, raw=True, reasoning_format=get_reasoning_format(model)
                            )
                            new_form_data['previous_response_id'] = last_response_id
                        else:
                            tool_messages = convert_output_to_messages(
                                output,
                                raw=True,
                                reasoning_format=get_reasoning_format(model),
                                flatten_tool_images=True,
                            )

                            # Chat Completions providers don't support multimodal
                            # tool messages.  Extract images into a user message.
                            image_urls = []
                            for message in tool_messages:
                                if message.get('role') == 'tool' and isinstance(message.get('content'), list):
                                    text_parts = []
                                    for part in message['content']:
                                        if part.get('type') == 'input_text':
                                            text_parts.append(part.get('text', ''))
                                        elif part.get('type') == 'input_image':
                                            image_urls.append(part.get('image_url', ''))
                                    message['content'] = ''.join(text_parts)

                            new_form_data['messages'] = [
                                *form_data['messages'],
                                *tool_messages,
                            ]

                            if image_urls:
                                new_form_data['messages'].append(
                                    {
                                        'role': 'user',
                                        'content': [
                                            {
                                                'type': 'text',
                                                'text': 'Here are the images from the tool results above. Please analyze them.',
                                            },
                                            *[{'type': 'image_url', 'image_url': {'url': url}} for url in image_urls],
                                        ],
                                    }
                                )

                        if filter_functions:
                            new_form_data, _ = await process_filter_functions(
                                request=request,
                                filter_context=filter_context,
                                filter_functions=filter_functions,
                                filter_type='request',
                                form_data=new_form_data,
                                extra_params=extra_params,
                            )

                        new_form_data = normalize_messages_for_model(new_form_data)

                        res = await generate_chat_completion(
                            request,
                            new_form_data,
                            user,
                            bypass_system_prompt=True,
                        )

                        if isinstance(res, StreamingResponse):
                            # Save accumulated output and start fresh.
                            # Responses API output_index values are relative
                            # to the current response — a clean output list
                            # keeps indices aligned. The display prefix
                            # ensures the UI shows tool history during
                            # streaming.
                            prior_output = list(full_output())
                            # Trim the trailing empty placeholder message
                            # so it doesn't persist as a ghost item once
                            # the new stream produces real content.
                            if (
                                prior_output
                                and prior_output[-1].get('type') == 'message'
                                and prior_output[-1].get('status') == 'in_progress'
                            ):
                                msg_parts = prior_output[-1].get('content', [])
                                if not msg_parts or (len(msg_parts) == 1 and not msg_parts[0].get('text', '').strip()):
                                    prior_output.pop()
                            output = []
                            stream_completed = await stream_with_retries(
                                res,
                                new_form_data,
                                bypass_system_prompt=True,
                                contextual_base_messages=form_data['messages'],
                            )
                            if not stream_completed:
                                return
                            output[:0] = prior_output
                            prior_output = []
                        elif getattr(res, 'status_code', 200) >= 400:
                            await emit_message_error(get_message_error_content(get_response_error_detail(res)))
                            break
                        else:
                            break
                    except Exception as e:
                        error_content = get_message_error_content(e)
                        log.exception('Tool-call continuation failed: %s', error_content)
                        await emit_message_error(error_content)
                        break

                if (
                    max_tool_call_iterations is not None
                    and tool_calls
                    and tool_call_iterations >= max_tool_call_iterations
                ):
                    log.warning('Tool-call iteration limit reached (%s)', max_tool_call_iterations)
                    error_content = f'Tool-call limit reached ({max_tool_call_iterations} iterations).'
                    await emit_message_error(error_content)

                if DETECT_CODE_INTERPRETER:
                    MAX_RETRIES = 5
                    retries = 0

                    while output and output[-1].get('type') == 'open_webui:code_interpreter' and retries < MAX_RETRIES:
                        await event_emitter(
                            {
                                'type': 'chat:completion',
                                'data': {
                                    'output': full_output(),
                                },
                            }
                        )

                        retries += 1
                        log.debug('Attempt count: %s', retries)

                        ci_item = output[-1]
                        ci_output = ''
                        try:
                            if ci_item.get('attributes', {}).get('type') == 'code':
                                code = ci_item.get('code', '')
                                # Sanitize code (strips ANSI codes and markdown fences)
                                code = sanitize_code(code)

                                if CODE_INTERPRETER_BLOCKED_MODULES:
                                    blocking_code = textwrap.dedent(f"""
                                        import builtins
    
                                        BLOCKED_MODULES = {CODE_INTERPRETER_BLOCKED_MODULES}
    
                                        _real_import = builtins.__import__
                                        def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
                                            if name.split('.')[0] in BLOCKED_MODULES:
                                                importer_name = globals.get('__name__') if globals else None
                                                if importer_name == '__main__':
                                                    raise ImportError(
                                                        f"Direct import of module {{name}} is restricted."
                                                    )
                                            return _real_import(name, globals, locals, fromlist, level)
    
                                        builtins.__import__ = restricted_import
                                    """)
                                    code = blocking_code + '\n' + code

                                ci_engine = await Config.get('code_interpreter.engine')
                                if ci_engine == 'pyodide':
                                    ci_output = await event_caller(
                                        {
                                            'type': 'execute:python',
                                            'data': {
                                                'id': str(uuid4()),
                                                'code': code,
                                                'session_id': metadata.get('session_id', None),
                                                'files': metadata.get('files', []),
                                            },
                                        }
                                    )
                                elif ci_engine == 'jupyter':
                                    ci_output = await execute_code_jupyter(
                                        await Config.get('code_interpreter.jupyter.url'),
                                        code,
                                        (
                                            await Config.get('code_interpreter.jupyter.auth_token')
                                            if await Config.get('code_interpreter.jupyter.auth') == 'token'
                                            else None
                                        ),
                                        (
                                            await Config.get('code_interpreter.jupyter.auth_password')
                                            if await Config.get('code_interpreter.jupyter.auth') == 'password'
                                            else None
                                        ),
                                        await Config.get('code_interpreter.jupyter.timeout'),
                                    )
                                else:
                                    ci_output = {'stdout': 'Code interpreter engine not configured.'}

                                log.debug('Code interpreter output: %s', ci_output)

                                # Handle error responses from event_caller
                                # (e.g. session disconnected, timeout)
                                if isinstance(ci_output, dict) and ci_output.get('error'):
                                    ci_output = {'stderr': ci_output['error']}

                                if isinstance(ci_output, dict):
                                    stdout = ci_output.get('stdout', '')

                                    if isinstance(stdout, str):
                                        stdoutLines = stdout.split('\n')
                                        for idx, line in enumerate(stdoutLines):
                                            if re.match(r'data:image/\w+;base64', line):
                                                image_url = await get_image_url_from_base64(
                                                    request,
                                                    line,
                                                    metadata,
                                                    user,
                                                )
                                                if image_url:
                                                    stdoutLines[idx] = f'![Output Image]({image_url})'

                                        ci_output['stdout'] = '\n'.join(stdoutLines)

                                    result = ci_output.get('result', '')

                                    if isinstance(result, str):
                                        resultLines = result.split('\n')
                                        for idx, line in enumerate(resultLines):
                                            if re.match(r'data:image/\w+;base64', line):
                                                image_url = await get_image_url_from_base64(
                                                    request,
                                                    line,
                                                    metadata,
                                                    user,
                                                )
                                                resultLines[idx] = f'![Output Image]({image_url})'
                                        ci_output['result'] = '\n'.join(resultLines)
                        except Exception as e:
                            ci_output = str(e)

                        ci_item['output'] = ci_output
                        ci_item['status'] = 'completed'

                        output.append(
                            {
                                'type': 'message',
                                'id': output_id('msg'),
                                'status': 'in_progress',
                                'role': 'assistant',
                                'content': [{'type': 'output_text', 'text': ''}],
                            }
                        )

                        await event_emitter(
                            {
                                'type': 'chat:completion',
                                'data': {
                                    'output': full_output(),
                                },
                            }
                        )

                        try:
                            new_form_data = {
                                **form_data,
                                'model': model_id,
                                'stream': True,
                                'metadata': metadata,
                                'messages': [
                                    *form_data['messages'],
                                    *convert_output_to_messages(
                                        output,
                                        raw=True,
                                        reasoning_format=get_reasoning_format(model),
                                        flatten_tool_images=True,
                                    ),
                                ],
                            }

                            if filter_functions:
                                new_form_data, _ = await process_filter_functions(
                                    request=request,
                                    filter_context=filter_context,
                                    filter_functions=filter_functions,
                                    filter_type='request',
                                    form_data=new_form_data,
                                    extra_params=extra_params,
                                )

                            new_form_data = normalize_messages_for_model(new_form_data)

                            res = await generate_chat_completion(
                                request,
                                new_form_data,
                                user,
                                bypass_system_prompt=True,
                            )

                            if isinstance(res, StreamingResponse):
                                stream_completed = await stream_with_retries(
                                    res,
                                    new_form_data,
                                    bypass_system_prompt=True,
                                    contextual_base_messages=form_data['messages'],
                                )
                                if not stream_completed:
                                    return
                            elif getattr(res, 'status_code', 200) >= 400:
                                await emit_message_error(get_message_error_content(get_response_error_detail(res)))
                                break
                            else:
                                break
                        except Exception as e:
                            error_content = get_message_error_content(e)
                            log.exception('Code interpreter continuation failed: %s', error_content)
                            await emit_message_error(error_content)
                            break

                # Mark all in-progress items as completed
                for item in output:
                    if item.get('status') == 'in_progress':
                        item['status'] = 'completed'

                current_output = full_output()
                title = await Chats.get_chat_title_by_id(metadata['chat_id']) if save_to_chat else ''
                data = {
                    'done': True,
                    'output': current_output,
                    'title': title,
                    **({'usage': usage} if usage else {}),
                }

                if save_to_chat:
                    # Save final output once. The delta path keeps in-progress
                    # state in response_streams instead of writing tokens to DB.
                    await Chats.upsert_message_to_chat_by_id_and_message_id(
                        metadata['chat_id'],
                        metadata['message_id'],
                        {
                            'done': True,
                            'output': current_output,
                            **({'usage': usage} if usage else {}),
                        },
                    )

                await clear_response_stream(request.app.state.redis, response_stream_task_id)
                await publish_chat_finished_event(
                    request, user, metadata, title, ''.join(content_parts), current_output
                )

                await event_emitter(
                    {
                        'type': 'chat:completion',
                        'data': data,
                    }
                )

                ctx['assistant_message'] = {
                    'content': ''.join(content_parts) or get_output_text(current_output),
                    'output': current_output,
                    **({'usage': usage} if usage else {}),
                }
                await outlet_filter_handler(ctx)
                await background_tasks_handler(ctx)
            except asyncio.CancelledError:
                log.warning('Task was cancelled!')

                # Close the response body iterator to trigger cleanup
                # in stream_wrapper's finally block and release the
                # upstream connection.  Without this, the async
                # generator is orphaned and may spin in anyio internals.
                if hasattr(response, 'body_iterator') and hasattr(response.body_iterator, 'aclose'):
                    try:
                        await asyncio.shield(response.body_iterator.aclose())
                    except (asyncio.CancelledError, Exception):
                        pass

                async def save_cancelled_state():
                    await event_emitter({'type': 'chat:tasks:cancel'})
                    if save_to_chat:
                        await Chats.upsert_message_to_chat_by_id_and_message_id(
                            metadata['chat_id'],
                            metadata['message_id'],
                            {
                                'done': True,
                                'output': full_output(),
                            },
                        )
                    await clear_response_stream(request.app.state.redis, response_stream_task_id)

                try:
                    await asyncio.shield(save_cancelled_state())
                except (asyncio.CancelledError, Exception):
                    pass
                raise  # re-raise CancelledError for proper propagation

            if response.background is not None:
                await response.background()

        return await response_handler(response, events)

    else:
        # Fallback to the original response
        async def stream_wrapper(original_generator, events):
            def wrap_item(item):
                return f'data: {item}\n\n'

            assistant_message = {}
            filter_context = FilterContext()
            has_api_outlet_filters = ENABLE_API_OUTLET_FILTERS and bool(filter_functions)
            if ENABLE_API_OUTLET_FILTERS and not has_api_outlet_filters:
                try:
                    model_id = model.get('id') if isinstance(model, dict) else model
                    has_api_outlet_filters = bool(
                        (isinstance(model, dict) and 'pipeline' in model)
                        or get_sorted_filters(model_id, request.app.state.MODELS)
                    )
                except Exception:
                    has_api_outlet_filters = True

            for event in events:
                event, _ = await process_filter_functions(
                    request=request,
                    filter_context=filter_context,
                    filter_functions=filter_functions,
                    filter_type='stream',
                    form_data=event,
                    extra_params=extra_params,
                )

                if event:
                    yield wrap_item(JSONCodec.dumps(event))

            async for data in original_generator:
                if filter_functions:
                    line = data.decode('utf-8', 'replace') if isinstance(data, bytes) else data
                    if isinstance(line, str) and line.startswith('data:'):
                        payload = line.removeprefix('data:').strip()
                        if payload and payload != '[DONE]':
                            try:
                                event = JSONCodec.loads(payload)
                            except JSONCodec.JSONDecodeError:
                                event = None

                            if isinstance(event, dict):
                                event, _ = await process_filter_functions(
                                    request=request,
                                    filter_context=filter_context,
                                    filter_functions=filter_functions,
                                    filter_type='stream',
                                    form_data=event,
                                    extra_params=extra_params,
                                )
                                data = wrap_item(JSONCodec.dumps(event)) if event else None

                if data:
                    if has_api_outlet_filters:
                        update_assistant_message_from_stream(assistant_message, data)
                    yield data

            if has_api_outlet_filters and assistant_message:
                ctx['assistant_message'] = assistant_message
                await outlet_filter_handler(ctx)

        return StreamingResponse(
            stream_wrapper(response.body_iterator, events),
            headers=dict(response.headers),
            background=response.background,
        )


async def process_chat_response(response, ctx):
    # Non-streaming response
    if not isinstance(response, StreamingResponse):
        return await non_streaming_chat_response_handler(response, ctx)

    # Non standard response
    if not any(
        content_type in response.headers['Content-Type']
        for content_type in ['text/event-stream', 'application/x-ndjson']
    ):
        return response

    # Streaming response
    return await streaming_chat_response_handler(response, ctx)
