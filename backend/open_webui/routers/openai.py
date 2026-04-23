import asyncio
import copy
import hashlib
import json
import logging
import re
import time
from typing import Optional
from urllib.parse import quote, urlparse

import aiohttp
from aiocache import cached
import requests


from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from fastapi import Depends, HTTPException, Request, APIRouter, status
from fastapi.responses import (
    FileResponse,
    StreamingResponse,
    JSONResponse,
    PlainTextResponse,
)
from pydantic import BaseModel, ConfigDict

from sqlalchemy.ext.asyncio import AsyncSession

from open_webui.internal.db import get_async_session

from open_webui.models.models import Models
from open_webui.models.access_grants import AccessGrants
from open_webui.models.files import Files
from open_webui.models.groups import Groups
from open_webui.utils.access_control import has_connection_access, check_model_access
from open_webui.config import (
    CACHE_DIR,
)
from open_webui.env import (
    MODELS_CACHE_TTL,
    AIOHTTP_CLIENT_SESSION_SSL,
    AIOHTTP_CLIENT_TIMEOUT,
    AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST,
    ENABLE_FORWARD_USER_INFO_HEADERS,
    FORWARD_SESSION_INFO_HEADER_CHAT_ID,
    BYPASS_MODEL_ACCESS_CONTROL,
    ENABLE_OPENAI_API_PASSTHROUGH,
)
from open_webui.models.users import UserModel

from open_webui.constants import ERROR_MESSAGES


from open_webui.utils.payload import (
    apply_model_params_to_body_openai,
    apply_system_prompt_to_body,
)
from open_webui.utils.misc import (
    convert_logit_bias_input_to_json,
    stream_chunks_handler,
)
from open_webui.utils.session_pool import (
    cleanup_response,
    get_session,
    stream_wrapper,
)

from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.headers import include_user_info_headers
from open_webui.utils.anthropic import is_anthropic_url, get_anthropic_models
from open_webui.storage.provider import Storage

log = logging.getLogger(__name__)


##########################################
#
# Utility functions
# Let the responses returned through this gate be worth
# the question that summoned them.
#
##########################################


async def send_get_request(
    request: Request = None,
    url=None,
    key=None,
    user: UserModel = None,
    config=None,
):
    timeout = aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST)
    try:
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            if request and config:
                headers, cookies = await get_headers_and_cookies(request, url, key, config, user=user)
            else:
                headers = {
                    **({'Authorization': f'Bearer {key}'} if key else {}),
                }
                cookies = None

                if ENABLE_FORWARD_USER_INFO_HEADERS and user:
                    headers = include_user_info_headers(headers, user)

            async with session.get(
                url,
                headers=headers,
                cookies=cookies,
                ssl=AIOHTTP_CLIENT_SESSION_SSL,
            ) as response:
                return await response.json()
    except Exception as e:
        # Handle connection error here
        log.error(f'Connection error: {e}')
        return None


async def get_models_request(
    request: Request = None,
    url=None,
    key=None,
    user: UserModel = None,
    config=None,
):
    if is_anthropic_url(url):
        return await get_anthropic_models(url, key, user=user)
    return await send_get_request(request, f'{url}/models', key, user=user, config=config)


def openai_reasoning_model_handler(payload):
    """
    Handle reasoning model specific parameters
    """
    if 'max_tokens' in payload:
        # Convert "max_tokens" to "max_completion_tokens" for all reasoning models
        payload['max_completion_tokens'] = payload['max_tokens']
        del payload['max_tokens']

    # Handle system role conversion based on model type
    if payload['messages'][0]['role'] == 'system':
        model_lower = payload['model'].lower()
        # Legacy models use "user" role instead of "system"
        if model_lower.startswith('o1-mini') or model_lower.startswith('o1-preview'):
            payload['messages'][0]['role'] = 'user'
        else:
            payload['messages'][0]['role'] = 'developer'

    return payload


def merge_response_tools(
    existing_tools: Optional[list], configured_tools: Optional[list]
) -> Optional[list]:
    if not configured_tools:
        return existing_tools
    if not existing_tools:
        return configured_tools

    merged_tools = []
    seen = set()

    for tool in [*existing_tools, *configured_tools]:
        try:
            marker = json.dumps(tool, sort_keys=True, ensure_ascii=False)
        except TypeError:
            marker = str(tool)

        if marker in seen:
            continue

        seen.add(marker)
        merged_tools.append(tool)

    return merged_tools


def apply_model_params_to_body_responses(params: dict, form_data: dict) -> dict:
    if not params:
        return form_data

    payload = apply_model_params_to_body_openai(params.copy(), {})
    configured_tools = payload.pop("tools", None)
    configured_tool_choice = payload.pop("tool_choice", None)

    if "max_tokens" in payload:
        form_data["max_output_tokens"] = payload.pop("max_tokens")

    if "reasoning_effort" in payload:
        reasoning = form_data.get("reasoning") or {}
        if not isinstance(reasoning, dict):
            reasoning = {}
        reasoning["effort"] = payload.pop("reasoning_effort")
        form_data["reasoning"] = reasoning

    form_data.update(payload)

    if configured_tools is not None:
        form_data["tools"] = merge_response_tools(
            form_data.get("tools"), configured_tools
        )

    if configured_tool_choice is not None and "tool_choice" not in form_data:
        form_data["tool_choice"] = configured_tool_choice

    return form_data


def apply_default_responses_reasoning_summary(
    model: str | None, form_data: dict
) -> dict:
    if not model or not model.lower().startswith("gpt-5"):
        return form_data

    reasoning = form_data.get("reasoning")
    if reasoning is None:
        reasoning = {}
    elif not isinstance(reasoning, dict):
        reasoning = {}
    else:
        reasoning = reasoning.copy()

    # Cherry Studio's GPT-5 requests effectively resolve to a detailed
    # reasoning summary when summary streaming is available. Mirror that
    # behavior here so Open WebUI has the best chance of receiving
    # reasoning summary deltas instead of only a completed placeholder.
    if reasoning.get("summary") in (None, "auto"):
        reasoning["summary"] = "detailed"

    form_data["reasoning"] = reasoning
    return form_data


def summarize_response_debug_value(value):
    if isinstance(value, str):
        return {"type": "str", "len": len(value)}
    if isinstance(value, list):
        return {"type": "list", "len": len(value)}
    if isinstance(value, dict):
        return {
            key: summarize_response_debug_value(item)
            for key, item in value.items()
        }
    return value


def stream_sse_lines(stream: aiohttp.StreamReader):
    async def yield_sse_lines():
        buffer = b""

        async for data, _ in stream.iter_chunks():
            if not data:
                continue

            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                yield line + b"\n"

        if buffer:
            yield buffer

    return yield_sse_lines()


async def get_headers_and_cookies(
    request: Request,
    url,
    key=None,
    config=None,
    metadata: Optional[dict] = None,
    user: UserModel = None,
):
    cookies = {}
    headers = {
        'Content-Type': 'application/json',
        **(
            {
                'HTTP-Referer': 'https://openwebui.com/',
                'X-Title': 'Open WebUI',
            }
            if 'openrouter.ai' in url
            else {}
        ),
    }

    if ENABLE_FORWARD_USER_INFO_HEADERS and user:
        headers = include_user_info_headers(headers, user)
        if metadata and metadata.get('chat_id'):
            headers[FORWARD_SESSION_INFO_HEADER_CHAT_ID] = metadata.get('chat_id')

    token = None
    auth_type = config.get('auth_type')

    if auth_type == 'bearer' or auth_type is None:
        # Default to bearer if not specified
        token = f'{key}'
    elif auth_type == 'none':
        token = None
    elif auth_type == 'session':
        cookies = request.cookies
        token = request.state.token.credentials
    elif auth_type == 'system_oauth':
        cookies = request.cookies

        oauth_token = None
        try:
            if request.cookies.get('oauth_session_id', None):
                oauth_token = await request.app.state.oauth_manager.get_oauth_token(
                    user.id,
                    request.cookies.get('oauth_session_id', None),
                )
        except Exception as e:
            log.error(f'Error getting OAuth token: {e}')

        if oauth_token:
            token = f'{oauth_token.get("access_token", "")}'

    elif auth_type in ('azure_ad', 'microsoft_entra_id'):
        token = get_microsoft_entra_id_access_token()

    if token:
        headers['Authorization'] = f'Bearer {token}'

    if config.get('headers') and isinstance(config.get('headers'), dict):
        headers = {**headers, **config.get('headers')}

    return headers, cookies


def get_microsoft_entra_id_access_token():
    """
    Get Microsoft Entra ID access token using DefaultAzureCredential for Azure OpenAI.
    Returns the token string or None if authentication fails.
    """
    try:
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), 'https://cognitiveservices.azure.com/.default'
        )
        return token_provider()
    except Exception as e:
        log.error(f'Error getting Microsoft Entra ID access token: {e}')
        return None


def get_openai_api_config_for_index(request: Request, idx: int):
    url = request.app.state.config.OPENAI_API_BASE_URLS[idx]
    key = request.app.state.config.OPENAI_API_KEYS[idx]
    api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
        str(idx),
        request.app.state.config.OPENAI_API_CONFIGS.get(url, {}),
    )
    return url, key, api_config


def build_openai_file_upload_headers(
    request: Request,
    url: str,
    key: Optional[str],
    config: Optional[dict] = None,
    metadata: Optional[dict] = None,
    user: Optional[UserModel] = None,
):
    config = config or {}
    headers = {
        **(
            {
                "HTTP-Referer": "https://openwebui.com/",
                "X-Title": "Open WebUI",
            }
            if "openrouter.ai" in url
            else {}
        ),
    }

    if ENABLE_FORWARD_USER_INFO_HEADERS and user:
        headers = include_user_info_headers(headers, user)
        if metadata and metadata.get("chat_id"):
            headers[FORWARD_SESSION_INFO_HEADER_CHAT_ID] = metadata.get("chat_id")

    token = None
    auth_type = config.get("auth_type")

    if auth_type == "bearer" or auth_type is None:
        token = key
    elif auth_type == "none":
        token = None
    elif auth_type in ("azure_ad", "microsoft_entra_id"):
        token = get_microsoft_entra_id_access_token()

    if token:
        headers["Authorization"] = f"Bearer {token}"

    if config.get("headers") and isinstance(config.get("headers"), dict):
        headers = {**headers, **config.get("headers")}

    headers.pop("Content-Type", None)
    return headers


def _extract_attachment_file_id(file_item: Optional[dict]) -> Optional[str]:
    if not isinstance(file_item, dict):
        return None

    nested = file_item.get("file")
    if isinstance(nested, dict) and nested.get("id"):
        return nested.get("id")

    file_id = file_item.get("id")
    if isinstance(file_id, str) and file_id:
        return file_id

    return None


def _extract_attachment_content_type(file_item: Optional[dict]) -> str:
    if not isinstance(file_item, dict):
        return ""

    content_type = file_item.get("content_type")
    if isinstance(content_type, str):
        return content_type

    nested = file_item.get("file")
    if isinstance(nested, dict):
        nested_meta = nested.get("meta") or {}
        nested_content_type = nested_meta.get("content_type")
        if isinstance(nested_content_type, str):
            return nested_content_type

    return ""


def _is_image_attachment(file_item: Optional[dict]) -> bool:
    if not isinstance(file_item, dict):
        return False

    attachment_type = file_item.get("type")
    if attachment_type == "image":
        return True

    return _extract_attachment_content_type(file_item).startswith("image/")


async def invalidate_cached_openai_file_ids(attached_files: list[dict]) -> None:
    for attached_file in attached_files:
        local_file_id = _extract_attachment_file_id(attached_file)
        if not local_file_id:
            continue

        await Files.update_file_data_by_id(
            local_file_id,
            {
                "openai_file_id": None,
                "openai_backend_index": None,
                "openai_api_base_url": None,
                "openai_upload_status": "stale",
                "openai_file_id_valid_for_next_request": False,
                "error": None,
            },
        )


async def mark_openai_file_id_consumed(local_file_id: str) -> None:
    await Files.update_file_data_by_id(
        local_file_id,
        {
            "openai_file_id_valid_for_next_request": False,
            "openai_upload_status": "consumed",
            "error": None,
        },
    )


def is_unknown_file_id_error(response_payload) -> bool:
    error_text = ""

    if isinstance(response_payload, dict):
        error = response_payload.get("error", response_payload)
        if isinstance(error, dict):
            error_text = (
                error.get("message")
                or error.get("detail")
                or json.dumps(error, ensure_ascii=False, default=str)
            )
        else:
            error_text = str(error)
    else:
        error_text = str(response_payload)

    return "unknown file_id" in error_text.lower()


def upload_local_file_to_openai(
    request: Request,
    file_item,
    *,
    idx: int = 0,
    metadata: Optional[dict] = None,
    user: Optional[UserModel] = None,
) -> str:
    url, key, api_config = get_openai_api_config_for_index(request, idx)
    data = file_item.data or {}

    if (
        data.get("openai_file_id")
        and data.get("openai_file_id_valid_for_next_request") is True
        and data.get("openai_backend_index") == idx
        and data.get("openai_api_base_url") == url
    ):
        return data["openai_file_id"]

    if api_config.get("azure", False):
        raise ValueError("Azure OpenAI file uploads are not supported by this override")

    if not file_item.path:
        raise ValueError("Local file path is missing")

    file_path = Storage.get_file(file_item.path)
    headers = build_openai_file_upload_headers(
        request,
        url,
        key,
        api_config,
        metadata=metadata,
        user=user,
    )

    content_type = ((file_item.meta or {}).get("content_type") or "").strip()
    if not content_type:
        content_type = "application/octet-stream"

    with open(file_path, "rb") as fh:
        response = requests.post(
            f"{url}/files",
            data={"purpose": "user_data"},
            files={"file": (file_item.filename, fh, content_type)},
            headers=headers,
            timeout=AIOHTTP_CLIENT_TIMEOUT,
        )

    try:
        response_payload = response.json()
    except ValueError:
        response_payload = response.text

    if response.status_code >= 400:
        if isinstance(response_payload, dict):
            error = response_payload.get("error", response_payload)
        else:
            error = response_payload
        raise ValueError(f"Upstream file upload failed: {error}")

    if not isinstance(response_payload, dict) or not response_payload.get("id"):
        raise ValueError("Upstream file upload returned no file id")

    return response_payload["id"]


async def ensure_openai_file_id(
    request: Request,
    file_id: str,
    *,
    idx: int = 0,
    metadata: Optional[dict] = None,
    user: Optional[UserModel] = None,
) -> Optional[str]:
    file_item = await Files.get_file_by_id(file_id)
    if not file_item:
        return None

    openai_file_id = await asyncio.to_thread(
        upload_local_file_to_openai,
        request,
        file_item,
        idx=idx,
        metadata=metadata,
        user=user,
    )

    if openai_file_id:
        await Files.update_file_data_by_id(
            file_item.id,
            {
                "openai_file_id": openai_file_id,
                "openai_backend_index": idx,
                "openai_api_base_url": request.app.state.config.OPENAI_API_BASE_URLS[idx],
                "openai_upload_status": "uploaded",
                "openai_file_id_valid_for_next_request": True,
                "openai_uploaded_at": int(time.time()),
                "status": "completed",
                "error": None,
            },
        )

    return openai_file_id


async def inject_openai_files_into_messages(
    request: Request,
    payload: dict,
    metadata: Optional[dict],
    user: UserModel,
    *,
    idx: int,
) -> dict:
    attached_files = (metadata or {}).get("files") or payload.pop("files", None) or []
    if not attached_files:
        return payload

    messages = payload.get("messages") or []
    if not messages:
        return payload

    file_parts = []

    for attached_file in attached_files:
        if _is_image_attachment(attached_file):
            continue

        local_file_id = _extract_attachment_file_id(attached_file)
        if not local_file_id:
            continue

        try:
            openai_file_id = await ensure_openai_file_id(
                request,
                local_file_id,
                idx=idx,
                metadata=metadata,
                user=user,
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=ERROR_MESSAGES.DEFAULT(str(e)),
            )

        if openai_file_id:
            file_parts.append({"type": "input_file", "file_id": openai_file_id})
            # Treat upstream file ids as single-use for request routing.
            # This avoids a stale cached file id adding a failed /responses
            # attempt before we re-upload on the next turn.
            await mark_openai_file_id_consumed(local_file_id)

    if not file_parts:
        return payload

    last_user_message = None
    for message in reversed(messages):
        if message.get("role") == "user":
            last_user_message = message
            break

    if last_user_message is None:
        messages.append({"role": "user", "content": file_parts})
        payload["messages"] = messages
        return payload

    content = last_user_message.get("content", "")
    if isinstance(content, str):
        content_parts = [{"type": "text", "text": content}] if content else []
    elif isinstance(content, list):
        content_parts = list(content)
    else:
        content_parts = [{"type": "text", "text": str(content)}]

    existing_file_ids = {
        part.get("file_id")
        or ((part.get("file") or {}).get("file_id") if isinstance(part, dict) else None)
        for part in content_parts
        if isinstance(part, dict)
        and part.get("type") in {"file", "input_file"}
    }
    existing_file_ids.discard(None)

    new_file_parts = [
        part for part in file_parts if part.get("file_id") not in existing_file_ids
    ]
    last_user_message["content"] = [*new_file_parts, *content_parts]
    payload["messages"] = messages
    return payload


def disable_local_file_context_capability(model: dict) -> dict:
    info = model.get("info") or {}
    meta = info.get("meta") or {}
    capabilities = meta.get("capabilities") or {}
    capabilities["file_context"] = False
    meta["capabilities"] = capabilities
    info["meta"] = meta
    model["info"] = info
    return model


##########################################
#
# API routes
#
##########################################

router = APIRouter()


@router.get('/config')
async def get_config(request: Request, user=Depends(get_admin_user)):
    return {
        'ENABLE_OPENAI_API': request.app.state.config.ENABLE_OPENAI_API,
        'OPENAI_API_BASE_URLS': request.app.state.config.OPENAI_API_BASE_URLS,
        'OPENAI_API_KEYS': request.app.state.config.OPENAI_API_KEYS,
        'OPENAI_API_CONFIGS': request.app.state.config.OPENAI_API_CONFIGS,
    }


class OpenAIConfigForm(BaseModel):
    ENABLE_OPENAI_API: Optional[bool] = None
    OPENAI_API_BASE_URLS: list[str]
    OPENAI_API_KEYS: list[str]
    OPENAI_API_CONFIGS: dict


@router.post('/config/update')
async def update_config(request: Request, form_data: OpenAIConfigForm, user=Depends(get_admin_user)):
    request.app.state.config.ENABLE_OPENAI_API = form_data.ENABLE_OPENAI_API
    request.app.state.config.OPENAI_API_BASE_URLS = form_data.OPENAI_API_BASE_URLS
    request.app.state.config.OPENAI_API_KEYS = form_data.OPENAI_API_KEYS

    # Check if API KEYS length is same than API URLS length
    if len(request.app.state.config.OPENAI_API_KEYS) != len(request.app.state.config.OPENAI_API_BASE_URLS):
        if len(request.app.state.config.OPENAI_API_KEYS) > len(request.app.state.config.OPENAI_API_BASE_URLS):
            request.app.state.config.OPENAI_API_KEYS = request.app.state.config.OPENAI_API_KEYS[
                : len(request.app.state.config.OPENAI_API_BASE_URLS)
            ]
        else:
            request.app.state.config.OPENAI_API_KEYS += [''] * (
                len(request.app.state.config.OPENAI_API_BASE_URLS) - len(request.app.state.config.OPENAI_API_KEYS)
            )

    request.app.state.config.OPENAI_API_CONFIGS = form_data.OPENAI_API_CONFIGS

    # Remove the API configs that are not in the API URLS
    keys = list(map(str, range(len(request.app.state.config.OPENAI_API_BASE_URLS))))
    request.app.state.config.OPENAI_API_CONFIGS = {
        key: value for key, value in request.app.state.config.OPENAI_API_CONFIGS.items() if key in keys
    }

    return {
        'ENABLE_OPENAI_API': request.app.state.config.ENABLE_OPENAI_API,
        'OPENAI_API_BASE_URLS': request.app.state.config.OPENAI_API_BASE_URLS,
        'OPENAI_API_KEYS': request.app.state.config.OPENAI_API_KEYS,
        'OPENAI_API_CONFIGS': request.app.state.config.OPENAI_API_CONFIGS,
    }


@router.post('/audio/speech')
async def speech(request: Request, user=Depends(get_verified_user)):
    idx = None
    try:
        idx = request.app.state.config.OPENAI_API_BASE_URLS.index('https://api.openai.com/v1')

        body = await request.body()
        name = hashlib.sha256(body).hexdigest()

        SPEECH_CACHE_DIR = CACHE_DIR / 'audio' / 'speech'
        SPEECH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        file_path = SPEECH_CACHE_DIR.joinpath(f'{name}.mp3')
        file_body_path = SPEECH_CACHE_DIR.joinpath(f'{name}.json')

        # Check if the file already exists in the cache
        if file_path.is_file():
            return FileResponse(file_path)

        url = request.app.state.config.OPENAI_API_BASE_URLS[idx]
        key = request.app.state.config.OPENAI_API_KEYS[idx]
        api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
            str(idx),
            request.app.state.config.OPENAI_API_CONFIGS.get(url, {}),  # Legacy support
        )

        headers, cookies = await get_headers_and_cookies(request, url, key, api_config, user=user)

        r = None
        try:
            session = await get_session()
            r = await session.post(
                url=f'{url}/audio/speech',
                data=body,
                headers=headers,
                cookies=cookies,
                ssl=AIOHTTP_CLIENT_SESSION_SSL,
            )

            r.raise_for_status()

            # Save the streaming content to a file
            with open(file_path, 'wb') as f:
                async for chunk in r.content.iter_chunked(8192):
                    f.write(chunk)

            with open(file_body_path, 'w') as f:
                json.dump(json.loads(body.decode('utf-8')), f)

            # Return the saved file
            return FileResponse(file_path)

        except Exception as e:
            log.exception(e)

            detail = None
            if r is not None:
                try:
                    res = await r.json()
                    if 'error' in res:
                        detail = f'External: {res["error"]}'
                except Exception:
                    detail = f'External: {e}'

            raise HTTPException(
                status_code=r.status if r else 500,
                detail=detail if detail else 'Open WebUI: Server Connection Error',
            )

    except ValueError:
        raise HTTPException(status_code=401, detail=ERROR_MESSAGES.OPENAI_NOT_FOUND)


async def get_all_models_responses(request: Request, user: UserModel) -> list:
    if not request.app.state.config.ENABLE_OPENAI_API:
        return []

    # Cache config values locally to avoid repeated Redis lookups.
    # Each access to request.app.state.config.<KEY> triggers a Redis GET;
    # caching here avoids hundreds of redundant round-trips.
    api_base_urls = request.app.state.config.OPENAI_API_BASE_URLS
    api_keys = list(request.app.state.config.OPENAI_API_KEYS)
    api_configs = request.app.state.config.OPENAI_API_CONFIGS

    # Check if API KEYS length is same than API URLS length
    num_urls = len(api_base_urls)
    num_keys = len(api_keys)

    if num_keys != num_urls:
        # if there are more keys than urls, remove the extra keys
        if num_keys > num_urls:
            api_keys = api_keys[:num_urls]
            request.app.state.config.OPENAI_API_KEYS = api_keys
        # if there are more urls than keys, add empty keys
        else:
            api_keys += [''] * (num_urls - num_keys)
            request.app.state.config.OPENAI_API_KEYS = api_keys

    request_tasks = []
    for idx, url in enumerate(api_base_urls):
        if (str(idx) not in api_configs) and (url not in api_configs):  # Legacy support
            request_tasks.append(get_models_request(request, url, api_keys[idx], user=user))
        else:
            api_config = api_configs.get(
                str(idx),
                api_configs.get(url, {}),  # Legacy support
            )

            enable = api_config.get('enable', True)
            model_ids = api_config.get('model_ids', [])

            if enable:
                if len(model_ids) == 0:
                    request_tasks.append(get_models_request(request, url, api_keys[idx], user=user, config=api_config))
                else:
                    model_list = {
                        'object': 'list',
                        'data': [
                            {
                                'id': model_id,
                                'name': model_id,
                                'owned_by': 'openai',
                                'openai': {'id': model_id},
                                'urlIdx': idx,
                            }
                            for model_id in model_ids
                        ],
                    }

                    request_tasks.append(asyncio.ensure_future(asyncio.sleep(0, model_list)))
            else:
                request_tasks.append(asyncio.ensure_future(asyncio.sleep(0, None)))

    responses = await asyncio.gather(*request_tasks)

    for idx, response in enumerate(responses):
        if response:
            url = api_base_urls[idx]
            api_config = api_configs.get(
                str(idx),
                api_configs.get(url, {}),  # Legacy support
            )

            connection_type = api_config.get('connection_type', 'external')
            prefix_id = api_config.get('prefix_id', None)
            tags = api_config.get('tags', [])

            model_list = response if isinstance(response, list) else response.get('data', [])
            if not isinstance(model_list, list):
                # Catch non-list responses
                model_list = []

            for model in model_list:
                # Remove name key if its value is None #16689
                if 'name' in model and model['name'] is None:
                    del model['name']

                if prefix_id:
                    model['id'] = f'{prefix_id}.{model.get("id", model.get("name", ""))}'

                if tags:
                    model['tags'] = tags

                if connection_type:
                    model['connection_type'] = connection_type

    log.debug(f'get_all_models:responses() {responses}')
    return responses


async def get_filtered_models(models, user, db=None):
    # Filter models based on user access control
    model_ids = [model['id'] for model in models.get('data', [])]
    model_infos = {model_info.id: model_info for model_info in await Models.get_models_by_ids(model_ids, db=db)}
    user_group_ids = {group.id for group in await Groups.get_groups_by_member_id(user.id, db=db)}

    # Batch-fetch accessible resource IDs in a single query instead of N has_access calls
    accessible_model_ids = await AccessGrants.get_accessible_resource_ids(
        user_id=user.id,
        resource_type='model',
        resource_ids=list(model_infos.keys()),
        permission='read',
        user_group_ids=user_group_ids,
        db=db,
    )

    filtered_models = []
    for model in models.get('data', []):
        model_info = model_infos.get(model['id'])
        if model_info:
            if user.id == model_info.user_id or model_info.id in accessible_model_ids:
                filtered_models.append(model)
    return filtered_models


@cached(
    ttl=MODELS_CACHE_TTL,
    key=lambda _, user: f'openai_all_models_{user.id}' if user else 'openai_all_models',
)
async def get_all_models(request: Request, user: UserModel) -> dict[str, list]:
    log.info('get_all_models()')

    if not request.app.state.config.ENABLE_OPENAI_API:
        return {'data': []}

    # Cache config value locally to avoid repeated Redis lookups inside
    # the nested loop in get_merged_models (one GET per model otherwise).
    api_base_urls = request.app.state.config.OPENAI_API_BASE_URLS

    responses = await get_all_models_responses(request, user=user)

    def extract_data(response):
        if response and 'data' in response:
            return response['data']
        if isinstance(response, list):
            return response
        return None

    def is_supported_openai_models(model_id):
        if any(
            name in model_id
            for name in [
                'babbage',
                'dall-e',
                'davinci',
                'embedding',
                'tts',
                'whisper',
            ]
        ):
            return False
        return True

    def get_merged_models(model_lists):
        log.debug(f'merge_models_lists {model_lists}')
        models = {}

        for idx, model_list in enumerate(model_lists):
            if model_list is not None and 'error' not in model_list:
                for model in model_list:
                    model_id = model.get('id') or model.get('name')

                    base_url = api_base_urls[idx]
                    hostname = urlparse(base_url).hostname if base_url else None
                    if hostname == 'api.openai.com' and not is_supported_openai_models(model_id):
                        # Skip unwanted OpenAI models
                        continue

                    if model_id and model_id not in models:
                        models[model_id] = disable_local_file_context_capability(
                            {
                                **model,
                                "name": model.get("name", model_id),
                                "owned_by": "openai",
                                "openai": model,
                                "connection_type": model.get(
                                    "connection_type", "external"
                                ),
                                "urlIdx": idx,
                            }
                        )

        return models

    models = get_merged_models(map(extract_data, responses))
    log.debug(f'models: {models}')

    request.app.state.OPENAI_MODELS = models
    return {'data': list(models.values())}


@router.get('/models')
@router.get('/models/{url_idx}')
async def get_models(request: Request, url_idx: Optional[int] = None, user=Depends(get_verified_user)):
    if not request.app.state.config.ENABLE_OPENAI_API:
        raise HTTPException(status_code=503, detail='OpenAI API is disabled')

    models = {
        'data': [],
    }

    if url_idx is None:
        models = await get_all_models(request, user=user)
    else:
        url = request.app.state.config.OPENAI_API_BASE_URLS[url_idx]
        key = request.app.state.config.OPENAI_API_KEYS[url_idx]

        api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
            str(url_idx),
            request.app.state.config.OPENAI_API_CONFIGS.get(url, {}),  # Legacy support
        )

        r = None
        async with aiohttp.ClientSession(
            trust_env=True,
            timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST),
        ) as session:
            try:
                headers, cookies = await get_headers_and_cookies(request, url, key, api_config, user=user)

                if api_config.get('azure', False):
                    models = {
                        'data': api_config.get('model_ids', []) or [],
                        'object': 'list',
                    }
                elif is_anthropic_url(url):
                    models = await get_anthropic_models(url, key, user=user)
                    if models is None:
                        raise Exception('Failed to connect to Anthropic API')
                else:
                    async with session.get(
                        f'{url}/models',
                        headers=headers,
                        cookies=cookies,
                        ssl=AIOHTTP_CLIENT_SESSION_SSL,
                    ) as r:
                        if r.status != 200:
                            error_detail = f'HTTP Error: {r.status}'
                            try:
                                res = await r.json()
                                if 'error' in res:
                                    error_detail = f'External Error: {res["error"]}'
                            except Exception:
                                pass
                            raise Exception(error_detail)

                        response_data = await r.json()

                        if 'api.openai.com' in url:
                            response_data['data'] = [
                                model
                                for model in response_data.get('data', [])
                                if not any(
                                    name in model['id']
                                    for name in [
                                        'babbage',
                                        'dall-e',
                                        'davinci',
                                        'embedding',
                                        'tts',
                                        'whisper',
                                    ]
                                )
                            ]

                        models = response_data
            except aiohttp.ClientError as e:
                # ClientError covers all aiohttp requests issues
                log.exception(f'Client error: {str(e)}')
                raise HTTPException(status_code=500, detail='Open WebUI: Server Connection Error')
            except Exception as e:
                log.exception(f'Unexpected error: {e}')
                error_detail = f'Unexpected error: {str(e)}'
                raise HTTPException(status_code=500, detail=error_detail)

    if user.role == 'user' and not BYPASS_MODEL_ACCESS_CONTROL:
        models['data'] = await get_filtered_models(models, user)

    return models


class ConnectionVerificationForm(BaseModel):
    url: str
    key: str

    config: Optional[dict] = None


@router.post('/verify')
async def verify_connection(
    request: Request,
    form_data: ConnectionVerificationForm,
    user=Depends(get_admin_user),
):
    url = form_data.url
    key = form_data.key

    api_config = form_data.config or {}

    async with aiohttp.ClientSession(
        trust_env=True,
        timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST),
    ) as session:
        try:
            headers, cookies = await get_headers_and_cookies(request, url, key, api_config, user=user)

            if api_config.get('azure', False):
                # Only set api-key header if not using Azure Entra ID authentication
                auth_type = api_config.get('auth_type', 'bearer')
                if auth_type not in ('azure_ad', 'microsoft_entra_id'):
                    headers['api-key'] = key

                api_version = api_config.get('api_version', '') or '2023-03-15-preview'
                async with session.get(
                    url=f'{url}/openai/models?api-version={api_version}',
                    headers=headers,
                    cookies=cookies,
                    ssl=AIOHTTP_CLIENT_SESSION_SSL,
                ) as r:
                    try:
                        response_data = await r.json()
                    except Exception:
                        response_data = await r.text()

                    if r.status != 200:
                        if isinstance(response_data, (dict, list)):
                            return JSONResponse(status_code=r.status, content=response_data)
                        else:
                            return PlainTextResponse(status_code=r.status, content=response_data)

                    return response_data
            elif is_anthropic_url(url):
                result = await get_anthropic_models(url, key)
                if result is None:
                    raise HTTPException(status_code=500, detail=ERROR_MESSAGES.SERVER_CONNECTION_ERROR)
                if 'error' in result:
                    raise HTTPException(status_code=500, detail=result['error'])
                return result
            else:
                async with session.get(
                    f'{url}/models',
                    headers=headers,
                    cookies=cookies,
                    ssl=AIOHTTP_CLIENT_SESSION_SSL,
                ) as r:
                    try:
                        response_data = await r.json()
                    except Exception:
                        response_data = await r.text()

                    if r.status != 200:
                        if isinstance(response_data, (dict, list)):
                            return JSONResponse(status_code=r.status, content=response_data)
                        else:
                            return PlainTextResponse(status_code=r.status, content=response_data)

                    return response_data

        except aiohttp.ClientError as e:
            # ClientError covers all aiohttp requests issues
            log.exception(f'Client error: {str(e)}')
            raise HTTPException(status_code=500, detail=ERROR_MESSAGES.SERVER_CONNECTION_ERROR)
        except Exception as e:
            log.exception(f'Unexpected error: {e}')
            raise HTTPException(status_code=500, detail=ERROR_MESSAGES.SERVER_CONNECTION_ERROR)


def get_azure_allowed_params(api_version: str) -> set[str]:
    allowed_params = {
        'messages',
        'temperature',
        'role',
        'content',
        'contentPart',
        'contentPartImage',
        'enhancements',
        'dataSources',
        'n',
        'stream',
        'stop',
        'max_tokens',
        'presence_penalty',
        'frequency_penalty',
        'logit_bias',
        'user',
        'function_call',
        'functions',
        'tools',
        'tool_choice',
        'top_p',
        'log_probs',
        'top_logprobs',
        'response_format',
        'seed',
        'max_completion_tokens',
        'reasoning_effort',
    }

    try:
        if api_version >= '2024-09-01-preview':
            allowed_params.add('stream_options')
    except ValueError:
        log.debug(f'Invalid API version {api_version} for Azure OpenAI. Defaulting to allowed parameters.')

    return allowed_params


def is_openai_new_model(model: str) -> bool:
    model_lower = model.lower()
    # o-series models (o1, o3, o4, o5, ...)
    if re.match(r'^o\d+', model_lower):
        return True
    # gpt-N where N >= 5 (gpt-5, gpt-5.2, gpt-6, ...)
    m = re.match(r'^gpt-(\d+)', model_lower)
    if m and int(m.group(1)) >= 5:
        return True
    return False


def _sanitize_model_for_url(model: str) -> str:
    """Sanitize a model name before interpolating it into a URL path.

    Rejects path traversal attempts (../, /, \\) and percent-encodes
    the name so it is safe to use as a single URL path segment
    (e.g. Azure deployment name).
    """
    if not model or '..' in model or '/' in model or '\\' in model:
        raise HTTPException(
            status_code=400,
            detail='Invalid model name: must not be empty or contain path separators or traversal sequences',
        )
    return quote(model, safe='')


def convert_to_azure_payload(url, payload: dict, api_version: str):
    model = payload.get('model', '')

    # Filter allowed parameters based on Azure OpenAI API
    allowed_params = get_azure_allowed_params(api_version)

    # Special handling for o-series models
    if is_openai_new_model(model):
        # Convert max_tokens to max_completion_tokens for o-series models
        if 'max_tokens' in payload:
            payload['max_completion_tokens'] = payload['max_tokens']
            del payload['max_tokens']

        # Remove temperature if not 1 for o-series models
        if 'temperature' in payload and payload['temperature'] != 1:
            log.debug(
                f'Removing temperature parameter for o-series model {model} as only default value (1) is supported'
            )
            del payload['temperature']

    # Filter out unsupported parameters
    payload = {k: v for k, v in payload.items() if k in allowed_params}

    # Sanitize model name to prevent path traversal in the deployment URL
    model = _sanitize_model_for_url(model)

    url = f'{url}/openai/deployments/{model}'
    return url, payload


# Fields accepted by the Responses API for each input item type.
RESPONSES_ALLOWED_FIELDS: dict[str, set[str]] = {
    'message': {'type', 'role', 'content'},
    'function_call': {'type', 'call_id', 'name', 'arguments', 'id'},
    'function_call_output': {'type', 'call_id', 'output'},
}


def _normalize_stored_item(item: dict) -> dict:
    """Strip local-only fields from a stored output item before replaying it.

    Open WebUI stores extra bookkeeping fields (``id``, ``status``,
    ``started_at``, ``ended_at``, ``duration``, ``_tag_type``,
    ``attributes``, ``summary``, etc.) that the Responses API does
    not accept.  This helper returns a copy containing only the
    fields the API understands.
    """
    item_type = item.get('type', '')
    allowed = RESPONSES_ALLOWED_FIELDS.get(item_type)
    if allowed is None:
        # Unknown type — pass through as-is (e.g. reasoning, extension items).
        return item
    return {k: v for k, v in item.items() if k in allowed}


def convert_to_responses_payload(payload: dict) -> dict:
    """
    Convert Chat Completions payload to Responses API format.

    Chat Completions: { messages: [{role, content}], ... }
    Responses API: { input: [{type: "message", role, content: [...]}], instructions: "system" }
    """
    messages = payload.pop('messages', [])

    system_content = ''
    input_items = []

    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')

        # Check for stored output items (from previous Responses API turn)
        stored_output = msg.get('output')
        if stored_output and isinstance(stored_output, list):
            input_items.extend(_normalize_stored_item(item) for item in stored_output)
            continue

        if role == 'system':
            if isinstance(content, str):
                system_content = content
            elif isinstance(content, list):
                system_content = '\n'.join(p.get('text', '') for p in content if p.get('type') == 'text')
            continue

        # Handle assistant messages with tool_calls (from convert_output_to_messages)
        if role == 'assistant' and msg.get('tool_calls'):
            # Add text content as message if present
            if content:
                text = (
                    content
                    if isinstance(content, str)
                    else '\n'.join(p.get('text', '') for p in content if p.get('type') == 'text')
                )
                if text.strip():
                    input_items.append(
                        {
                            'type': 'message',
                            'role': 'assistant',
                            'content': [{'type': 'output_text', 'text': text}],
                        }
                    )
            # Convert each tool_call to a function_call input item
            for tool_call in msg['tool_calls']:
                func = tool_call.get('function', {})
                input_items.append(
                    {
                        'type': 'function_call',
                        'call_id': tool_call.get('id', ''),
                        'name': func.get('name', ''),
                        'arguments': func.get('arguments', '{}'),
                    }
                )
            continue

        # Handle tool result messages
        if role == 'tool':
            input_items.append(
                {
                    'type': 'function_call_output',
                    'call_id': msg.get('tool_call_id', ''),
                    'output': msg.get('content', ''),
                }
            )
            continue

        # Convert content format
        text_type = 'output_text' if role == 'assistant' else 'input_text'

        if isinstance(content, str):
            content_parts = [{'type': text_type, 'text': content}]
        elif isinstance(content, list):
            content_parts = []
            for part in content:
                part_type = part.get("type")
                if part_type == "text":
                    content_parts.append(
                        {"type": text_type, "text": part.get("text", "")}
                    )
                elif part_type == "image_url":
                    url_data = part.get("image_url", {})
                    url = (
                        url_data.get("url", "")
                        if isinstance(url_data, dict)
                        else url_data
                    )
                    content_parts.append({"type": "input_image", "image_url": url})
                elif part_type in {"input_text", "output_text", "input_image"}:
                    content_parts.append(part)
                elif part_type in {"file", "input_file"}:
                    file_data = part.get("file", {}) if isinstance(part, dict) else {}
                    file_id = part.get("file_id") or file_data.get("file_id")
                    file_url = part.get("file_url") or file_data.get("file_url")
                    if file_id:
                        content_parts.append({"type": "input_file", "file_id": file_id})
                    elif file_url:
                        content_parts.append(
                            {"type": "input_file", "file_url": file_url}
                        )
        else:
            content_parts = [{'type': text_type, 'text': str(content)}]

        input_items.append({'type': 'message', 'role': role, 'content': content_parts})

    responses_payload = {**payload, 'input': input_items}

    # Forward previous_response_id when the middleware has set it
    # (only used when ENABLE_RESPONSES_API_STATEFUL is enabled).
    previous_response_id = responses_payload.pop('previous_response_id', None)
    if previous_response_id:
        responses_payload['previous_response_id'] = previous_response_id

    if system_content:
        responses_payload['instructions'] = system_content

    incoming_params = responses_payload.pop("params", None)
    if isinstance(incoming_params, dict) and incoming_params:
        params = incoming_params.copy()
        system = params.pop("system", None)
        if system and not responses_payload.get("instructions"):
            responses_payload["instructions"] = system
        if params:
            responses_payload = apply_model_params_to_body_responses(
                params, responses_payload
            )

    if "max_tokens" in responses_payload:
        responses_payload["max_output_tokens"] = responses_payload.pop("max_tokens")
    if "max_completion_tokens" in responses_payload:
        responses_payload["max_output_tokens"] = responses_payload.pop(
            "max_completion_tokens"
        )
    if "reasoning_effort" in responses_payload:
        reasoning = responses_payload.get("reasoning") or {}
        if not isinstance(reasoning, dict):
            reasoning = {}
        reasoning["effort"] = responses_payload.pop("reasoning_effort")
        responses_payload["reasoning"] = reasoning

    responses_payload = apply_default_responses_reasoning_summary(
        responses_payload.get("model"),
        responses_payload,
    )

    # Remove Chat Completions-only parameters not supported by the Responses API
    for unsupported_key in (
        "params",
        "files",
        "model_item",
        "parent_message",
        "background_tasks",
        "features",
        "variables",
        "session_id",
        "chat_id",
        "id",
        "parent_id",
        "filter_ids",
        "tool_ids",
        "skill_ids",
        "terminal_id",
        "tool_servers",
        "user",
        "stream_options",
        "logit_bias",
        "frequency_penalty",
        "presence_penalty",
        "stop",
    ):
        responses_payload.pop(unsupported_key, None)

    # Convert Chat Completions tools format to Responses API format
    # Chat Completions: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    # Responses API:    {"type": "function", "name": ..., "description": ..., "parameters": ...}
    if 'tools' in responses_payload and isinstance(responses_payload['tools'], list):
        converted_tools = []
        for tool in responses_payload['tools']:
            if isinstance(tool, dict) and 'function' in tool:
                func = tool['function']
                converted_tool = {'type': tool.get('type', 'function')}
                if isinstance(func, dict):
                    converted_tool['name'] = func.get('name', '')
                    if 'description' in func:
                        converted_tool['description'] = func['description']
                    if 'parameters' in func:
                        converted_tool['parameters'] = func['parameters']
                    if 'strict' in func:
                        converted_tool['strict'] = func['strict']
                converted_tools.append(converted_tool)
            else:
                # Already in correct format or unknown format, pass through
                converted_tools.append(tool)
        responses_payload['tools'] = converted_tools

    return responses_payload


def extract_text_from_response_parts(parts: Optional[list], separator: str = "") -> str:
    texts = []
    for part in parts or []:
        if not isinstance(part, dict):
            continue

        text = part.get("text")
        if isinstance(text, str) and text:
            texts.append(text)

    return separator.join(texts).strip()


def extract_chat_compatible_text_from_responses(response: dict) -> tuple[str, str]:
    output = response.get("output")
    if not isinstance(output, list):
        output = []

    assistant_messages = []
    reasoning_messages = []

    for item in output:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == "message" and item.get("role") == "assistant":
            content = extract_text_from_response_parts(item.get("content"))
            if content:
                assistant_messages.append(content)
        elif item_type == "reasoning":
            summary = extract_text_from_response_parts(
                item.get("summary"), separator="\n\n"
            )
            content = extract_text_from_response_parts(
                item.get("content"), separator="\n\n"
            )
            reasoning_text = summary or content
            if reasoning_text:
                reasoning_messages.append(reasoning_text)

    assistant_content = assistant_messages[-1].strip() if assistant_messages else ""
    reasoning_content = reasoning_messages[-1].strip() if reasoning_messages else ""

    if not assistant_content:
        assistant_content = response.get("output_text", "") or ""

    return assistant_content, reasoning_content


def convert_responses_result(response: dict) -> dict:
    """
    Convert non-streaming Responses API result into Chat Completions-compatible
    shape for existing Open WebUI task and non-stream consumers.
    """
    assistant_content, reasoning_content = extract_chat_compatible_text_from_responses(
        response
    )

    response["choices"] = [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": assistant_content,
                **(
                    {"reasoning_content": reasoning_content}
                    if reasoning_content
                    else {}
                ),
            },
            "finish_reason": "stop",
        }
    ]
    response["done"] = True
    return response
def build_openai_chat_task_metadata(form_data: dict, user: UserModel) -> dict:
    metadata = form_data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    return {
        **metadata,
        "user_id": user.id,
        "chat_id": metadata.get("chat_id") or form_data.get("chat_id"),
        "message_id": metadata.get("message_id") or form_data.get("id"),
        "parent_message": metadata.get("parent_message")
        or form_data.get("parent_message"),
        "parent_message_id": metadata.get("parent_message_id")
        or form_data.get("parent_id"),
        "session_id": metadata.get("session_id") or form_data.get("session_id"),
        "filter_ids": metadata.get("filter_ids", form_data.get("filter_ids", [])),
        "tool_ids": metadata.get("tool_ids", form_data.get("tool_ids")),
        "tool_servers": metadata.get("tool_servers", form_data.get("tool_servers")),
        "files": metadata.get("files", form_data.get("files")),
        "features": metadata.get("features", form_data.get("features", {})),
        "variables": metadata.get("variables", form_data.get("variables", {})),
        "model": metadata.get("model") or form_data.get("model"),
        "params": metadata.get("params", {}),
        "direct": metadata.get("direct", False),
    }


def summarize_error_message(error) -> str:
    if isinstance(error, dict):
        if isinstance(error.get("detail"), str):
            return error["detail"]
        nested_error = error.get("error")
        if isinstance(nested_error, dict):
            if isinstance(nested_error.get("message"), str):
                return nested_error["message"]
            if isinstance(nested_error.get("detail"), str):
                return nested_error["detail"]
        if isinstance(error.get("message"), str):
            return error["message"]
        return json.dumps(error, ensure_ascii=False, default=str)

    return str(error)


async def emit_openai_chat_compat_error(metadata: dict, error) -> None:
    chat_id = metadata.get("chat_id")
    message_id = metadata.get("message_id")
    if not chat_id or not message_id:
        return

    error_message = summarize_error_message(error)

    try:
        from open_webui.models.chats import Chats
        from open_webui.socket.main import get_event_emitter

        if not chat_id.startswith("local:"):
            await Chats.upsert_message_to_chat_by_id_and_message_id(
                chat_id,
                message_id,
                {
                    "error": {"content": error_message},
                },
            )

        event_emitter = get_event_emitter(metadata)
        if event_emitter:
            await event_emitter(
                {
                    "type": "chat:message:error",
                    "data": {"error": {"content": error_message}},
                }
            )
            await event_emitter({"type": "chat:tasks:cancel"})
    except Exception as e:
        log.debug(f"Failed to emit /responses compatibility error: {e}")


async def process_openai_responses_chat_compatibility(
    request: Request,
    form_data: dict,
    user: UserModel,
):
    from open_webui.main import chat_completion as main_chat_completion

    payload = dict(form_data or {})
    return await main_chat_completion(request, payload, user)


@router.post("/chat/completions")
async def generate_chat_completion(
    request: Request,
    form_data: dict,
    user=Depends(get_verified_user),
    bypass_system_prompt: bool = False,
):
    # NOTE: We intentionally do NOT use Depends(get_async_session) here.
    # Database operations (get_model_by_id, AccessGrants.has_access) manage their own short-lived sessions.
    # This prevents holding a connection during the entire LLM call (30-60+ seconds),
    # which would exhaust the connection pool under concurrent load.

    # bypass_filter is read from request.state to prevent external clients from
    # setting it via query parameter (CVE fix). Only internal server-side callers
    # (e.g. utils/chat.py) should set request.state.bypass_filter = True.
    bypass_filter = getattr(request.state, 'bypass_filter', False)
    if BYPASS_MODEL_ACCESS_CONTROL:
        bypass_filter = True

    idx = 0

    payload = {**form_data}
    metadata = payload.pop('metadata', None)

    incoming_params = payload.pop("params", None)
    if isinstance(incoming_params, dict) and incoming_params:
        params = incoming_params.copy()
        system = params.pop("system", None)
        if params:
            payload = apply_model_params_to_body_openai(params, payload)
        if system and not bypass_system_prompt:
            payload = apply_system_prompt_to_body(system, payload, metadata, user)

    model_id = form_data.get("model")
    model_info = await Models.get_model_by_id(model_id)

    # Check model info and override the payload
    if model_info:
        if model_info.base_model_id:
            base_model_id = (
                request.base_model_id if hasattr(request, 'base_model_id') else model_info.base_model_id
            )  # Use request's base_model_id if available
            payload['model'] = base_model_id
            model_id = base_model_id

        params = model_info.params.model_dump()

        if params:
            system = params.pop('system', None)

            payload = apply_model_params_to_body_openai(params, payload)
            if not bypass_system_prompt:
                payload = apply_system_prompt_to_body(system, payload, metadata, user)

        await check_model_access(user, model_info, bypass_filter)
    else:
        await check_model_access(user, None, bypass_filter)

    # Check if model is already in app state cache to avoid expensive get_all_models() call
    models = request.app.state.OPENAI_MODELS
    if not models or model_id not in models:
        await get_all_models(request, user=user)
        models = request.app.state.OPENAI_MODELS
    model = models.get(model_id)

    if model:
        idx = model['urlIdx']
    else:
        raise HTTPException(
            status_code=404,
            detail=ERROR_MESSAGES.MODEL_NOT_FOUND(),
        )

    # Get the API config for the model
    api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
        str(idx),
        request.app.state.config.OPENAI_API_CONFIGS.get(
            request.app.state.config.OPENAI_API_BASE_URLS[idx], {}
        ),  # Legacy support
    )

    prefix_id = api_config.get('prefix_id', None)
    if prefix_id:
        payload['model'] = payload['model'].replace(f'{prefix_id}.', '')

    # Add user info to the payload if the model is a pipeline
    if 'pipeline' in model and model.get('pipeline'):
        payload['user'] = {
            'name': user.name,
            'id': user.id,
            'email': user.email,
            'role': user.role,
        }

    url = request.app.state.config.OPENAI_API_BASE_URLS[idx]
    key = request.app.state.config.OPENAI_API_KEYS[idx]

    # Check if model is a reasoning model that needs special handling
    if is_openai_new_model(payload['model']):
        payload = openai_reasoning_model_handler(payload)
    elif 'api.openai.com' not in url:
        # Remove "max_completion_tokens" from the payload for backward compatibility
        if 'max_completion_tokens' in payload:
            payload['max_tokens'] = payload['max_completion_tokens']
            del payload['max_completion_tokens']

    if 'max_tokens' in payload and 'max_completion_tokens' in payload:
        del payload['max_tokens']

    # Convert the modified body back to JSON
    if 'logit_bias' in payload and payload['logit_bias']:
        logit_bias = convert_logit_bias_input_to_json(payload['logit_bias'])

        if logit_bias:
            payload['logit_bias'] = json.loads(logit_bias)

    headers, cookies = await get_headers_and_cookies(request, url, key, api_config, metadata, user=user)

    is_responses = api_config.get("api_type") == "responses"
    attached_files = []
    payload_before_openai_file_injection = None
    if is_responses:
        attached_files = (
            (metadata or {}).get("files") or payload.get("files", None) or []
        )
        if attached_files:
            payload_before_openai_file_injection = copy.deepcopy(payload)
        payload = await inject_openai_files_into_messages(
            request,
            payload,
            metadata,
            user,
            idx=idx,
        )

    api_version = None
    if api_config.get("azure", False):
        api_version = api_config.get("api_version", "2023-03-15-preview")

    if api_config.get('azure', False):
        # Only set api-key header if not using Azure Entra ID authentication
        auth_type = api_config.get("auth_type", "bearer")
        if auth_type not in ("azure_ad", "microsoft_entra_id"):
            headers["api-key"] = key

        headers["api-version"] = api_version

    def build_openai_request(base_payload: dict) -> tuple[str, str]:
        outbound_payload = copy.deepcopy(base_payload)

        if not is_responses and "messages" in outbound_payload:
            for message in outbound_payload["messages"]:
                if message.get("role") == "tool" and isinstance(message.get("content"), list):
                    message["content"] = "".join(
                        part.get("text", "")
                        for part in message["content"]
                        if part.get("type") in ("input_text", "text")
                    )

        if api_config.get("azure", False):
            request_url_local, outbound_payload = convert_to_azure_payload(
                url, outbound_payload, api_version
            )

            if is_responses:
                outbound_payload = convert_to_responses_payload(outbound_payload)
                request_url_local = (
                    f"{request_url_local}/responses?api-version={api_version}"
                )
            else:
                request_url_local = (
                    f"{request_url_local}/chat/completions?api-version={api_version}"
                )
        else:
            if is_responses:
                outbound_payload = convert_to_responses_payload(outbound_payload)
                request_url_local = f"{url}/responses"
            else:
                request_url_local = f"{url}/chat/completions"

        return request_url_local, json.dumps(outbound_payload)

    request_url, payload = build_openai_request(payload)

    r = None
    session = None
    streaming = False
    response = None

    try:
        retry_attempted = False

        while True:
            session = aiohttp.ClientSession(
                trust_env=True,
                timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT),
            )

            r = await session.request(
                method="POST",
                url=request_url,
                data=payload,
                headers=headers,
                cookies=cookies,
                ssl=AIOHTTP_CLIENT_SESSION_SSL,
            )

            # Check if response is SSE
            if "text/event-stream" in r.headers.get("Content-Type", ""):
                streaming = True
                return StreamingResponse(
                    stream_wrapper(r, session, stream_sse_lines),
                    status_code=r.status,
                    headers=dict(r.headers),
                )

            try:
                response = await r.json()
            except Exception as e:
                log.error(e)
                response = await r.text()

            if (
                r.status >= 400
                and is_responses
                and attached_files
                and payload_before_openai_file_injection is not None
                and not retry_attempted
                and is_unknown_file_id_error(response)
            ):
                log.info(
                    "Retrying /responses request after stale upstream file id for chat_id=%s",
                    (metadata or {}).get("chat_id"),
                )
                retry_attempted = True
                await cleanup_response(r, session)
                r = None
                session = None

                invalidate_cached_openai_file_ids(attached_files)
                refreshed_payload = await inject_openai_files_into_messages(
                    request,
                    copy.deepcopy(payload_before_openai_file_injection),
                    metadata,
                    user,
                    idx=idx,
                )
                request_url, payload = build_openai_request(refreshed_payload)
                continue

            if r.status >= 400:
                if isinstance(response, (dict, list)):
                    return JSONResponse(status_code=r.status, content=response)
                else:
                    return PlainTextResponse(status_code=r.status, content=response)

            # Convert Responses API result to simple format
            if is_responses and isinstance(response, dict):
                response = convert_responses_result(response)

            return response
    except Exception as e:
        log.exception(e)

        raise HTTPException(
            status_code=r.status if r else 500,
            detail=ERROR_MESSAGES.SERVER_CONNECTION_ERROR,
        )
    finally:
        if not streaming:
            await cleanup_response(r, session)


async def embeddings(request: Request, form_data: dict, user):
    """
    Calls the embeddings endpoint for OpenAI-compatible providers.

    Args:
        request (Request): The FastAPI request context.
        form_data (dict): OpenAI-compatible embeddings payload.
        user (UserModel): The authenticated user.

    Returns:
        dict: OpenAI-compatible embeddings response.
    """
    idx = 0
    # Prepare payload/body
    body = json.dumps(form_data)
    # Find correct backend url/key based on model
    model_id = form_data.get('model')
    # Check if model is already in app state cache to avoid expensive get_all_models() call
    models = request.app.state.OPENAI_MODELS
    if not models or model_id not in models:
        await get_all_models(request, user=user)
        models = request.app.state.OPENAI_MODELS
    if model_id in models:
        idx = models[model_id]['urlIdx']

    url = request.app.state.config.OPENAI_API_BASE_URLS[idx]
    key = request.app.state.config.OPENAI_API_KEYS[idx]
    api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
        str(idx),
        request.app.state.config.OPENAI_API_CONFIGS.get(url, {}),  # Legacy support
    )

    r = None
    session = None
    streaming = False

    headers, cookies = await get_headers_and_cookies(request, url, key, api_config, user=user)
    try:
        session = await get_session()
        r = await session.request(
            method='POST',
            url=f'{url}/embeddings',
            data=body,
            headers=headers,
            cookies=cookies,
            timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT),
            ssl=AIOHTTP_CLIENT_SESSION_SSL,
        )

        if 'text/event-stream' in r.headers.get('Content-Type', ''):
            streaming = True
            return StreamingResponse(
                stream_wrapper(r, session, stream_sse_lines),
                status_code=r.status,
                headers=dict(r.headers),
            )
        else:
            try:
                response_data = await r.json()
            except Exception:
                response_data = await r.text()

            if r.status >= 400:
                if isinstance(response_data, (dict, list)):
                    return JSONResponse(status_code=r.status, content=response_data)
                else:
                    return PlainTextResponse(status_code=r.status, content=response_data)

            return response_data
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=r.status if r else 500,
            detail=ERROR_MESSAGES.SERVER_CONNECTION_ERROR,
        )
    finally:
        if not streaming:
            await cleanup_response(r, session)


class ResponsesForm(BaseModel):
    model_config = ConfigDict(extra='allow')

    model: str
    input: Optional[list | str] = None
    instructions: Optional[str] = None
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    top_p: Optional[float] = None
    tools: Optional[list] = None
    tool_choice: Optional[str | dict] = None
    text: Optional[dict] = None
    truncation: Optional[str] = None
    metadata: Optional[dict] = None
    store: Optional[bool] = None
    reasoning: Optional[dict] = None
    previous_response_id: Optional[str] = None


@router.post('/responses')
async def responses(
    request: Request,
    form_data: dict,
    user=Depends(get_verified_user),
):
    """
    Forward requests to the OpenAI Responses API endpoint.
    Routes to the correct upstream backend based on the model field.

    If the caller still sends Chat Completions-shaped payloads, reuse the
    existing chat route so the browser can switch to /responses without
    breaking the current frontend request shape.
    """
    payload = dict(form_data or {})
    raw_payload_keys = sorted(payload.keys())
    debug_info = {
        "raw_payload_keys": raw_payload_keys,
    }

    if "messages" in payload:
        return await process_openai_responses_chat_compatibility(
            request, payload, user
        )

    # Some callers still send Chat-Completions-style tuning knobs nested under
    # `params`, which the Responses API rejects as an unsupported top-level key.
    # Normalize those values into the Responses payload and drop `params`.
    incoming_params = payload.pop("params", None)
    debug_info["incoming_params"] = summarize_response_debug_value(incoming_params)
    if isinstance(incoming_params, dict) and incoming_params:
        payload = apply_model_params_to_body_responses(incoming_params, payload)

    legacy_params = {}
    if "reasoning_effort" in payload:
        legacy_params["reasoning_effort"] = payload.pop("reasoning_effort")
    if "max_tokens" in payload:
        legacy_params["max_tokens"] = payload.pop("max_tokens")
    if legacy_params:
        payload = apply_model_params_to_body_responses(legacy_params, payload)

    model_id = payload.get("model", "")
    debug_info["model_id"] = model_id
    model_info = await Models.get_model_by_id(model_id)

    if model_info:
        debug_info["model_found"] = True
        if model_info.base_model_id:
            base_model_id = (
                request.base_model_id
                if hasattr(request, "base_model_id")
                else model_info.base_model_id
            )
            payload["model"] = base_model_id
            model_id = base_model_id
            debug_info["resolved_model_id"] = model_id

        params = model_info.params.model_dump()
        debug_info["model_params"] = summarize_response_debug_value(params)
        if params:
            payload = apply_model_params_to_body_responses(params, payload)

        if not BYPASS_MODEL_ACCESS_CONTROL and user.role == "user":
            user_group_ids = {
                group.id for group in await Groups.get_groups_by_member_id(user.id)
            }
            if not (
                user.id == model_info.user_id
                or await AccessGrants.has_access(
                    user_id=user.id,
                    resource_type="model",
                    resource_id=model_info.id,
                    permission="read",
                    user_group_ids=user_group_ids,
                )
            ):
                raise HTTPException(
                    status_code=403,
                    detail="Model not found",
                )
    elif not BYPASS_MODEL_ACCESS_CONTROL and user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Model not found",
        )
    else:
        debug_info["model_found"] = False

    payload = apply_default_responses_reasoning_summary(model_id, payload)

    debug_info["final_payload_keys"] = sorted(payload.keys())
    debug_info["final_reasoning"] = summarize_response_debug_value(
        payload.get("reasoning")
    )
    debug_info["final_has_params"] = "params" in payload
    log.info(
        "responses_debug request %s",
        json.dumps(debug_info, ensure_ascii=False, default=str),
    )

    body = json.dumps(payload)

    idx = 0
    if model_id:
        models = request.app.state.OPENAI_MODELS
        if not models or model_id not in models:
            await get_all_models(request, user=user)
            models = request.app.state.OPENAI_MODELS
        if model_id in models:
            idx = models[model_id]['urlIdx']

    url = request.app.state.config.OPENAI_API_BASE_URLS[idx]
    key = request.app.state.config.OPENAI_API_KEYS[idx]
    api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
        str(idx),
        request.app.state.config.OPENAI_API_CONFIGS.get(url, {}),  # Legacy support
    )

    r = None
    session = None
    streaming = False

    try:
        headers, cookies = await get_headers_and_cookies(request, url, key, api_config, user=user)

        if api_config.get('azure', False):
            auth_type = api_config.get('auth_type', 'bearer')
            if auth_type not in ('azure_ad', 'microsoft_entra_id'):
                headers['api-key'] = key

            is_azure_v1 = bool(re.search(r'/openai/v1(?:/|$)', url))

            if is_azure_v1:
                request_url = f'{url.rstrip("/")}/responses'
            else:
                api_version = api_config.get('api_version', '2023-03-15-preview')
                headers['api-version'] = api_version
                model = _sanitize_model_for_url(payload.get('model', ''))
                request_url = f'{url}/openai/deployments/{model}/responses?api-version={api_version}'
        else:
            request_url = f'{url}/responses'

        session = await get_session()
        r = await session.request(
            method='POST',
            url=request_url,
            data=body,
            headers=headers,
            cookies=cookies,
            ssl=AIOHTTP_CLIENT_SESSION_SSL,
            timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT),
        )

        # Check if response is SSE
        if 'text/event-stream' in r.headers.get('Content-Type', ''):
            streaming = True
            return StreamingResponse(
                stream_wrapper(r, session, stream_sse_lines),
                status_code=r.status,
                headers=dict(r.headers),
            )
        else:
            try:
                response_data = await r.json()
            except Exception:
                response_data = await r.text()

            if r.status >= 400:
                log.info(
                    "responses_debug error %s",
                    json.dumps(
                        {
                            **debug_info,
                            "status": r.status,
                            "request_url": request_url,
                            "error_response": summarize_response_debug_value(
                                response_data
                            ),
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                )
                if isinstance(response_data, (dict, list)):
                    return JSONResponse(status_code=r.status, content=response_data)
                else:
                    return PlainTextResponse(status_code=r.status, content=response_data)

            return response_data

    except HTTPException:
        raise
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=r.status if r else 500,
            detail=ERROR_MESSAGES.SERVER_CONNECTION_ERROR,
        )
    finally:
        if not streaming:
            await cleanup_response(r, session)


@router.api_route('/{path:path}', methods=['GET', 'POST', 'PUT', 'DELETE'])
async def proxy(path: str, request: Request, user=Depends(get_verified_user)):
    """
    Deprecated: proxy all requests to OpenAI API.
    Disabled by default. Set ENABLE_OPENAI_API_PASSTHROUGH=True to enable.
    """

    if not ENABLE_OPENAI_API_PASSTHROUGH:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Direct API passthrough is disabled. Set ENABLE_OPENAI_API_PASSTHROUGH=True to enable.',
        )

    body = await request.body()

    # Parse JSON body to resolve model-based routing
    payload = None
    if body:
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            payload = None

    idx = 0
    model_id = payload.get('model') if isinstance(payload, dict) else None
    if model_id:
        models = request.app.state.OPENAI_MODELS
        if not models or model_id not in models:
            await get_all_models(request, user=user)
            models = request.app.state.OPENAI_MODELS
        if model_id in models:
            idx = models[model_id]['urlIdx']

    url = request.app.state.config.OPENAI_API_BASE_URLS[idx]
    key = request.app.state.config.OPENAI_API_KEYS[idx]
    api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
        str(idx),
        request.app.state.config.OPENAI_API_CONFIGS.get(
            request.app.state.config.OPENAI_API_BASE_URLS[idx], {}
        ),  # Legacy support
    )

    r = None
    session = None
    streaming = False

    try:
        headers, cookies = await get_headers_and_cookies(request, url, key, api_config, user=user)

        if api_config.get('azure', False):
            # Only set api-key header if not using Azure Entra ID authentication
            auth_type = api_config.get('auth_type', 'bearer')
            if auth_type not in ('azure_ad', 'microsoft_entra_id'):
                headers['api-key'] = key

            is_azure_v1 = bool(re.search(r'/openai/v1(?:/|$)', url))

            if is_azure_v1:
                qs = request.url.query
                request_url = f'{url.rstrip("/")}/{path}' + (f'?{qs}' if qs else '')
            else:
                api_version = api_config.get('api_version', '2023-03-15-preview')
                headers['api-version'] = api_version

                payload = json.loads(body)
                url, payload = convert_to_azure_payload(url, payload, api_version)
                body = json.dumps(payload).encode()

                request_url = f'{url}/{path}?api-version={api_version}'
        else:
            request_url = f'{url}/{path}'

        session = await get_session()
        r = await session.request(
            method=request.method,
            url=request_url,
            data=body,
            headers=headers,
            cookies=cookies,
            ssl=AIOHTTP_CLIENT_SESSION_SSL,
            timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT),
        )

        # Check if response is SSE
        if 'text/event-stream' in r.headers.get('Content-Type', ''):
            streaming = True
            return StreamingResponse(
                stream_wrapper(r, session, stream_sse_lines),
                status_code=r.status,
                headers=dict(r.headers),
            )
        else:
            try:
                response_data = await r.json()
            except Exception:
                response_data = await r.text()

            if r.status >= 400:
                if isinstance(response_data, (dict, list)):
                    return JSONResponse(status_code=r.status, content=response_data)
                else:
                    return PlainTextResponse(status_code=r.status, content=response_data)

            return response_data

    except HTTPException:
        raise
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=r.status if r else 500,
            detail='Open WebUI: Server Connection Error',
        )
    finally:
        if not streaming:
            await cleanup_response(r, session)
