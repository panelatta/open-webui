from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import time
import re
from typing import Optional
from urllib.parse import quote, urlparse

import aiofiles
import aiohttp
from aiocache import cached
import requests
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from open_webui.config import (
    CACHE_DIR,
)
from open_webui.constants import ERROR_MESSAGES
from open_webui.env import (
    AIOHTTP_CLIENT_SESSION_SSL,
    AIOHTTP_CLIENT_TIMEOUT,
    AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST,
    BYPASS_MODEL_ACCESS_CONTROL,
    ENABLE_FORWARD_USER_INFO_HEADERS,
    ENABLE_OPENAI_API_PASSTHROUGH,
    ENABLE_RESPONSES_API_BACKGROUND_RESUME,
    RESPONSES_API_BACKGROUND_RESUME_BASE_URL_ALLOWLIST,
    FORWARD_SESSION_INFO_HEADER_CHAT_ID,
    MODELS_CACHE_TTL,
    RESPONSES_API_BACKGROUND_RESUME_MODEL_ALLOWLIST,
    REDIS_KEY_PREFIX,
)
from open_webui.events import EVENTS, publish_event, publish_model_provider_request_failed
from open_webui.internal.db import get_async_session
from open_webui.models.access_grants import AccessGrants
from open_webui.models.config import Config
from open_webui.models.files import Files
from open_webui.models.groups import Groups
from open_webui.models.models import Models
from open_webui.models.users import UserModel
from open_webui.utils.access_control import check_model_access, has_connection_access, has_permission
from open_webui.utils.anthropic import ANTHROPIC_VERSION, get_anthropic_models, is_anthropic_url
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.embedding_policy import EMBEDDING_DISABLED_MESSAGE
from open_webui.utils.headers import get_custom_headers, include_user_info_headers
from open_webui.utils.json_codec import JSONCodec
from open_webui.utils.misc import convert_logit_bias_input_to_json
from open_webui.utils.model_ids import strip_provider_model_prefix
from open_webui.utils.payload import (
    apply_model_params_to_body_openai,
    apply_system_prompt_to_body,
)
from open_webui.utils.session_pool import (
    cleanup_response,
    get_client_timeout,
    get_session,
    stream_wrapper,
)
from open_webui.storage.provider import Storage
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)


##########################################
#
# Utility functions
# Let the responses returned through this gate be worth
# the question that summoned them.
#
##########################################

# Headers that become stale after aiohttp auto-decompresses the upstream
# response body.  Forwarding them verbatim causes desktop / programmatic
# clients to attempt decompression of an already-decoded payload, resulting
# in ZlibError.  See https://github.com/aio-libs/aiohttp/issues/4462.
_STRIP_PROXY_HEADERS = frozenset({'Content-Encoding', 'Content-Length', 'Transfer-Encoding'})
_MODEL_LIST_TIMEOUT = aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST)
_UNSUPPORTED_OPENAI_MODEL_KEYWORDS = ('babbage', 'dall-e', 'davinci', 'embedding', 'tts', 'whisper')
BASE_MODELS_CACHE_KEY = f'{REDIS_KEY_PREFIX}:models:base'


def _clean_proxy_headers(raw_headers) -> dict:
    """Return a copy of *raw_headers* with stale encoding headers removed."""
    return {k: v for k, v in raw_headers.items() if k not in _STRIP_PROXY_HEADERS}


async def send_get_request(
    request: Request = None,
    url=None,
    key=None,
    user: UserModel = None,
    config=None,
):
    try:
        async with aiohttp.ClientSession(timeout=_MODEL_LIST_TIMEOUT, trust_env=True) as session:
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
                return await response.json(loads=JSONCodec.loads)
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


async def get_headers_and_cookies(
    request: Request,
    url,
    key=None,
    config=None,
    metadata: dict | None = None,
    user: UserModel = None,
):
    cookies = {}
    headers = {
        'Content-Type': 'application/json',
        **(
            {
                # LICENSE covers this Open WebUI upstream metadata identifier.
                # Do not alter, remove, obscure, or replace it except as LICENSE permits:
                # https://docs.openwebui.com/license.
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
        custom_headers = await get_custom_headers(config.get('headers'), user, metadata, request=request)
        headers.update(custom_headers)

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
    url: str,
    key: str,
    api_config: dict,
    metadata: Optional[dict] = None,
    user: Optional[UserModel] = None,
) -> str:
    data = file_item.data or {}

    if (
        data.get("openai_file_id")
        and data.get("openai_file_id_valid_for_next_request") is True
        and data.get("openai_backend_index") == idx
        and data.get("openai_api_base_url") == url
    ):
        return data["openai_file_id"]

    if api_config.get("azure") or api_config.get("provider") == "azure":
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

    url, key, api_config = await get_openai_connection(idx)
    openai_file_id = await asyncio.to_thread(
        upload_local_file_to_openai,
        request,
        file_item,
        idx=idx,
        url=url,
        key=key,
        api_config=api_config,
        metadata=metadata,
        user=user,
    )

    if openai_file_id:
        await Files.update_file_data_by_id(
            file_item.id,
            {
                "openai_file_id": openai_file_id,
                "openai_backend_index": idx,
                "openai_api_base_url": url,
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


##########################################
#
# API routes
#
##########################################

router = APIRouter()

LLAMACPP_LOADED_STATES = {'loaded', 'sleeping'}
LLAMACPP_UNLOADED_STATES = {'loading', 'unloaded'}
MODEL_MANAGEMENT_ENDPOINTS = {
    'llama.cpp': {
        'list': '/models',
        'download': '/models',
        'delete': '/models',
        'load': '/models/load',
        'unload': '/models/unload',
        'sse': '/models/sse',
    },
    'lmstudio': {
        'list': '/api/v1/models',
        'download': '/api/v1/models/download',
        'download_status': '/api/v1/models/download/status/{job_id}',
        'load': '/api/v1/models/load',
        'unload': '/api/v1/models/unload',
    },
}


def get_model_management_root_url(url: str, provider: str) -> str:
    root_url = url.rstrip('/')
    if provider in ('llama.cpp', 'lmstudio'):
        for suffix in ('/api/v1', '/api/v0', '/v1'):
            if root_url.endswith(suffix):
                return root_url.removesuffix(suffix)

    return root_url


def get_provider_model_loaded_state(model: dict, provider: str, manual_model_ids: bool = False) -> bool | None:
    if provider == 'lmstudio':
        if model.get('loaded_instances'):
            return True

        state = model.get('state')
        if state == 'loaded':
            return True
        if state == 'not-loaded':
            return False

        return None

    if provider != 'llama.cpp':
        return None

    status = model.get('status')
    if isinstance(status, dict):
        value = status.get('value')
        if value in LLAMACPP_LOADED_STATES:
            return True
        if value in LLAMACPP_UNLOADED_STATES:
            return False

    if not manual_model_ids and 'status' not in model:
        return True

    return None


OPENAI_CONFIG_KEYS = {
    'ENABLE_OPENAI_API': 'openai.enable',
    'OPENAI_API_BASE_URLS': 'openai.api_base_urls',
    'OPENAI_API_KEYS': 'openai.api_keys',
    'OPENAI_API_CONFIGS': 'openai.api_configs',
}


async def get_openai_config() -> dict:
    values = await Config.get_many(*OPENAI_CONFIG_KEYS.values())
    return {field: values[storage_key] for field, storage_key in OPENAI_CONFIG_KEYS.items() if storage_key in values}


async def get_openai_runtime_config() -> tuple[bool, list[str], list[str], dict]:
    values = await Config.get_many('openai.enable', 'openai.api_base_urls', 'openai.api_keys', 'openai.api_configs')
    return (
        values.get('openai.enable'),
        values.get('openai.api_base_urls') or [],
        values.get('openai.api_keys') or [],
        values.get('openai.api_configs') or {},
    )


async def normalize_openai_api_keys(api_base_urls: list[str], api_keys: list[str]) -> list[str]:
    if len(api_keys) > len(api_base_urls):
        api_keys = api_keys[: len(api_base_urls)]
    elif len(api_keys) < len(api_base_urls):
        api_keys = [*api_keys, *([''] * (len(api_base_urls) - len(api_keys)))]

    await Config.upsert({'openai.api_keys': api_keys})
    return api_keys


async def get_openai_connection(idx: int) -> tuple[str, str, dict]:
    _, api_base_urls, api_keys, api_configs = await get_openai_runtime_config()
    url = api_base_urls[idx]
    key = api_keys[idx]
    api_config = api_configs.get(str(idx), api_configs.get(url, {}))
    return url, key, api_config


async def clear_openai_model_cache(request: Request):
    await get_all_models.cache.clear()
    redis = getattr(request.app.state, 'redis', None)
    if redis is not None:
        await redis.delete(BASE_MODELS_CACHE_KEY)
    request.app.state.BASE_MODELS = []
    request.app.state.OPENAI_MODELS = {}
    models = getattr(request.app.state, 'MODELS', None)
    if hasattr(models, 'clear'):
        models.clear()
    else:
        request.app.state.MODELS = {}


async def get_model_management_connection(url_idx: int) -> tuple[str, str, dict, str]:
    if not await Config.get('openai.enable'):
        raise HTTPException(status_code=503, detail='OpenAI API is disabled')

    try:
        url, key, api_config = await get_openai_connection(url_idx)
    except IndexError:
        raise HTTPException(status_code=404, detail='Connection not found')

    provider = api_config.get('provider', '')
    if provider not in MODEL_MANAGEMENT_ENDPOINTS:
        raise HTTPException(
            status_code=400,
            detail=f'Provider "{provider or "default"}" does not support model management',
        )

    return get_model_management_root_url(url, provider), key, api_config, provider


def get_model_management_path(provider: str, operation: str, path_params: dict | None = None) -> str:
    try:
        path = MODEL_MANAGEMENT_ENDPOINTS[provider][operation]
    except KeyError:
        raise HTTPException(status_code=400, detail=f'Provider "{provider}" does not support {operation}')

    return path.format(**(path_params or {}))


def get_model_management_payload(provider: str, operation: str, payload: dict | None) -> dict | None:
    if provider == 'lmstudio' and operation == 'unload' and payload:
        return {'instance_id': payload.get('instance_id') or payload.get('model')}

    return payload


async def send_model_management_request(
    request: Request,
    url_idx: int,
    operation: str,
    method: str = 'GET',
    payload: dict | None = None,
    query: dict | None = None,
    path_params: dict | None = None,
    stream: bool = False,
    user: UserModel | None = None,
):
    root_url, key, api_config, provider = await get_model_management_connection(url_idx)
    path = get_model_management_path(provider, operation, path_params=path_params)
    payload = get_model_management_payload(provider, operation, payload)
    headers, cookies = await get_headers_and_cookies(request, root_url, key, api_config, user=user)

    response = None
    streaming = False
    try:
        session = await get_session()
        response = await session.request(
            method,
            f'{root_url}{path}',
            json=payload,
            params=query,
            headers=headers,
            cookies=cookies,
            ssl=AIOHTTP_CLIENT_SESSION_SSL,
            timeout=get_client_timeout(stream=stream),
        )

        if not response.ok:
            try:
                error = await response.json(loads=JSONCodec.loads)
            except Exception:
                error = await response.text()
            raise HTTPException(status_code=response.status, detail=error)

        if stream:
            streaming = True
            return StreamingResponse(
                stream_wrapper(response, passthrough=True),
                status_code=response.status,
                headers=_clean_proxy_headers(response.headers),
            )

        try:
            return await response.json(loads=JSONCodec.loads)
        except Exception:
            return {'success': True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=response.status if response else 500, detail=str(e))
    finally:
        if not streaming:
            await cleanup_response(response)


async def get_anthropic_request_target(request: Request, form_data: dict, user: UserModel):
    """Resolve the upstream connection, payload and auth headers for a native Anthropic request."""
    requested_model = form_data.get('model')
    if not requested_model:
        raise HTTPException(status_code=400, detail='model is required')

    payload = {**form_data}
    model_id = requested_model
    model_info = await Models.get_model_by_id(model_id)
    await check_model_access(user, model_info, BYPASS_MODEL_ACCESS_CONTROL)

    if model_info and model_info.base_model_id:
        model_id = model_info.base_model_id
        payload['model'] = model_id

    models = request.app.state.OPENAI_MODELS
    if not models or model_id not in models:
        await get_all_models(request, user=user)
        models = request.app.state.OPENAI_MODELS

    model = models.get(model_id)
    if not model or 'urlIdx' not in model:
        raise HTTPException(status_code=404, detail=ERROR_MESSAGES.MODEL_NOT_FOUND())

    url, key, api_config = await get_openai_connection(model['urlIdx'])
    prefix_id = api_config.get('prefix_id')
    payload['model'] = strip_provider_model_prefix(payload['model'], prefix_id)

    headers, cookies = await get_headers_and_cookies(request, url, key, api_config, user=user)

    # Anthropic's native endpoints reject bearer auth, the key belongs in x-api-key.
    if is_anthropic_url(url):
        headers.setdefault('anthropic-version', ANTHROPIC_VERSION)
        if api_config.get('auth_type') in (None, 'bearer'):
            headers.pop('Authorization', None)
            headers.setdefault('x-api-key', key)

    return requested_model, payload, url, key, headers, cookies


async def count_anthropic_tokens(request: Request, form_data: dict, user: UserModel) -> int:
    """Forward an Anthropic token-count request through an OpenAI-compatible connection."""
    requested_model, payload, url, key, headers, cookies = await get_anthropic_request_target(request, form_data, user)
    request_url = f'{url.rstrip("/")}/messages/count_tokens'
    response = None

    try:
        session = await get_session()
        response = await session.request(
            method='POST',
            url=request_url,
            data=JSONCodec.dumps(payload),
            headers=headers,
            cookies=cookies,
            ssl=AIOHTTP_CLIENT_SESSION_SSL,
            timeout=get_client_timeout(),
        )

        try:
            response_data = await response.json(loads=JSONCodec.loads)
        except Exception:
            response_data = await response.text()

        if response.status >= 400:
            await publish_model_provider_request_failed(
                request,
                actor=user,
                provider='openai-compatible',
                base_url=url,
                api_key=key,
                status=response.status,
                requested_model=requested_model,
                upstream_error=response_data,
            )
            raise HTTPException(status_code=response.status, detail=response_data)

        input_tokens = response_data.get('input_tokens') if isinstance(response_data, dict) else None
        if isinstance(input_tokens, bool) or not isinstance(input_tokens, int) or input_tokens < 0:
            raise HTTPException(status_code=502, detail='Invalid token-count response from upstream provider')

        return input_tokens
    except HTTPException:
        raise
    except Exception:
        log.exception('Failed to count Anthropic tokens for model %s', requested_model)
        raise HTTPException(status_code=502, detail=ERROR_MESSAGES.SERVER_CONNECTION_ERROR)
    finally:
        await cleanup_response(response)


@router.get('/config')
async def get_config(request: Request, user=Depends(get_admin_user)):
    return await get_openai_config()


class OpenAIConfigForm(BaseModel):
    ENABLE_OPENAI_API: bool | None = None
    OPENAI_API_BASE_URLS: list[str]
    OPENAI_API_KEYS: list[str]
    OPENAI_API_CONFIGS: dict


@router.post('/config/update')
async def update_config(request: Request, form_data: OpenAIConfigForm, user=Depends(get_admin_user)):
    api_keys = form_data.OPENAI_API_KEYS

    if len(api_keys) > len(form_data.OPENAI_API_BASE_URLS):
        api_keys = api_keys[: len(form_data.OPENAI_API_BASE_URLS)]
    elif len(api_keys) < len(form_data.OPENAI_API_BASE_URLS):
        api_keys = [*api_keys, *([''] * (len(form_data.OPENAI_API_BASE_URLS) - len(api_keys)))]

    valid_keys = set(map(str, range(len(form_data.OPENAI_API_BASE_URLS))))
    api_configs = {key: value for key, value in form_data.OPENAI_API_CONFIGS.items() if key in valid_keys}

    await Config.upsert(
        {
            'openai.enable': form_data.ENABLE_OPENAI_API,
            'openai.api_base_urls': form_data.OPENAI_API_BASE_URLS,
            'openai.api_keys': api_keys,
            'openai.api_configs': api_configs,
        }
    )

    await clear_openai_model_cache(request)

    await publish_event(
        request,
        EVENTS.MODEL_PROVIDER_CONFIG_UPDATED,
        actor=user,
        subject_id='openai',
        subject_type='model.provider_config',
        data={
            'provider': 'openai',
            'enabled': form_data.ENABLE_OPENAI_API,
            'base_url_count': len(form_data.OPENAI_API_BASE_URLS),
        },
    )

    return {
        'ENABLE_OPENAI_API': form_data.ENABLE_OPENAI_API,
        'OPENAI_API_BASE_URLS': form_data.OPENAI_API_BASE_URLS,
        'OPENAI_API_KEYS': api_keys,
        'OPENAI_API_CONFIGS': api_configs,
    }


@router.post('/audio/speech')
async def speech(request: Request, user=Depends(get_verified_user)):
    if user.role != 'admin' and not await has_permission(user.id, 'chat.tts', await Config.get('user.permissions')):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    idx = None
    try:
        _, api_base_urls, _, _ = await get_openai_runtime_config()
        idx = api_base_urls.index('https://api.openai.com/v1')

        body = await request.body()
        name = hashlib.sha256(body).hexdigest()

        SPEECH_CACHE_DIR = CACHE_DIR / 'audio' / 'speech'
        SPEECH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        file_path = SPEECH_CACHE_DIR.joinpath(f'{name}.mp3')
        file_body_path = SPEECH_CACHE_DIR.joinpath(f'{name}.json')

        # Check if the file already exists in the cache
        if file_path.is_file():
            return FileResponse(file_path)

        url, key, api_config = await get_openai_connection(idx)

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

            async with aiofiles.open(file_path, 'wb') as f:
                async for chunk in r.content.iter_chunked(8192):
                    await f.write(chunk)

            async with aiofiles.open(file_body_path, 'w') as f:
                await f.write(JSONCodec.dumps(JSONCodec.loads(body.decode('utf-8'))))

            # Return the saved file
            return FileResponse(file_path)

        except Exception as e:
            log.exception(e)

            detail = None
            if r is not None:
                try:
                    res = await r.json(loads=JSONCodec.loads)
                    if 'error' in res:
                        detail = f'External: {res["error"]}'
                except Exception:
                    detail = f'External: {e}'

            # LICENSE covers this Open WebUI error identifier.
            # Do not alter, remove, obscure, or replace it except as LICENSE permits:
            # https://docs.openwebui.com/license.
            raise HTTPException(
                status_code=r.status if r else 500,
                detail=detail if detail else 'Open WebUI: Server Connection Error',
            )

    except ValueError:
        raise HTTPException(status_code=401, detail=ERROR_MESSAGES.OPENAI_NOT_FOUND)


async def get_all_models_responses(request: Request, user: UserModel) -> list:
    enable_openai_api, api_base_urls, api_keys, api_configs = await get_openai_runtime_config()
    if not enable_openai_api:
        return []

    num_urls = len(api_base_urls)
    num_keys = len(api_keys)

    if num_keys != num_urls:
        api_keys = await normalize_openai_api_keys(api_base_urls, api_keys)

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
            provider = api_config.get('provider', '')

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
                    if model.get('name'):
                        model['name'] = f'{prefix_id}.{model["name"]}'

                if tags:
                    model['tags'] = tags

                if connection_type:
                    model['connection_type'] = connection_type

                if provider:
                    model['provider'] = provider

    log.debug('get_all_models:responses() %s', responses)
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
    # key_builder (not key) is the per-call hook in aiocache 0.12; `key=` is a
    # static key, so a `key=lambda` collapsed every caller to one shared entry.
    key_builder=lambda _func, request, user=None: f'openai_all_models_{user.id}' if user else 'openai_all_models',
)
async def get_all_models(request: Request, user: UserModel) -> dict[str, list]:
    log.info('get_all_models()')

    enable_openai_api, api_base_urls, _, api_configs = await get_openai_runtime_config()
    if not enable_openai_api:
        request.app.state.OPENAI_MODELS = {}
        return {'data': []}

    responses = await get_all_models_responses(request, user=user)

    def extract_data(response):
        if response and 'data' in response:
            return response['data']
        if isinstance(response, list):
            return response
        return None

    def is_supported_openai_models(model_id):
        return not any(name in model_id for name in _UNSUPPORTED_OPENAI_MODEL_KEYWORDS)

    def get_merged_models(model_lists):
        log.debug('merge_models_lists %s', model_lists)
        models = {}

        for idx, model_list in enumerate(model_lists):
            if model_list is not None and 'error' not in model_list:
                base_url = api_base_urls[idx]
                hostname = urlparse(base_url).hostname if base_url else None
                api_config = api_configs.get(str(idx), api_configs.get(base_url, {}))

                for model in model_list:
                    model_id = model.get('id') or model.get('name')

                    if hostname == 'api.openai.com' and not is_supported_openai_models(model_id):
                        # Skip unwanted OpenAI models
                        continue

                    if model_id and model_id not in models:
                        provider = model.get('provider', '')
                        merged = {
                            **model,
                            'name': model.get('name', model_id),
                            'owned_by': 'openai',
                            'openai': model,
                            'connection_type': model.get('connection_type', 'external'),
                            'provider': provider,
                            'urlIdx': idx,
                        }

                        loaded = get_provider_model_loaded_state(
                            model,
                            provider,
                            manual_model_ids=bool(api_config.get('model_ids')),
                        )
                        if loaded is not None:
                            merged['loaded'] = loaded

                        models[model_id] = merged

        return models

    models = get_merged_models(map(extract_data, responses))
    log.debug('models: %s', models)

    request.app.state.OPENAI_MODELS = models
    return {'data': list(models.values())}


@router.get('/models')
@router.get('/models/{url_idx}', dependencies=[Depends(get_admin_user)])
async def get_models(request: Request, url_idx: int | None = None, user=Depends(get_verified_user)):
    if not await Config.get('openai.enable'):
        raise HTTPException(status_code=503, detail='OpenAI API is disabled')

    models = {
        'data': [],
    }

    if url_idx is None:
        models = await get_all_models(request, user=user)
    else:
        url, key, api_config = await get_openai_connection(url_idx)

        r = None
        async with aiohttp.ClientSession(
            trust_env=True,
            timeout=_MODEL_LIST_TIMEOUT,
        ) as session:
            try:
                headers, cookies = await get_headers_and_cookies(request, url, key, api_config, user=user)

                if api_config.get('azure') or api_config.get('provider') == 'azure':
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
                                res = await r.json(loads=JSONCodec.loads)
                                if 'error' in res:
                                    error_detail = f'External Error: {res["error"]}'
                            except Exception:
                                pass
                            raise Exception(error_detail)

                        response_data = await r.json(loads=JSONCodec.loads)

                        if 'api.openai.com' in url:
                            response_data['data'] = [
                                model
                                for model in response_data.get('data', [])
                                if not any(name in model['id'] for name in _UNSUPPORTED_OPENAI_MODEL_KEYWORDS)
                            ]

                        models = response_data
            except aiohttp.ClientError as e:
                # ClientError covers all aiohttp requests issues
                log.exception(f'Client error: {str(e)}')
                # LICENSE covers this Open WebUI error identifier.
                # Do not alter, remove, obscure, or replace it except as LICENSE permits:
                # https://docs.openwebui.com/license.
                raise HTTPException(status_code=500, detail='Open WebUI: Server Connection Error')
            except Exception as e:
                log.exception(f'Unexpected error: {e}')
                error_detail = f'Unexpected error: {str(e)}'
                raise HTTPException(status_code=500, detail=error_detail)

    if user.role == 'user' and not BYPASS_MODEL_ACCESS_CONTROL:
        models['data'] = await get_filtered_models(models, user)

    return models


class ProviderModelOperationForm(BaseModel):
    model: str
    model_config = ConfigDict(extra='allow')


@router.get('/models/{url_idx}/catalog')
async def get_provider_model_catalog(request: Request, url_idx: int, user=Depends(get_admin_user)):
    return await send_model_management_request(request, url_idx, 'list', user=user)


@router.post('/models/{url_idx}/download')
async def download_provider_model(
    request: Request,
    url_idx: int,
    form_data: ProviderModelOperationForm,
    user=Depends(get_admin_user),
):
    root_url, _, api_config, provider = await get_model_management_connection(url_idx)
    payload = form_data.model_dump(exclude_none=True)
    payload['model'] = strip_provider_model_prefix(payload['model'], api_config.get('prefix_id'))

    result = await send_model_management_request(request, url_idx, 'download', 'POST', payload, user=user)
    await clear_openai_model_cache(request)
    await publish_event(
        request,
        EVENTS.MODEL_PROVIDER_MODEL_CREATED,
        actor=user,
        subject_id=payload['model'],
        data={'provider': provider, 'url_idx': url_idx, 'base_url': root_url},
    )
    return result


@router.get('/models/{url_idx}/download/status/{job_id}')
async def get_provider_model_download_status(
    request: Request,
    url_idx: int,
    job_id: str,
    user=Depends(get_admin_user),
):
    return await send_model_management_request(
        request,
        url_idx,
        'download_status',
        path_params={'job_id': job_id},
        user=user,
    )


@router.post('/models/{url_idx}/load')
async def load_provider_model(
    request: Request,
    url_idx: int,
    form_data: ProviderModelOperationForm,
    user=Depends(get_admin_user),
):
    _, _, api_config, _ = await get_model_management_connection(url_idx)
    payload = form_data.model_dump(exclude_none=True)
    payload['model'] = strip_provider_model_prefix(payload['model'], api_config.get('prefix_id'))

    result = await send_model_management_request(request, url_idx, 'load', 'POST', payload, user=user)
    await clear_openai_model_cache(request)
    return result


@router.post('/models/{url_idx}/unload')
async def unload_provider_model(
    request: Request,
    url_idx: int,
    form_data: ProviderModelOperationForm,
    user=Depends(get_admin_user),
):
    _, _, api_config, _ = await get_model_management_connection(url_idx)
    payload = form_data.model_dump(exclude_none=True)
    payload['model'] = strip_provider_model_prefix(payload['model'], api_config.get('prefix_id'))

    result = await send_model_management_request(request, url_idx, 'unload', 'POST', payload, user=user)
    await clear_openai_model_cache(request)
    return result


@router.get('/models/{url_idx}/sse')
async def stream_provider_model_events(request: Request, url_idx: int, user=Depends(get_admin_user)):
    return await send_model_management_request(request, url_idx, 'sse', stream=True, user=user)


@router.delete('/models/{url_idx}')
async def delete_provider_model(
    request: Request,
    url_idx: int,
    model: str,
    user=Depends(get_admin_user),
):
    root_url, _, api_config, provider = await get_model_management_connection(url_idx)
    actual_model = strip_provider_model_prefix(model, api_config.get('prefix_id'))

    result = await send_model_management_request(
        request,
        url_idx,
        'delete',
        'DELETE',
        query={'model': actual_model},
        user=user,
    )
    await clear_openai_model_cache(request)
    await publish_event(
        request,
        EVENTS.MODEL_PROVIDER_MODEL_DELETED,
        actor=user,
        subject_id=actual_model,
        data={'provider': provider, 'url_idx': url_idx, 'base_url': root_url},
    )
    return result


class ConnectionVerificationForm(BaseModel):
    url: str
    key: str

    config: dict | None = None


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
        timeout=_MODEL_LIST_TIMEOUT,
    ) as session:
        try:
            headers, cookies = await get_headers_and_cookies(request, url, key, api_config, user=user)

            if api_config.get('azure') or api_config.get('provider') == 'azure':
                # Only set api-key header if not using Azure Entra ID authentication
                auth_type = api_config.get('auth_type', 'bearer')
                if auth_type not in ('azure_ad', 'microsoft_entra_id'):
                    headers['api-key'] = key

                # Azure v1 format: base URL already ends with /openai/v1,
                # use standard /models endpoint without api-version.
                is_azure_v1 = bool(re.search(r'/openai/v1(?:/|$)', url))

                if is_azure_v1:
                    verify_url = f'{url.rstrip("/")}/models'
                else:
                    api_version = api_config.get('api_version', '') or '2023-03-15-preview'
                    verify_url = f'{url}/openai/models?api-version={api_version}'

                async with session.get(
                    url=verify_url,
                    headers=headers,
                    cookies=cookies,
                    ssl=AIOHTTP_CLIENT_SESSION_SSL,
                ) as r:
                    try:
                        response_data = await r.json(loads=JSONCodec.loads)
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
                        response_data = await r.json(loads=JSONCodec.loads)
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
        log.debug('Invalid API version %s for Azure OpenAI. Defaulting to allowed parameters.', api_version)

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
                'Removing temperature parameter for o-series model %s as only default value (1) is supported', model
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


def _responses_background_resume_model_ids(*candidates) -> list[str]:
    model_ids = []

    def add(candidate):
        if isinstance(candidate, (list, tuple, set)):
            for item in candidate:
                add(item)
            return

        if not candidate:
            return

        candidate = str(candidate)
        if candidate and candidate not in model_ids:
            model_ids.append(candidate)

    for candidate in candidates:
        add(candidate)

    return model_ids


def responses_background_resume_enabled(model_id: str | list | tuple | set, url: str, api_config: dict) -> bool:
    if not ENABLE_RESPONSES_API_BACKGROUND_RESUME:
        return False

    model_ids = _responses_background_resume_model_ids(model_id)

    # Azure retrieve/resume path support varies by deployment. Require an
    # explicit per-route opt-in instead of enabling through broad allowlists.
    if (api_config.get('azure') or api_config.get('provider') == 'azure') and (
        api_config.get('responses_background_resume') is not True
    ):
        return False

    if api_config.get('responses_background_resume') is True:
        return True

    if any(candidate in RESPONSES_API_BACKGROUND_RESUME_MODEL_ALLOWLIST for candidate in model_ids):
        return True

    normalized_url = (url or '').rstrip('/')
    if normalized_url and normalized_url in RESPONSES_API_BACKGROUND_RESUME_BASE_URL_ALLOWLIST:
        return True

    return False


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

        if role in ('system', 'developer'):
            if isinstance(content, str):
                system_content = content
            elif isinstance(content, list):
                system_content = '\n'.join(p.get('text', '') for p in content if p.get('type') in {'text', 'input_text'})
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
                if part.get('type') == 'text':
                    content_parts.append({'type': text_type, 'text': part.get('text', '')})
                elif part.get('type') == 'image_url':
                    url_data = part.get('image_url', {})
                    if isinstance(url_data, dict):
                        url = url_data.get('url', '')
                        detail = url_data.get('detail') or 'auto'
                    else:
                        url = url_data if isinstance(url_data, str) else ''
                        detail = 'auto'
                    content_parts.append({'type': 'input_image', 'image_url': url, 'detail': detail})
                elif part.get('type') in {'input_file', 'file'}:
                    input_file = {'type': 'input_file'}

                    file_id = part.get('file_id')
                    nested_file = part.get('file')
                    if not file_id and isinstance(nested_file, dict):
                        file_id = nested_file.get('file_id')

                    if file_id:
                        input_file['file_id'] = file_id
                    if part.get('file_data'):
                        input_file['file_data'] = part.get('file_data')
                    elif isinstance(nested_file, dict) and nested_file.get('file_data'):
                        input_file['file_data'] = nested_file['file_data']
                    if part.get('file_url'):
                        input_file['file_url'] = part.get('file_url')

                    filename = part.get('filename')
                    if not filename and isinstance(nested_file, dict):
                        filename = nested_file.get('filename')
                    if filename:
                        input_file['filename'] = filename

                    if len(input_file) > 1:
                        content_parts.append(input_file)
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

    if 'max_tokens' in responses_payload:
        responses_payload['max_output_tokens'] = responses_payload.pop('max_tokens')

    if 'max_completion_tokens' in responses_payload:
        responses_payload['max_output_tokens'] = responses_payload.pop('max_completion_tokens')

    # Remove Chat Completions-only parameters not supported by the Responses API
    for unsupported_key in (
        'stream_options',
        'logit_bias',
        'frequency_penalty',
        'presence_penalty',
        'stop',
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
    Convert non-streaming Responses API result to Chat Completions format.

    Extracts text from message output items so all downstream consumers
    (frontend tasks, get_content_from_response) work without modification.
    """
    output_items = response.get('output', [])

    content = ''
    for item in output_items:
        if item.get('type') == 'message':
            for part in item.get('content', []):
                if part.get('type') == 'output_text':
                    content += part.get('text', '')

    return {
        'id': response.get('id', ''),
        'object': 'chat.completion',
        'model': response.get('model', ''),
        'choices': [
            {
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': content,
                },
                'finish_reason': 'stop',
            }
        ],
        'usage': response.get('usage', {}),
    }


@router.post('/chat/completions')
async def generate_chat_completion(
    request: Request,
    form_data: dict,
    user=Depends(get_verified_user),
):
    if not await Config.get('openai.enable'):
        raise HTTPException(status_code=503, detail='OpenAI API is disabled')

    # NOTE: We intentionally do NOT use Depends(get_async_session) here.
    # Database operations (get_model_by_id, AccessGrants.has_access) manage their own short-lived sessions.
    # This prevents holding a connection during the entire LLM call (30-60+ seconds),
    # which would exhaust the connection pool under concurrent load.

    # bypass_filter and bypass_system_prompt are read from request.state to prevent
    # external clients from setting them via query parameter. Only internal
    # server-side callers (e.g. utils/chat.py) should set
    # request.state.bypass_filter / request.state.bypass_system_prompt = True.
    bypass_filter = getattr(request.state, 'bypass_filter', False)
    if BYPASS_MODEL_ACCESS_CONTROL:
        bypass_filter = True
    bypass_system_prompt = getattr(request.state, 'bypass_system_prompt', False)

    idx = 0

    payload = {**form_data}
    metadata = payload.pop('metadata', None)

    model_id = form_data.get('model')
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
                payload = await apply_system_prompt_to_body(system, payload, metadata, user)

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

    url, key, api_config = await get_openai_connection(idx)

    prefix_id = api_config.get('prefix_id', None)
    payload['model'] = strip_provider_model_prefix(payload['model'], prefix_id)

    # Add user info to the payload if the model is a pipeline
    if 'pipeline' in model and model.get('pipeline'):
        payload['user'] = {
            'name': user.name,
            'id': user.id,
            'email': user.email,
            'role': user.role,
        }

    background_resume_model_ids = _responses_background_resume_model_ids(
        form_data.get('model'),
        getattr(model_info, 'base_model_id', None) if model_info else None,
        model_id,
    )

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
            payload['logit_bias'] = JSONCodec.loads(logit_bias)

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

    is_azure_v1 = False
    api_version = None
    is_azure_route = api_config.get("azure") or api_config.get("provider") == "azure"
    if is_azure_route:
        api_version = api_config.get("api_version", "2023-03-15-preview")
        is_azure_v1 = bool(re.search(r"/openai/v1(?:/|$)", url))

        # Only set api-key header if not using Azure Entra ID authentication
        auth_type = api_config.get("auth_type", "bearer")
        if auth_type not in ("azure_ad", "microsoft_entra_id"):
            headers["api-key"] = key

        if not is_azure_v1:
            headers["api-version"] = api_version

    def build_openai_request(base_payload: dict) -> tuple[str, str]:
        outbound_payload = copy.deepcopy(base_payload)

        def apply_background_resume_flags(payload: dict) -> dict:
            if payload.get('stream') and responses_background_resume_enabled(
                [*background_resume_model_ids, payload.get('model', '')],
                url,
                api_config,
            ):
                payload['background'] = True
                payload['store'] = True
            return payload

        if not is_responses and "messages" in outbound_payload:
            for message in outbound_payload["messages"]:
                if message.get("role") == "tool" and isinstance(message.get("content"), list):
                    message["content"] = "".join(
                        part.get("text", "")
                        for part in message["content"]
                        if part.get("type") in ("input_text", "text")
                    )

        if is_azure_route:
            if is_azure_v1:
                if is_responses:
                    outbound_payload = convert_to_responses_payload(outbound_payload)
                    outbound_payload = apply_background_resume_flags(outbound_payload)
                    request_url_local = f'{url.rstrip("/")}/responses'
                else:
                    request_url_local = f'{url.rstrip("/")}/chat/completions'
            else:
                request_url_local, outbound_payload = convert_to_azure_payload(
                    url, outbound_payload, api_version
                )

                if is_responses:
                    outbound_payload = convert_to_responses_payload(outbound_payload)
                    outbound_payload = apply_background_resume_flags(outbound_payload)
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
                outbound_payload = apply_background_resume_flags(outbound_payload)
                request_url_local = f"{url}/responses"
            else:
                request_url_local = f"{url}/chat/completions"

        return request_url_local, json.dumps(outbound_payload)

    is_streaming_request = bool(payload.get('stream', False))
    if not is_streaming_request:
        payload.pop('stream_options', None)

    requested_model = payload.get("model")
    request_url, payload = build_openai_request(payload)

    r = None
    streaming = False
    response = None

    try:
        retry_attempted = False

        while True:
            session = await get_session()

            r = await session.request(
                method="POST",
                url=request_url,
                data=payload,
                headers=headers,
                cookies=cookies,
                ssl=AIOHTTP_CLIENT_SESSION_SSL,
                timeout=get_client_timeout(stream=is_streaming_request),
            )

            # Check if response is SSE
            if "text/event-stream" in r.headers.get("Content-Type", ""):
                # If the provider returned an error status with SSE content-type,
                # read the body and return a proper error response instead of
                # streaming the error back (which hides the error from logs).
                if r.status >= 400:
                    error_body = await r.text()
                    log.error(
                        "Provider returned HTTP %d with SSE content-type: %s",
                        r.status,
                        error_body[:1000],
                    )
                    try:
                        error_json = json.loads(error_body)
                        await publish_model_provider_request_failed(
                            request,
                            actor=user,
                            provider='openai-compatible',
                            base_url=url,
                            api_key=key,
                            status=r.status,
                            requested_model=requested_model,
                            upstream_error=error_json,
                        )
                        return JSONResponse(status_code=r.status, content=error_json)
                    except json.JSONDecodeError:
                        await publish_model_provider_request_failed(
                            request,
                            actor=user,
                            provider='openai-compatible',
                            base_url=url,
                            api_key=key,
                            status=r.status,
                            requested_model=requested_model,
                            upstream_error=error_body,
                        )
                        return JSONResponse(
                            status_code=r.status,
                            content={
                                "error": {
                                    "message": error_body,
                                    "code": r.status,
                                }
                            },
                        )

                streaming = True
                response_headers = _clean_proxy_headers(r.headers)
                if is_responses:
                    response_headers['x-openwebui-openai-url-idx'] = str(idx)
                    response_headers['x-openwebui-openai-base-url'] = url
                return StreamingResponse(
                    stream_wrapper(r),
                    status_code=r.status,
                    headers=response_headers,
                )

            try:
                response = await r.json(loads=JSONCodec.loads)
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
                await cleanup_response(r)
                r = None

                await invalidate_cached_openai_file_ids(attached_files)
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
                await publish_model_provider_request_failed(
                    request,
                    actor=user,
                    provider='openai-compatible',
                    base_url=url,
                    api_key=key,
                    status=r.status,
                    requested_model=requested_model,
                    upstream_error=response,
                )
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
            await cleanup_response(r)


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
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=EMBEDDING_DISABLED_MESSAGE)

    idx = 0
    # Prepare payload/body
    body = JSONCodec.dumps(form_data)
    # Find correct backend url/key based on model
    model_id = form_data.get('model')
    # Check if model is already in app state cache to avoid expensive get_all_models() call
    models = request.app.state.OPENAI_MODELS
    if not models or model_id not in models:
        await get_all_models(request, user=user)
        models = request.app.state.OPENAI_MODELS
    if model_id in models:
        idx = models[model_id]['urlIdx']

    url, key, api_config = await get_openai_connection(idx)

    r = None
    streaming = False

    headers, cookies = await get_headers_and_cookies(request, url, key, api_config, user=user)

    if api_config.get('azure') or api_config.get('provider') == 'azure':
        # Only set api-key header if not using Azure Entra ID authentication
        auth_type = api_config.get('auth_type', 'bearer')
        if auth_type not in ('azure_ad', 'microsoft_entra_id'):
            headers['api-key'] = key

        # Azure v1 format: base URL already ends with /openai/v1,
        # model stays in the payload, no deployment URL rewriting.
        is_azure_v1 = bool(re.search(r'/openai/v1(?:/|$)', url))

        if is_azure_v1:
            embeddings_url = f'{url.rstrip("/")}/embeddings'
        else:
            api_version = api_config.get('api_version', '2023-03-15-preview')
            model = _sanitize_model_for_url(form_data.get('model', ''))
            embeddings_url = f'{url}/openai/deployments/{model}/embeddings?api-version={api_version}'
            headers['api-version'] = api_version
    else:
        embeddings_url = f'{url}/embeddings'
    requested_model = form_data.get('model')

    try:
        session = await get_session()
        r = await session.request(
            method='POST',
            url=embeddings_url,
            data=body,
            headers=headers,
            cookies=cookies,
            timeout=get_client_timeout(),
            ssl=AIOHTTP_CLIENT_SESSION_SSL,
        )

        if 'text/event-stream' in r.headers.get('Content-Type', ''):
            streaming = True
            return StreamingResponse(
                stream_wrapper(r, passthrough=True),
                status_code=r.status,
                headers=_clean_proxy_headers(r.headers),
            )
        else:
            try:
                response_data = await r.json(loads=JSONCodec.loads)
            except Exception:
                response_data = await r.text()

            if r.status >= 400:
                await publish_model_provider_request_failed(
                    request,
                    actor=user,
                    provider='openai-compatible',
                    base_url=url,
                    api_key=key,
                    status=r.status,
                    requested_model=requested_model,
                    upstream_error=response_data,
                )
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
            await cleanup_response(r)


class ResponsesForm(BaseModel):
    model_config = ConfigDict(extra='allow')

    model: str
    input: list | str | None = None
    instructions: str | None = None
    stream: bool | None = None
    background: bool | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    top_p: float | None = None
    tools: list | None = None
    tool_choice: str | dict | None = None
    text: dict | None = None
    truncation: str | None = None
    metadata: dict | None = None
    store: bool | None = None
    reasoning: dict | None = None
    previous_response_id: str | None = None


async def resolve_openai_route(
    request: Request,
    model_id: str | None,
    user,
    route_idx: int | None = None,
):
    if route_idx is not None:
        state_config = getattr(getattr(request.app, 'state', None), 'config', None)
        if state_config is not None and hasattr(state_config, 'OPENAI_API_BASE_URLS'):
            urls = list(getattr(state_config, 'OPENAI_API_BASE_URLS', []) or [])
            keys = list(getattr(state_config, 'OPENAI_API_KEYS', []) or [])
            api_configs = getattr(state_config, 'OPENAI_API_CONFIGS', {}) or {}
        else:
            _, urls, keys, api_configs = await get_openai_runtime_config()

        if route_idx < 0 or route_idx >= len(urls):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Stored response route is invalid; cannot resume response stream.',
            )
        idx = route_idx
        if len(keys) < len(urls):
            keys = [*keys, *([''] * (len(urls) - len(keys)))]
        url = urls[idx]
        key = keys[idx]
        api_config = api_configs.get(str(idx), api_configs.get(url, {})) if isinstance(api_configs, dict) else {}
        return idx, url, key, api_config
    else:
        if not model_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Model or stored response route is required to resolve upstream provider.',
            )

        models = request.app.state.OPENAI_MODELS
        if not models or model_id not in models:
            await get_all_models(request, user=user)
            models = request.app.state.OPENAI_MODELS

        if model_id not in models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Model route not found; refusing to default to upstream index 0.',
            )

        idx = models[model_id]['urlIdx']

    url, key, api_config = await get_openai_connection(idx)

    return idx, url, key, api_config


async def resume_response_stream(
    request: Request,
    model_id: str,
    response_id: str,
    starting_after: int | None,
    route_idx: int | None,
    user,
):
    if not ENABLE_RESPONSES_API_BACKGROUND_RESUME:
        return None

    model_info = await Models.get_model_by_id(model_id)
    await check_model_access(user, model_info, BYPASS_MODEL_ACCESS_CONTROL)

    idx, url, key, api_config = await resolve_openai_route(
        request,
        model_id=model_id,
        user=user,
        route_idx=route_idx,
    )

    if not responses_background_resume_enabled(
        _responses_background_resume_model_ids(
            model_id,
            getattr(model_info, 'base_model_id', None) if model_info else None,
        ),
        url,
        api_config,
    ):
        log.info('Responses background resume is not enabled for model=%s url=%s', model_id, url)
        return None

    headers, cookies = await get_headers_and_cookies(request, url, key, api_config, user=user)

    if api_config.get('azure') or api_config.get('provider') == 'azure':
        log.info('Responses background resume is disabled for Azure routes in this patch')
        return None

    params = {'stream': 'true'}
    if starting_after is not None:
        params['starting_after'] = str(starting_after)

    request_url = f'{url}/responses/{quote(response_id, safe="")}'

    r = None
    streaming = False
    try:
        session = await get_session()
        r = await session.request(
            method='GET',
            url=request_url,
            params=params,
            headers=headers,
            cookies=cookies,
            ssl=AIOHTTP_CLIENT_SESSION_SSL,
            timeout=get_client_timeout(stream=True),
        )

        if 'text/event-stream' in r.headers.get('Content-Type', '') and r.status < 400:
            streaming = True
            response_headers = _clean_proxy_headers(r.headers)
            response_headers['x-openwebui-openai-url-idx'] = str(idx)
            response_headers['x-openwebui-openai-base-url'] = url
            return StreamingResponse(
                stream_wrapper(r),
                status_code=r.status,
                headers=response_headers,
            )

        try:
            response_data = await r.json()
        except Exception:
            response_data = await r.text()

        log.info(
            'Responses background resume returned non-stream response '
            '(status=%s, response_id=%s): %s',
            r.status,
            response_id,
            summarize_response_debug_value(response_data),
        )

        return JSONResponse(status_code=r.status, content=response_data)
    except Exception:
        log.exception('Responses background resume failed for %s', response_id)
        return None
    finally:
        if not streaming:
            await cleanup_response(r)


@router.get('/responses/{response_id}')
async def get_response(
    response_id: str,
    request: Request,
    model: str,
    route_idx: Optional[int] = None,
    stream: Optional[bool] = None,
    starting_after: Optional[int] = None,
    user=Depends(get_verified_user),
):
    """
    Retrieve or resume a background Responses API response.

    Prefer route_idx from stored message metadata when available. The model is
    still required for access control and as a route fallback for manual calls.
    """
    model_info = await Models.get_model_by_id(model)
    await check_model_access(user, model_info, BYPASS_MODEL_ACCESS_CONTROL)

    idx, url, key, api_config = await resolve_openai_route(
        request,
        model_id=model,
        user=user,
        route_idx=route_idx,
    )

    if not responses_background_resume_enabled(
        _responses_background_resume_model_ids(
            model,
            getattr(model_info, 'base_model_id', None) if model_info else None,
        ),
        url,
        api_config,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Responses background retrieve/resume is not enabled for this provider.',
        )

    headers, cookies = await get_headers_and_cookies(request, url, key, api_config, user=user)

    if api_config.get('azure') or api_config.get('provider') == 'azure':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Responses background retrieve/resume is not implemented for Azure routes.',
        )

    params = {}
    if stream is not None:
        params['stream'] = 'true' if stream else 'false'
    if starting_after is not None:
        params['starting_after'] = str(starting_after)

    request_url = f'{url}/responses/{quote(response_id, safe="")}'

    r = None
    streaming = False
    try:
        session = await get_session()
        r = await session.request(
            method='GET',
            url=request_url,
            params=params,
            headers=headers,
            cookies=cookies,
            ssl=AIOHTTP_CLIENT_SESSION_SSL,
            timeout=get_client_timeout(stream=stream is True),
        )

        if 'text/event-stream' in r.headers.get('Content-Type', ''):
            streaming = True
            response_headers = _clean_proxy_headers(r.headers)
            response_headers['x-openwebui-openai-url-idx'] = str(idx)
            response_headers['x-openwebui-openai-base-url'] = url
            return StreamingResponse(
                stream_wrapper(r),
                status_code=r.status,
                headers=response_headers,
            )

        try:
            response_data = await r.json()
        except Exception:
            response_data = await r.text()

        if r.status >= 400:
            if isinstance(response_data, (dict, list)):
                return JSONResponse(status_code=r.status, content=response_data)
            return PlainTextResponse(status_code=r.status, content=response_data)

        return response_data
    finally:
        if not streaming:
            await cleanup_response(r)


@router.post('/responses')
async def responses(
    request: Request,
    form_data: ResponsesForm,
    user=Depends(get_verified_user),
):
    """
    Forward requests to the OpenAI Responses API endpoint.
    Routes to the correct upstream backend based on the model field.

    If the caller still sends Chat Completions-shaped payloads, reuse the
    existing chat route so the browser can switch to /responses without
    breaking the current frontend request shape.
    """
    payload = form_data.model_dump(exclude_none=True)
    is_streaming_request = bool(payload.get('stream', False))

    model_id = form_data.model

    # Enforce per-model access control
    model_info = await Models.get_model_by_id(model_id)
    await check_model_access(user, model_info, BYPASS_MODEL_ACCESS_CONTROL)

    idx, url, key, api_config = await resolve_openai_route(
        request,
        model_id=model_id,
        user=user,
    )

    if payload.get('stream') and responses_background_resume_enabled(
        _responses_background_resume_model_ids(
            model_id,
            getattr(model_info, 'base_model_id', None) if model_info else None,
            payload.get('model', ''),
        ),
        url,
        api_config,
    ):
        payload['background'] = True
        payload['store'] = True

    payload['model'] = strip_provider_model_prefix(payload['model'], api_config.get('prefix_id'))
    body = JSONCodec.dumps(payload)

    r = None
    streaming = False
    debug_info = {
        'model': model_id,
        'url_idx': idx,
        'url': url,
    }

    try:
        headers, cookies = await get_headers_and_cookies(request, url, key, api_config, user=user)

        if api_config.get('azure') or api_config.get('provider') == 'azure':
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
            timeout=get_client_timeout(stream=is_streaming_request),
        )

        # Check if response is SSE
        if 'text/event-stream' in r.headers.get('Content-Type', ''):
            streaming = True
            response_headers = _clean_proxy_headers(r.headers)
            response_headers['x-openwebui-openai-url-idx'] = str(idx)
            response_headers['x-openwebui-openai-base-url'] = url
            return StreamingResponse(
                stream_wrapper(r, passthrough=True),
                status_code=r.status,
                headers=response_headers,
            )
        else:
            try:
                response_data = await r.json(loads=JSONCodec.loads)
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
                await publish_model_provider_request_failed(
                    request,
                    actor=user,
                    provider='openai-compatible',
                    base_url=url,
                    api_key=key,
                    status=r.status,
                    requested_model=payload.get('model'),
                    upstream_error=response_data,
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
            await cleanup_response(r)


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
            payload = JSONCodec.loads(body)
        except (JSONCodec.JSONDecodeError, ValueError):
            payload = None
    is_streaming_request = bool(payload.get('stream', False)) if isinstance(payload, dict) else False

    idx = 0
    model_id = payload.get('model') if isinstance(payload, dict) else None
    if model_id:
        models = request.app.state.OPENAI_MODELS
        if not models or model_id not in models:
            await get_all_models(request, user=user)
            models = request.app.state.OPENAI_MODELS
        if model_id in models:
            idx = models[model_id]['urlIdx']

    url, key, api_config = await get_openai_connection(idx)
    base_url = url

    r = None
    streaming = False

    try:
        headers, cookies = await get_headers_and_cookies(request, url, key, api_config, user=user)

        if api_config.get('azure') or api_config.get('provider') == 'azure':
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

                payload = JSONCodec.loads(body)
                url, payload = convert_to_azure_payload(url, payload, api_version)
                body = JSONCodec.dumps(payload).encode()

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
            timeout=get_client_timeout(stream=is_streaming_request),
        )

        # Check if response is SSE
        if 'text/event-stream' in r.headers.get('Content-Type', ''):
            streaming = True
            return StreamingResponse(
                stream_wrapper(r, passthrough=True),
                status_code=r.status,
                headers=_clean_proxy_headers(r.headers),
            )
        else:
            try:
                response_data = await r.json(loads=JSONCodec.loads)
            except Exception:
                response_data = await r.text()

            if r.status >= 400:
                await publish_model_provider_request_failed(
                    request,
                    actor=user,
                    provider='openai-compatible',
                    base_url=base_url,
                    api_key=key,
                    status=r.status,
                    requested_model=model_id,
                    upstream_error=response_data,
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
        # LICENSE covers this Open WebUI error identifier.
        # Do not alter, remove, obscure, or replace it except as LICENSE permits:
        # https://docs.openwebui.com/license.
        raise HTTPException(
            status_code=r.status if r else 500,
            detail='Open WebUI: Server Connection Error',
        )
    finally:
        if not streaming:
            await cleanup_response(r)
