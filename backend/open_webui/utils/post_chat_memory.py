import json
import logging
import os
import re
from typing import Any, Optional

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from open_webui.models.memories import Memories
from open_webui.routers.memories import (
    AddMemoryForm,
    MemoryUpdateModel,
    QueryMemoryForm,
    add_memory,
    query_memory,
    update_memory_by_id,
)
from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.misc import get_last_user_message

log = logging.getLogger(__name__)

DEFAULT_ENABLED_MODEL_IDS = {'gpt-immanuel-pastor'}
DEFAULT_FALLBACK_EXTRACTOR_MODEL_ID = 'gpt-5.5'
EXTRACTOR_TASK_NAME = 'post_chat_memory_extraction'


def _enabled_model_ids() -> set[str]:
    configured = os.getenv('POST_CHAT_MEMORY_EXTRACTOR_MODEL_IDS')
    if configured is None:
        return DEFAULT_ENABLED_MODEL_IDS
    return {item.strip() for item in configured.split(',') if item.strip()}


def _max_memories_per_turn() -> int:
    try:
        return max(1, min(5, int(os.getenv('POST_CHAT_MEMORY_EXTRACTOR_MAX_PER_TURN', '3'))))
    except ValueError:
        return 3


def _normalize_memory(content: str) -> str:
    return re.sub(r'\s+', ' ', content or '').strip().lower()


EXPLICIT_MEMORY_PATTERNS = (
    r'^记住(?:[:：，,\s]|这个|这件事|一点|一下)',
    r'^(?:请|麻烦)(?:你)?(?:帮我)?(?:记住|记得)',
    r'^(?:请)?帮我记',
    r'^(?:你)?以后(?:要)?记得',
    r'^(?:别忘|不要忘)',
    r'^更新\s*memory',
    r'^更新记忆',
)


def _explicit_memory_request(messages: list[dict]) -> Optional[str]:
    latest = get_last_user_message(messages) or ''
    if not latest:
        return None

    normalized_latest = re.sub(r'\s+', ' ', latest).strip()
    if not any(re.search(pattern, normalized_latest, flags=re.I) for pattern in EXPLICIT_MEMORY_PATTERNS):
        return None

    normalized_latest = re.sub(
        r'^(?:(?:请|麻烦)(?:你)?(?:帮我)?(?:记住|记得)|(?:请)?帮我记(?:住|得|一下|一点)?|记住(?:这个|这件事|一点|一下)?|(?:你)?以后(?:要)?记得|别忘|不要忘|更新\s*memory|更新记忆)[:：，,\s]*',
        '',
        normalized_latest,
        flags=re.I,
    ).strip()
    if not normalized_latest:
        return None

    if len(normalized_latest) > 240:
        normalized_latest = normalized_latest[:240].rstrip() + '...'

    return f'User 明确要求记住：{normalized_latest}'


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text':
                    parts.append(str(item.get('text', '')))
                elif item.get('type') == 'image_url':
                    parts.append('[image]')
        return '\n'.join(part for part in parts if part)
    return ''


def _compact_messages(messages: list[dict], max_messages: int = 10, max_chars: int = 9000) -> str:
    compact = []
    for message in messages[-max_messages:]:
        role = message.get('role', 'unknown')
        content = _message_content_to_text(message.get('content', ''))
        content = re.sub(r'<details\b[^>]*>.*?</details>', '', content, flags=re.S | re.I)
        content = re.sub(r'\s+', ' ', content).strip()
        if not content:
            continue
        compact.append(f'{role}: {content}')

    transcript = '\n'.join(compact)
    if len(transcript) > max_chars:
        transcript = transcript[-max_chars:]
    return transcript


def _extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _response_content(response: Any) -> str:
    if isinstance(response, JSONResponse):
        try:
            response = json.loads(response.body.decode('utf-8', 'replace'))
        except Exception:
            return ''

    if not isinstance(response, dict):
        return ''

    choices = response.get('choices') or []
    if choices:
        message = choices[0].get('message') or {}
        return message.get('content') or message.get('reasoning_content') or ''

    return response.get('content') or response.get('reasoning_content') or ''


async def _add_memory_content(request, user, content: str, existing_contents: Optional[set[str]] = None) -> bool:
    content = (content or '').strip()
    normalized = _normalize_memory(content)
    if not normalized:
        return False

    if existing_contents is None:
        all_existing = await Memories.get_memories_by_user_id(user.id)
        existing_contents = {_normalize_memory(memory.content) for memory in all_existing}

    if normalized in existing_contents:
        return False

    memory = await add_memory(request, AddMemoryForm(content=content), user)
    if not memory:
        return False

    existing_contents.add(normalized)
    return True


def _select_extractor_model_id(model: dict, models: dict) -> Optional[str]:
    configured = os.getenv('POST_CHAT_MEMORY_EXTRACTOR_MODEL')
    if configured and configured in models:
        return configured

    base_model_id = (model.get('info') or {}).get('base_model_id')
    if base_model_id and base_model_id in models:
        return base_model_id

    if DEFAULT_FALLBACK_EXTRACTOR_MODEL_ID in models:
        return DEFAULT_FALLBACK_EXTRACTOR_MODEL_ID

    model_id = model.get('id') or (model.get('info') or {}).get('id')
    return model_id if model_id in models else None


async def _get_relevant_memories(request, user, messages: list[dict]) -> list[dict]:
    query = get_last_user_message(messages) or _compact_messages(messages, max_messages=4, max_chars=2000)
    if not query:
        return []

    try:
        results = await query_memory(request, QueryMemoryForm(content=query, k=8), user)
    except HTTPException as e:
        if e.status_code == 404:
            return []
        raise
    except Exception as e:
        log.debug(f'Post-chat memory query failed: {e}')
        return []

    memories = []
    documents = getattr(results, 'documents', None) or []
    ids = getattr(results, 'ids', None) or []
    if not documents:
        return []

    for idx, content in enumerate(documents[0] or []):
        memory_id = ids[0][idx] if ids and ids[0] and idx < len(ids[0]) else None
        if memory_id and content:
            memories.append({'id': memory_id, 'content': content})
    return memories


def _build_extraction_prompt(
    transcript: str,
    relevant_memories: list[dict],
    max_memories: int,
) -> str:
    existing = '\n'.join(
        f'- id={memory["id"]}: {memory["content"]}' for memory in relevant_memories
    ) or '(none)'

    return f'''你是 Open WebUI 后台 post-chat memory extractor。你的任务是从刚完成的一轮牧养对话中，判断是否需要更新长期 memory。你不是聊天助手，不要安慰用户，只输出 JSON。

你每一轮都要主动评估是否有值得保存的新信息。只保存会影响未来长期陪伴的信息，例如：反复触发点、重要关系/工作/家庭事件、长期偏好、长期属灵分辨方式、用户明确要求以后记住的内容。不要保存一次性寒暄、当天短暂状态、普通回复内容、未经确认的猜测，或只适合当前对话的细节。用户明确说“请记住”“以后记得”“别忘”“更新 memory/记忆”时，必须输出 add 或 update。

如果新信息只是补充已有 memory，请在 updates 中用已有 id 输出合并后的完整 content；只有在确实是新的长期背景时才放入 adds。每条 content 用简洁中文第三人称，以 "User ..." 开头。最多输出 {max_memories} 条 add/update 总和。避免和已有 memory 语义重复。

相关已有 memory：
{existing}

刚完成的对话片段：
{transcript}

严格输出如下 JSON，不要 Markdown，不要额外解释：
{{"updates":[{{"id":"existing-memory-id","content":"User ..."}}],"adds":[{{"content":"User ..."}}]}}
如果没有值得长期保存的新信息，输出：
{{"updates":[],"adds":[]}}
'''


async def run_post_chat_memory_extractor(
    request,
    user,
    model: dict,
    metadata: dict,
    messages: list[dict],
) -> dict[str, int]:
    if metadata.get('task'):
        return {'added': 0, 'updated': 0}

    chat_id = metadata.get('chat_id', '')
    if not chat_id or chat_id.startswith('local:') or chat_id.startswith('channel:'):
        return {'added': 0, 'updated': 0}

    if not getattr(request.app.state.config, 'ENABLE_MEMORIES', False):
        return {'added': 0, 'updated': 0}

    model_id = model.get('id') or (model.get('info') or {}).get('id') or metadata.get('model_id')
    if model_id not in _enabled_model_ids():
        return {'added': 0, 'updated': 0}

    if not messages:
        return {'added': 0, 'updated': 0}

    explicit_memory = _explicit_memory_request(messages)

    models = request.app.state.MODELS or {}
    extractor_model_id = _select_extractor_model_id(model, models)
    if not extractor_model_id:
        if explicit_memory:
            try:
                added = 1 if await _add_memory_content(request, user, explicit_memory) else 0
                return {'added': added, 'updated': 0}
            except Exception as e:
                log.debug(f'Post-chat explicit memory fallback failed: {e}')
        log.debug('Post-chat memory extractor skipped: no available extractor model')
        return {'added': 0, 'updated': 0}

    transcript = _compact_messages(messages)
    if not transcript:
        return {'added': 0, 'updated': 0}

    relevant_memories = await _get_relevant_memories(request, user, messages)
    max_memories = _max_memories_per_turn()
    prompt = _build_extraction_prompt(transcript, relevant_memories, max_memories)

    payload = {
        'model': extractor_model_id,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'max_completion_tokens': 800,
        'metadata': {
            'task': EXTRACTOR_TASK_NAME,
            'chat_id': chat_id,
            'source_model_id': model_id,
        },
    }

    try:
        response = await generate_chat_completion(request, form_data=payload, user=user)
    except Exception as e:
        log.debug(f'Post-chat memory extraction completion failed: {e}')
        if explicit_memory:
            try:
                added = 1 if await _add_memory_content(request, user, explicit_memory) else 0
                return {'added': added, 'updated': 0}
            except Exception as fallback_error:
                log.debug(f'Post-chat explicit memory fallback failed: {fallback_error}')
        return {'added': 0, 'updated': 0}

    parsed = _extract_json_object(_response_content(response))

    if not parsed:
        if explicit_memory:
            parsed = {'updates': [], 'adds': [{'content': explicit_memory}]}
        else:
            log.debug('Post-chat memory extraction returned no parseable JSON')
            return {'added': 0, 'updated': 0}

    if explicit_memory and not (parsed.get('updates') or parsed.get('adds')):
        parsed['adds'] = [{'content': explicit_memory}]

    relevant_ids = {memory['id'] for memory in relevant_memories}
    all_existing = await Memories.get_memories_by_user_id(user.id)
    existing_contents = {_normalize_memory(memory.content) for memory in all_existing}

    updated = 0
    added = 0
    remaining = max_memories

    for item in parsed.get('updates') or []:
        if remaining <= 0:
            break
        memory_id = item.get('id')
        content = (item.get('content') or '').strip()
        if not memory_id or memory_id not in relevant_ids or not content:
            continue
        normalized = _normalize_memory(content)
        if normalized in existing_contents:
            continue
        try:
            memory = await update_memory_by_id(
                memory_id=memory_id,
                request=request,
                form_data=MemoryUpdateModel(content=content),
                user=user,
            )
            if memory:
                existing_contents.add(normalized)
                updated += 1
                remaining -= 1
        except Exception as e:
            log.debug(f'Post-chat memory update failed for {memory_id}: {e}')

    for item in parsed.get('adds') or []:
        if remaining <= 0:
            break
        content = (item.get('content') or '').strip()
        if not content:
            continue
        normalized = _normalize_memory(content)
        if normalized in existing_contents:
            continue
        try:
            if await _add_memory_content(request, user, content, existing_contents):
                added += 1
                remaining -= 1
        except Exception as e:
            log.debug(f'Post-chat memory add failed: {e}')

    if added or updated:
        log.info(
            'Post-chat memory extractor stored memories: added=%s updated=%s chat_id=%s model=%s extractor_model=%s',
            added,
            updated,
            chat_id,
            model_id,
            extractor_model_id,
        )

    return {'added': added, 'updated': updated}
