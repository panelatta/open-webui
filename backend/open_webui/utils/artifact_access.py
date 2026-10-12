"""Authorization for hosted files, Responses state and cached media."""

import hashlib
import json
from pathlib import Path

from fastapi import HTTPException
from open_webui.models.chats import Chats
from open_webui.models.files import Files
from open_webui.models.models import Models
from open_webui.models.security import SecurityArtifacts
from open_webui.utils.access_control import check_model_access
from open_webui.utils.access_control.files import has_access_to_file


def provider_scope(url, key, config=None):
    # Bind to both the provider and its credential/config identity, not its mutable list index.
    config = config or {}
    identity = [
        url.rstrip("/"),
        key or "",
        config.get("auth_type"),
        config.get("headers", {}),
        config.get("azure"),
        config.get("provider"),
    ]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


async def require_file_read(file_id, user):
    if user is None or user.role not in {"user", "admin"}:
        raise HTTPException(403, "File is not accessible.")
    file = await Files.get_file_by_id(file_id)
    if not file or not (
        file.user_id == user.id
        or user.role == "admin"
        or await has_access_to_file(file_id, "read", user)
    ):
        raise HTTPException(403, "File is not accessible.")
    return file


async def require_artifact(kind, external_id, scope, user):
    if not isinstance(external_id, str) or not external_id:
        raise HTTPException(403, "Provider artifact is not accessible.")
    artifact = await SecurityArtifacts.get(kind, scope, external_id)
    if artifact is None:
        raise HTTPException(403, "Provider artifact is not accessible.")
    if artifact.local_file_id:
        await require_file_read(artifact.local_file_id, user)
    elif artifact.user_id != user.id:
        # Only the original live chat ACL can delegate provider state; a copied share snapshot cannot.
        if not artifact.chat_id or not await Chats.get_accessible_chat_by_id(
            artifact.chat_id, user
        ):
            raise HTTPException(403, "Provider artifact is not accessible.")
    if artifact.model_id:
        await check_model_access(
            user, await Models.get_model_by_id(artifact.model_id), False
        )
    return artifact


async def validate_provider_references(payload, scope, user):
    if payload.get("previous_response_id"):
        await require_artifact("response", payload["previous_response_id"], scope, user)
    if payload.get("conversation"):
        value = payload["conversation"]
        await require_artifact(
            "conversation",
            value.get("id") if isinstance(value, dict) else value,
            scope,
            user,
        )

    async def visit(value):
        if isinstance(value, list):
            for item in value:
                await visit(item)
        elif isinstance(value, dict):
            if value.get("file_id"):
                await require_artifact("file", value["file_id"], scope, user)
            if value.get("type") == "item_reference":
                await require_artifact("item", value.get("id"), scope, user)
            for k in ("content", "file"):
                if k in value:
                    await visit(value[k])

    # Hosted tools can reference the same private files through auto containers,
    # existing containers, or vector stores instead of input_file content parts.
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "file_search":
            for store_id in tool.get("vector_store_ids") or []:
                await require_artifact("vector_store", store_id, scope, user)
        if tool.get("type") == "code_interpreter":
            container = tool.get("container")
            if isinstance(container, str):
                await require_artifact("container", container, scope, user)
            elif isinstance(container, dict):
                for file_id in container.get("file_ids") or []:
                    await require_artifact("file", file_id, scope, user)
    await visit(payload.get("input"))
    await visit(payload.get("messages"))


async def record_provider_response(data, scope, user, model_id, chat_id=None):
    if not isinstance(data, dict):
        return
    response = data.get("response") if isinstance(data.get("response"), dict) else data
    response_id = response.get("id")
    if not response_id or not (
        response.get("object") == "response"
        or str(data.get("type", "")).startswith("response.")
    ):
        return
    # Metadata arriving directly at /openai/* is untrusted.
    if chat_id and not await Chats.get_accessible_chat_by_id(
        chat_id, user, permission="write"
    ):
        chat_id = None
    await SecurityArtifacts.record(
        "response", scope, response_id, user.id, chat_id=chat_id, model_id=model_id
    )
    for item in response.get("output") or []:
        if isinstance(item, dict) and item.get("id"):
            await SecurityArtifacts.record(
                "item", scope, item["id"], user.id, chat_id=chat_id, model_id=model_id
            )
    for item in response.get("output") or []:
        if isinstance(item, dict) and item.get("container_id"):
            await SecurityArtifacts.record(
                "container",
                scope,
                item["container_id"],
                user.id,
                chat_id=chat_id,
                model_id=model_id,
            )
    conversation = response.get("conversation")
    if isinstance(conversation, dict) and conversation.get("id"):
        await SecurityArtifacts.record(
            "conversation",
            scope,
            conversation["id"],
            user.id,
            chat_id=chat_id,
            model_id=model_id,
        )


async def owned_response_stream(stream, scope, user, model_id, chat_id=None):
    """Observe complete SSE events without changing any streamed bytes."""
    lines = []
    size = 0
    recorded = set()
    try:
        async for chunk in stream:
            raw = (
                chunk.decode("utf-8", errors="replace")
                if isinstance(chunk, bytes)
                else chunk
            )
            for line in raw.splitlines(keepends=True):
                if not line.strip():
                    if lines:
                        try:
                            data = json.loads("\n".join(lines))
                        except (ValueError, TypeError):
                            data = None
                        if isinstance(data, dict):
                            obj = data.get("response") or {}
                            identifier = (
                                obj.get("id") if isinstance(obj, dict) else None
                            )
                            if identifier and (
                                identifier not in recorded
                                or obj.get("output")
                                or obj.get("conversation")
                            ):
                                await record_provider_response(
                                    data, scope, user, model_id, chat_id
                                )
                                recorded.add(identifier)
                            # Output items can be referenced by native Responses callers.
                            item = data.get("item")
                            if isinstance(item, dict) and item.get("container_id"):
                                await SecurityArtifacts.record(
                                    "container",
                                    scope,
                                    item["container_id"],
                                    user.id,
                                    model_id=model_id,
                                )
                            if isinstance(item, dict) and item.get("id"):
                                await SecurityArtifacts.record(
                                    "item",
                                    scope,
                                    item["id"],
                                    user.id,
                                    model_id=model_id,
                                )
                        lines, size = [], 0
                elif line.startswith("data:") and size < 8 * 1024 * 1024:
                    value = line[5:].strip()
                    lines.append(value)
                    size += len(value)
            yield chunk
    finally:
        if hasattr(stream, "aclose"):
            await stream.aclose()


async def register_cache_file(path, user):
    from open_webui.config import CACHE_DIR

    relative = Path(path).resolve().relative_to(Path(CACHE_DIR).resolve()).as_posix()
    await SecurityArtifacts.record("cache", "local", relative, user.id)


async def require_cache_read(path, user):
    from open_webui.config import CACHE_DIR

    root = Path(CACHE_DIR).resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root) or resolved == root or not resolved.is_file():
        raise HTTPException(404, "File not found")
    relative = resolved.relative_to(root).as_posix()
    artifact = await SecurityArtifacts.get("cache", "local", relative)
    if user.role != "admin" and (artifact is None or artifact.user_id != user.id):
        raise HTTPException(404, "File not found")
    return str(resolved)
