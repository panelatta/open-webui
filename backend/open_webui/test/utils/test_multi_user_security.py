import json
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from open_webui.models import security
from open_webui.models.security import SecurityArtifacts, SessionRevocations
from open_webui.routers import openai
from open_webui.utils import artifact_access, auth

OWNER = SimpleNamespace(id="owner", role="user")
OTHER = SimpleNamespace(id="other", role="user")


@pytest.fixture
async def registry(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(security.SecurityArtifact.__table__.create)
        await connection.run_sync(security.RevokedSession.__table__.create)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def context(db=None):
        async with sessions() as session:
            yield session

    monkeypatch.setattr(security, "get_async_db_context", context)
    monkeypatch.setattr(artifact_access, "check_model_access", AsyncMock())
    monkeypatch.setattr(
        artifact_access.Models, "get_model_by_id", AsyncMock(return_value=None)
    )
    yield
    await engine.dispose()


@pytest.mark.anyio
async def test_foreign_hosted_file_rejected_before_upload(monkeypatch):
    file = SimpleNamespace(id="file-a", user_id="owner")
    monkeypatch.setattr(
        artifact_access.Files, "get_file_by_id", AsyncMock(return_value=file)
    )
    monkeypatch.setattr(
        artifact_access, "has_access_to_file", AsyncMock(return_value=False)
    )
    upload = Mock()
    monkeypatch.setattr(openai, "upload_local_file_to_openai", upload)
    for metadata, payload_files in [
        (None, [{"id": file.id}]),
        ({"files": [{"id": file.id}]}, None),
    ]:
        with pytest.raises(HTTPException) as error:
            await openai.inject_openai_files_into_messages(
                None,
                {
                    "messages": [{"role": "user", "content": "read it"}],
                    "files": payload_files,
                },
                metadata,
                OTHER,
                idx=0,
            )
        assert error.value.status_code == 403
    upload.assert_not_called()


@pytest.mark.anyio
async def test_hosted_file_owner_and_shared_read_with_revocation(monkeypatch, registry):
    file = SimpleNamespace(id="file-a", user_id="owner")
    monkeypatch.setattr(
        artifact_access.Files, "get_file_by_id", AsyncMock(return_value=file)
    )
    grant = AsyncMock(return_value=True)
    monkeypatch.setattr(artifact_access, "has_access_to_file", grant)
    monkeypatch.setattr(
        openai,
        "get_openai_connection",
        AsyncMock(return_value=("https://model.example/v1", "key", {})),
    )
    upload = Mock(return_value="upstream-file-a")
    monkeypatch.setattr(openai, "upload_local_file_to_openai", upload)
    monkeypatch.setattr(openai.Files, "update_file_data_by_id", AsyncMock())
    scope = artifact_access.provider_scope("https://model.example/v1", "key", {})
    assert (
        await openai.ensure_openai_file_id(None, file.id, user=OWNER)
        == "upstream-file-a"
    )
    assert (
        await openai.ensure_openai_file_id(None, file.id, user=OTHER)
        == "upstream-file-a"
    )
    row = await SecurityArtifacts.get("file", scope, "upstream-file-a")
    assert row.user_id == OWNER.id and row.local_file_id == file.id
    await artifact_access.require_artifact("file", "upstream-file-a", scope, OTHER)
    grant.return_value = False
    with pytest.raises(HTTPException):
        await artifact_access.require_artifact("file", "upstream-file-a", scope, OTHER)
    with pytest.raises(HTTPException):
        await openai.ensure_openai_file_id(None, file.id, user=OTHER)
    assert upload.call_count == 2


@pytest.mark.anyio
@pytest.mark.parametrize("role", ["pending", "unknown"])
async def test_unapproved_file_access(role):
    with pytest.raises(HTTPException):
        await artifact_access.require_file_read(
            "file-a", SimpleNamespace(id="owner", role=role)
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "payload",
    [
        {"previous_response_id": "resp-a"},
        {"conversation": "conv-a"},
        {"conversation": {"id": "conv-a"}},
        {"input": [{"type": "item_reference", "id": "item-a"}]},
        {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_file", "file_id": "file-a"}],
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "file", "file": {"file_id": "file-a"}}],
                }
            ]
        },
    ],
)
async def test_unknown_native_references_fail_closed(registry, payload):
    with pytest.raises(HTTPException) as error:
        await artifact_access.validate_provider_references(payload, "scope", OTHER)
    assert error.value.status_code == 403


@pytest.mark.anyio
async def test_response_owner_provider_and_shared_chat_acl(registry, monkeypatch):
    access = AsyncMock(return_value=None)
    monkeypatch.setattr(artifact_access.Chats, "get_accessible_chat_by_id", access)
    await SecurityArtifacts.record(
        "response", "scope-a", "resp-a", OWNER.id, model_id="model-a", chat_id="chat-a"
    )
    await artifact_access.require_artifact("response", "resp-a", "scope-a", OWNER)
    for scope, user in [("scope-a", OTHER), ("scope-b", OWNER)]:
        with pytest.raises(HTTPException):
            await artifact_access.require_artifact("response", "resp-a", scope, user)
    access.return_value = SimpleNamespace(id="chat-a")
    await artifact_access.require_artifact("response", "resp-a", "scope-a", OTHER)
    access.return_value = None
    with pytest.raises(HTTPException):
        await artifact_access.require_artifact("response", "resp-a", "scope-a", OTHER)


@pytest.mark.anyio
async def test_artifact_cannot_be_reassigned(registry):
    await SecurityArtifacts.record("response", "scope", "id", OWNER.id)
    await SecurityArtifacts.record("response", "scope", "id", OWNER.id)
    with pytest.raises(HTTPException):
        await SecurityArtifacts.record("response", "scope", "id", OTHER.id)
    assert (await SecurityArtifacts.get("response", "scope", "id")).user_id == OWNER.id


def test_provider_binding_changes_with_credential_or_destination():
    scope = artifact_access.provider_scope("https://a/v1", "secret", {})
    assert scope == artifact_access.provider_scope("https://a/v1/", "secret", {})
    assert scope != artifact_access.provider_scope("https://b/v1", "secret", {})
    assert scope != artifact_access.provider_scope("https://a/v1", "different", {})
    assert scope != artifact_access.provider_scope(
        "https://a/v1", "secret", {"headers": {"x-tenant": "b"}}
    )


@pytest.mark.anyio
async def test_stream_ownership_persisted_before_id_is_yielded(registry):
    event = {
        "type": "response.created",
        "response": {"id": "resp-a", "object": "response"},
    }
    chunks = [
        b"event: response.created\n",
        ("data: " + json.dumps(event) + "\n").encode(),
        b"\n",
        b'data: {"type":"response.output_item.added","item":{"id":"item-a"}}\n',
        b"\n",
        b"data: [DONE]\n",
        b"\n",
    ]

    async def source():
        for chunk in chunks:
            yield chunk

    got = []
    async for chunk in artifact_access.owned_response_stream(
        source(), "scope", OWNER, "model"
    ):
        got.append(chunk)
        if len(got) == 3:
            assert await SecurityArtifacts.get("response", "scope", "resp-a")
    assert got == chunks
    assert (await SecurityArtifacts.get("item", "scope", "item-a")).user_id == OWNER.id


@pytest.mark.anyio
async def test_persistent_per_session_revocation_and_expiry(registry):
    assert not await SessionRevocations.is_revoked("session-one")
    await SessionRevocations.revoke("session-one", int(time.time()) + 600)
    await SessionRevocations.revoke("session-one", int(time.time()) + 600)
    assert await SessionRevocations.is_revoked("session-one")
    assert not await SessionRevocations.is_revoked("session-two")
    await SessionRevocations.revoke("expired", int(time.time()) - 1)
    assert not await SessionRevocations.is_revoked("expired")


@pytest.mark.anyio
async def test_signout_persists_without_redis(registry, monkeypatch):
    from open_webui.socket import main as socket

    decoded = {"id": OWNER.id, "jti": "session-one", "exp": int(time.time()) + 60}
    monkeypatch.setattr(auth, "decode_token", lambda _: decoded)
    monkeypatch.setattr(auth, "is_valid_token", AsyncMock(return_value=True))
    monkeypatch.setattr(socket, "disconnect_user_sessions", AsyncMock())
    await auth.invalidate_token(
        SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(redis=None))), "token"
    )
    assert await SessionRevocations.is_revoked("session-one")
    socket.disconnect_user_sessions.assert_awaited_once_with(OWNER.id)


@pytest.mark.anyio
async def test_cache_owner_isolation_and_symlink_escape(
    registry, monkeypatch, tmp_path
):
    from open_webui import config

    root = tmp_path / "cache"
    root.mkdir()
    monkeypatch.setattr(config, "CACHE_DIR", root)
    file = root / "audio.mp3"
    file.write_bytes(b"private")
    await artifact_access.register_cache_file(file, OWNER)
    assert await artifact_access.require_cache_read("audio.mp3", OWNER) == str(file)
    with pytest.raises(HTTPException):
        await artifact_access.require_cache_read("audio.mp3", OTHER)
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    (root / "link.txt").symlink_to(outside)
    for path in ["../outside.txt", "link.txt"]:
        with pytest.raises(HTTPException):
            await artifact_access.require_cache_read(
                path, SimpleNamespace(id="admin", role="admin")
            )


@pytest.mark.parametrize("password", ["", " ", "\n\t"])
def test_empty_password_rejected_even_if_complexity_disabled(monkeypatch, password):
    monkeypatch.setattr(auth, "ENABLE_PASSWORD_VALIDATION", False)
    with pytest.raises(ValueError):
        auth.validate_password(password)
