from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from open_webui.models.chats import Chat, Chats
from open_webui.routers import chats as routes


@pytest.mark.anyio
async def test_metadata_list_pagination_scope_and_literal_search(monkeypatch):
    monkeypatch.setattr("open_webui.internal.db.DATABASE_ENABLE_SESSION_SHARING", True)
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as conn:
        await conn.run_sync(Chat.__table__.create)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with sessions() as db:
        db.add_all([
            Chat(id=f'c{i:03}', user_id='owner', title=f'Topic {i}', chat={'secret': 'body'},
                 created_at=1, updated_at=10, archived=False, pinned=i == 1, meta={})
            for i in range(55)
        ] + [
            Chat(id='archived', user_id='owner', title='100%_done', created_at=1, updated_at=2,
                 archived=True, folder_id='folder', meta={}),
            Chat(id='other', user_id='other', title='100%_done', created_at=1, updated_at=2, meta={}),
            Chat(id='internal', user_id='owner', title='100%_done', created_at=1, updated_at=2,
                 meta={'internal': True}),
        ])
        await db.commit()
        first = await Chats.get_managed_chat_list('owner', archived=False, db=db)
        second = await Chats.get_managed_chat_list('owner', archived=False, skip=50, db=db)
        assert first.total == second.total == 55
        assert len(first.items) == 50 and len(second.items) == 5
        assert not {c.id for c in first.items} & {c.id for c in second.items}
        assert first.items[1].pinned
        assert 'chat' not in first.items[0].model_dump()
        literal = await Chats.get_managed_chat_list('owner', query='%_', db=db)
        assert [c.id for c in literal.items] == ['archived']
        assert literal.items[0].archived
        folder = await Chats.get_managed_chat_list('owner', folder_id='folder', db=db)
        assert folder.total == 1
        root = await Chats.get_managed_chat_list('owner', folder_id='', db=db)
        assert root.total == 55
        all_chats = await Chats.get_managed_chat_list('owner', db=db)
        assert all_chats.total == 56
    await engine.dispose()


@pytest.fixture
def bulk(monkeypatch):
    owner = SimpleNamespace(id='owner', role='admin')
    db = SimpleNamespace(rollback=AsyncMock())
    lookup = AsyncMock(return_value=SimpleNamespace(archived=False, meta={}))
    monkeypatch.setattr(routes.Chats, 'get_chat_by_id_and_user_id', lookup)
    delete = AsyncMock(return_value=True)
    move = AsyncMock(return_value=True)
    archive = AsyncMock(return_value=True)
    monkeypatch.setattr(routes, 'delete_chat_by_id', delete)
    monkeypatch.setattr(routes, 'update_chat_folder_id_by_id', move)
    monkeypatch.setattr(routes, 'archive_chat_by_id', archive)
    return SimpleNamespace(user=owner, db=db, lookup=lookup, delete=delete, move=move, archive=archive)


@pytest.mark.anyio
async def test_bulk_ownership_admin_and_partial_failures(bulk):
    bulk.lookup.side_effect = [None, SimpleNamespace(archived=False, meta={}), SimpleNamespace(meta={'internal': True})]
    result = await routes.bulk_manage_chats(None, routes.BulkChatForm(
        ids=['foreign', 'owned', 'internal', 'owned'], action='delete'
    ), user=bulk.user, db=bulk.db)
    assert [(r.id, r.success) for r in result] == [('foreign', False), ('owned', True), ('internal', False)]
    bulk.delete.assert_awaited_once_with(None, 'owned', user=bulk.user, db=bulk.db)
    assert all(call.args[1] == 'owner' for call in bulk.lookup.call_args_list)


@pytest.mark.anyio
async def test_archive_and_unarchive_are_desired_states(bulk):
    bulk.lookup.side_effect = [
        SimpleNamespace(archived=True, meta={}), SimpleNamespace(archived=False, meta={})
    ]
    results = await routes.bulk_manage_chats(None, routes.BulkChatForm(ids=['already', 'new'], action='archive'), user=bulk.user, db=bulk.db)
    assert all(r.success for r in results)
    bulk.archive.assert_awaited_once_with(None, 'new', user=bulk.user, db=bulk.db)
    bulk.lookup.side_effect = None
    bulk.lookup.return_value = SimpleNamespace(archived=False, meta={})
    await routes.bulk_manage_chats(None, routes.BulkChatForm(ids=['new'], action='unarchive'), user=bulk.user, db=bulk.db)
    assert bulk.archive.await_count == 1


@pytest.mark.anyio
async def test_permission_denied_before_mutation(monkeypatch, bulk):
    bulk.user.role = 'user'
    monkeypatch.setattr(routes.Config, 'get', AsyncMock(return_value={}))
    monkeypatch.setattr(routes, 'has_permission', AsyncMock(return_value=False))
    with pytest.raises(HTTPException) as exc:
        await routes.bulk_manage_chats(None, routes.BulkChatForm(ids=['x'], action='delete'), user=bulk.user, db=bulk.db)
    assert exc.value.status_code == 403
    bulk.lookup.assert_not_awaited()
    bulk.delete.assert_not_awaited()


@pytest.mark.anyio
async def test_move_checks_folder_before_mutation(monkeypatch, bulk):
    from open_webui.routers import folders
    monkeypatch.setattr(folders, 'check_folders_permission', AsyncMock())
    monkeypatch.setattr(routes, 'has_folder_write_access', AsyncMock(return_value=False))
    with pytest.raises(HTTPException) as exc:
        await routes.bulk_manage_chats(None, routes.BulkChatForm(ids=['x'], action='move', folder_id='foreign'), user=bulk.user, db=bulk.db)
    assert exc.value.status_code == 404
    bulk.lookup.assert_not_awaited()
    await routes.bulk_manage_chats(None, routes.BulkChatForm(ids=['x'], action='move'), user=bulk.user, db=bulk.db)
    assert bulk.move.call_args.args[2].folder_id is None


@pytest.mark.anyio
async def test_failed_item_does_not_abort_remaining_items(bulk):
    bulk.delete.side_effect = [RuntimeError('private database details'), True, False]
    result = await routes.bulk_manage_chats(None, routes.BulkChatForm(ids=['a', 'b', 'c'], action='delete'), user=bulk.user, db=bulk.db)
    assert [r.success for r in result] == [False, True, False]
    assert 'private database details' not in result[0].error
    assert bulk.db.rollback.await_count == 2


@pytest.mark.parametrize('ids', [[], ['x'] * 51])
def test_batch_size_is_bounded(ids):
    with pytest.raises(ValidationError):
        routes.BulkChatForm(ids=ids, action='delete')


def test_http_route_order_and_validation(monkeypatch):
    app = FastAPI()
    app.include_router(routes.router, prefix='/chats')
    app.dependency_overrides[routes.get_verified_user] = lambda: SimpleNamespace(id='owner', role='admin')
    app.dependency_overrides[routes.get_async_session] = lambda: None
    mock = AsyncMock(return_value={'items': [], 'total': 0})
    monkeypatch.setattr(routes.Chats, 'get_managed_chat_list', mock)
    client = TestClient(app)
    assert client.get('/chats/manage').status_code == 200
    assert client.get('/chats/manage?page=0').status_code == 422
    assert client.post('/chats/bulk', json={'ids': [], 'action': 'delete'}).status_code == 422


@pytest.mark.anyio
@pytest.mark.parametrize(
    'original_folder,target_folder,archived,pinned',
    [
        (None, 'target', False, False),
        ('source', 'target', False, True),
        ('source', None, False, False),
        ('target', 'target', False, False),
        (None, 'target', True, True),
    ],
)
async def test_moving_chat_preserves_content_timestamp(
    monkeypatch, original_folder, target_folder, archived, pinned
):
    monkeypatch.setattr('open_webui.internal.db.DATABASE_ENABLE_SESSION_SHARING', True)
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as conn:
        await conn.run_sync(Chat.__table__.create)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    content = {'title': 'Historical chat', 'messages': [{'role': 'user', 'content': 'Keep me'}]}
    async with sessions() as db:
        row = Chat(
            id='historical', user_id='owner', title='Historical chat', chat=content,
            created_at=100, updated_at=200, last_read_at=150, meta={},
            folder_id=original_folder, archived=archived, pinned=pinned,
        )
        db.add(row)
        await db.commit()
        result = await Chats.update_chat_folder_id_by_id_and_user_id(
            'historical', 'owner', target_folder, db=db
        )
        assert result is not None
        assert result.updated_at == 200
        assert result.created_at == 100
        assert result.chat == content
        assert result.folder_id == target_folder
        assert result.pinned is False
        assert result.archived is (False if target_folder is not None else archived)
        await db.refresh(row)
        assert row.updated_at == 200
    await engine.dispose()
