from io import BytesIO
from types import SimpleNamespace

import pytest
from starlette.datastructures import Headers, UploadFile

from open_webui.routers import files as files_router
from open_webui.routers import memories
from open_webui.utils.embedding_policy import (
    EMBEDDING_DISABLED_MESSAGE,
    apply_memory_only_embedding_policy,
)


@pytest.mark.anyio
async def test_memory_only_embedding_policy_keeps_memory_function_separate():
    calls = []

    async def memory_embedding(content, *, prefix=None, user=None):
        calls.append((content, prefix, user.id))
        return [0.25, 0.75]

    state = SimpleNamespace()
    apply_memory_only_embedding_policy(state, memory_embedding)

    with pytest.raises(RuntimeError, match='Embedding is disabled outside memory routes') as exc_info:
        await state.EMBEDDING_FUNCTION('not memory')

    request = SimpleNamespace(app=SimpleNamespace(state=state))
    user = SimpleNamespace(id='user-1')
    result = await memories.embed_memory_content(request, 'remember this', user)

    assert result == [0.25, 0.75]
    assert calls == [('remember this', None, 'user-1')]
    assert str(exc_info.value) == EMBEDDING_DISABLED_MESSAGE


@pytest.mark.anyio
async def test_file_upload_process_true_is_forced_to_storage_only(monkeypatch):
    captured = {}

    async def fake_config_get(key, default=None):
        return default

    def fake_storage_upload(file_obj, filename, tags):
        contents = file_obj.read()
        captured['storage_filename'] = filename
        captured['storage_tags'] = tags
        return contents, f'/tmp/{filename}'

    async def fake_insert_new_file(user_id, form_data, db=None):
        captured['user_id'] = user_id
        captured['form_data'] = form_data
        return SimpleNamespace(
            id=form_data.id,
            filename=form_data.filename,
            path=form_data.path,
            data=form_data.data,
            meta=form_data.meta,
        )

    async def fail_if_processed(*args, **kwargs):
        pytest.fail('local file processing must remain disabled')

    class FailBackgroundTasks:
        def add_task(self, *args, **kwargs):
            pytest.fail('local file processing must not be scheduled')

    monkeypatch.setattr(files_router.Config, 'get', staticmethod(fake_config_get))
    monkeypatch.setattr(files_router.Storage, 'upload_file', fake_storage_upload)
    monkeypatch.setattr(files_router.Files, 'insert_new_file', fake_insert_new_file)
    monkeypatch.setattr(files_router, 'process_uploaded_file', fail_if_processed)

    upload = UploadFile(
        BytesIO(b'chat attachment'),
        filename='chat.txt',
        headers=Headers({'content-type': 'text/plain'}),
    )
    user = SimpleNamespace(id='user-1', email='user@example.com', name='User')

    result = await files_router.upload_file_handler(
        SimpleNamespace(),
        file=upload,
        process=True,
        process_in_background=True,
        user=user,
        background_tasks=FailBackgroundTasks(),
        db=None,
    )

    assert result.id == captured['form_data'].id
    assert captured['user_id'] == user.id
    assert captured['form_data'].data == {'content': 'chat attachment'}
    assert 'collection_name' not in captured['form_data'].meta
