import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from open_webui.models.auths import Auth
from open_webui.models.oauth_sessions import OAuthSession
from open_webui.models.users import User
from open_webui.models import users
from open_webui.utils import oauth_accounts as links
from open_webui.utils.auth import get_password_hash
from open_webui.utils.mfa import MfaConfigForm, is_mfa_required


@pytest.fixture
async def database(monkeypatch):
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as conn:
        for table in [User.__table__, Auth.__table__, OAuthSession.__table__]:
            await conn.run_sync(table.create)
        await conn.execute(text("CREATE UNIQUE INDEX uq_user_google_sub ON user (CAST(json_extract(oauth, '$.google.sub') AS TEXT))"))
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def context(db=None):
        if db is not None:
            yield db
        else:
            async with sessions() as session:
                yield session

    monkeypatch.setattr(links, 'get_async_db', context)
    monkeypatch.setattr(users, 'get_async_db_context', context)
    password = await get_password_hash('correct-password-123')
    async with sessions() as db:
        for uid in ['one', 'two']:
            db.add(User(id=uid, name=uid, email=uid+'@local.example', role='user',
                        oauth={}, created_at=1, updated_at=1, last_active_at=1))
            db.add(Auth(id=uid, email=uid+'@local.example', password=password, active=True,
                        session_stamp='stamp-'+uid))
        await db.commit()
    yield sessions
    await engine.dispose()


@pytest.mark.anyio
async def test_different_email_keeps_local_identity_and_revokes_sessions(database):
    await links.change_google_binding('one', 'stamp-one', identity={'sub': '123', 'email': 'other@gmail.com'})
    async with database() as db:
        user = await db.get(User, 'one')
        auth = await db.get(Auth, 'one')
        assert user.email == auth.email == 'one@local.example'
        assert user.oauth['google'] == {'sub': '123', 'email': 'other@gmail.com'}
        assert auth.session_stamp != 'stamp-one'
    # Stale callbacks cannot replay after the successful binding.
    with pytest.raises(HTTPException):
        await links.change_google_binding('one', 'stamp-one', identity={'sub': '456', 'email': 'new@gmail.com'})


@pytest.mark.anyio
async def test_identity_cannot_move_to_another_user(database):
    await links.change_google_binding('one', 'stamp-one', identity={'sub': '123', 'email': 'google@gmail.com'})
    with pytest.raises(HTTPException) as error:
        await links.change_google_binding('two', 'stamp-two', identity={'sub': '123', 'email': 'google@gmail.com'})
    assert error.value.status_code == 409
    async with database() as db:
        assert (await db.get(User, 'two')).oauth == {}
        assert (await db.get(Auth, 'two')).session_stamp == 'stamp-two'


@pytest.mark.anyio
async def test_unique_index_also_protects_normal_signup(database):
    from sqlalchemy.exc import IntegrityError
    async with database() as db:
        (await db.get(User, 'one')).oauth = {'google': {'sub': '123'}}
        await db.commit()
        (await db.get(User, 'two')).oauth = {'google': {'sub': 123}}
        with pytest.raises(IntegrityError):
            await db.commit()


@pytest.mark.anyio
async def test_unlink_checks_password_and_preserves_email(database):
    await links.change_google_binding('one', 'stamp-one', identity={'sub': '123', 'email': 'other@gmail.com'})
    async with database() as db:
        stamp = (await db.get(Auth, 'one')).session_stamp
    with pytest.raises(HTTPException):
        await links.change_google_binding('one', stamp, password='wrong')
    async with database() as db:
        assert (await db.get(User, 'one')).oauth.get('google')
        assert (await db.get(Auth, 'one')).session_stamp == stamp
    await links.change_google_binding('one', stamp, password='correct-password-123')
    async with database() as db:
        assert (await db.get(User, 'one')).oauth == {}
        assert (await db.get(User, 'one')).email == 'one@local.example'
        assert (await db.get(Auth, 'one')).session_stamp != stamp


def request(**claims):
    return SimpleNamespace(state=SimpleNamespace(auth_type='jwt', claims={
        'typ': 'session', 'auth_time': int(time.time()), **claims
    }))


@pytest.mark.parametrize('claims', [
    {'auth_time': 0}, {'auth_time': int(time.time())+1000},
    {'typ': 'automation'}, {'typ': 'subagent'}
])
def test_only_recent_human_sessions_can_manage(claims):
    with pytest.raises(HTTPException):
        links.require_recent_human(request(**claims), SimpleNamespace(role='user'))


def test_pending_user_cannot_link():
    with pytest.raises(HTTPException):
        links.require_recent_human(request(), SimpleNamespace(role='pending'))


def test_google_bypass_does_not_disable_password_mfa():
    config = MfaConfigForm(ENABLE_MFA=True, MFA_ALLOW_OAUTH_BYPASS=True)
    assert not is_mfa_required('oauth', config)
    assert is_mfa_required('password', config)
    assert is_mfa_required('trusted_header', config)


@pytest.mark.anyio
async def test_cross_origin_start_rejected(monkeypatch):
    monkeypatch.setattr(links.Config, 'get', AsyncMock(return_value='https://webui.example'))
    with pytest.raises(HTTPException):
        await links.require_same_origin(SimpleNamespace(headers={'origin':'https://evil.example'}))


@pytest.mark.anyio
@pytest.mark.parametrize('intent', [None, {'state':'wrong','expires_at':int(time.time())+300},
                                      {'state':'owui-link-test','expires_at':0}])
async def test_missing_or_expired_intent_never_creates_or_links(monkeypatch, intent):
    change = AsyncMock()
    monkeypatch.setattr(links, 'change_google_binding', change)
    req = SimpleNamespace(query_params={'state':'owui-link-test'}, session={links.LINK_SESSION:intent})
    response = await links.finish_link(req)
    assert response.headers['location'].endswith('result=failed')
    change.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize('mutation', ['wrong_user','wrong_stamp','unverified_email','missing_sub','denied_domain','valid'])
async def test_callback_binding_checks(monkeypatch, mutation):
    from open_webui.utils import oauth
    claims = {'jti':'jwt-one','session_stamp':'stamp-one','auth_time':int(time.time()),'typ':'session'}
    info = {'sub':'123','email':'other@gmail.com','email_verified':True}
    intent = {'state':'owui-link-test','user_id':'one','session_stamp':'stamp-one',
              'jti':'jwt-one','expires_at':int(time.time())+300}
    if mutation == 'wrong_user': intent['user_id'] = 'two'
    if mutation == 'wrong_stamp': intent['session_stamp'] = 'old'
    if mutation == 'unverified_email': info['email_verified'] = False
    if mutation == 'missing_sub': info.pop('sub')
    client = SimpleNamespace(authorize_access_token=AsyncMock(return_value={'userinfo':info}))
    manager = SimpleNamespace(get_client=lambda _: client)
    req = SimpleNamespace(query_params={'state':'owui-link-test'}, session={links.LINK_SESSION:intent},
        state=SimpleNamespace(auth_type='jwt',claims=claims),
        app=SimpleNamespace(state=SimpleNamespace(oauth_manager=manager)))
    monkeypatch.setattr(links, 'get_current_user', AsyncMock(return_value=SimpleNamespace(id='one',role='user')))
    monkeypatch.setattr(links, 'get_human_user', AsyncMock())
    monkeypatch.setattr(oauth, 'get_oauth_runtime_config', AsyncMock(return_value=SimpleNamespace(
        ENABLE_OAUTH=True, OAUTH_ALLOWED_DOMAINS=['example.com'] if mutation=='denied_domain' else ['*'])))
    change = AsyncMock()
    monkeypatch.setattr(links, 'change_google_binding', change)
    monkeypatch.setattr(links, 'disconnect_after_change', AsyncMock())
    response = await links.finish_link(req)
    assert links.LINK_SESSION not in req.session
    if mutation == 'valid':
        assert response.headers['location'].endswith('result=linked')
        change.assert_awaited_once_with('one', 'stamp-one', identity={'sub':'123','email':'other@gmail.com'})
    else:
        assert response.headers['location'].endswith('result=failed')
        change.assert_not_called()
