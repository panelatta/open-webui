"""Sliding session refresh: lineage creation, rotation, replay detection and revocation."""

import asyncio
import itertools
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Response
from sqlalchemy import event, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from starlette.requests import Request

from open_webui.internal import db as db_module
from open_webui.models import security
from open_webui.models.auths import Auth, MfaChallenge
from open_webui.models.oauth_sessions import OAuthSession
from open_webui.models.security import (
    RevokedSession,
    SessionChains,
    SessionLineage,
    SessionRevocations,
    digest,
)
from open_webui.models.users import User
from open_webui.utils import auth as auth_utils
from open_webui.utils import mfa as mfa_utils
from open_webui.utils import session_refresh as sr

ORIGIN = 'https://chat.example'
DAY = 86400
MONTH = 30 * DAY


class Clock:
    def __init__(self):
        self.now = int(time.time())

    def time(self):
        return self.now


class FakeConfig:
    values = {'auth.jwt_expiry': '24h', 'webui.url': ORIGIN, 'user.permissions': {}}

    @staticmethod
    async def get(key, default=None):
        return FakeConfig.values.get(key, default)


@pytest.fixture
async def env(monkeypatch, tmp_path):
    path = tmp_path / 'webui.db'
    engines = [
        create_async_engine(f'sqlite+aiosqlite:///{path}', connect_args={'timeout': 15}) for _ in range(2)
    ]
    for engine in engines:

        @event.listens_for(engine.sync_engine, 'connect')
        def _pragmas(connection, _record):
            cursor = connection.cursor()
            cursor.execute('PRAGMA journal_mode=WAL')
            cursor.execute('PRAGMA busy_timeout=15000')
            cursor.close()

    async with engines[0].begin() as conn:
        for table in [User.__table__, Auth.__table__, OAuthSession.__table__, RevokedSession.__table__, SessionLineage.__table__]:
            await conn.run_sync(table.create)
    makers = [async_sessionmaker(engine, expire_on_commit=False) for engine in engines]
    rotation = itertools.cycle(makers)
    # Alternate between two independent connection pools on the same WAL database file.
    monkeypatch.setattr(db_module, 'AsyncSessionLocal', lambda: next(rotation)())

    clock = Clock()
    monkeypatch.setattr(sr, 'time', clock)
    monkeypatch.setattr(security, 'time', clock)
    monkeypatch.setattr(sr, 'ENABLE_SESSION_REFRESH', True)
    monkeypatch.setattr(sr, 'MAX_LIFETIME_SECONDS', MONTH)
    monkeypatch.setattr(sr, 'Config', FakeConfig)
    FakeConfig.values['auth.jwt_expiry'] = '24h'
    mfa_config = SimpleNamespace(value=mfa_utils.MfaConfigForm())
    monkeypatch.setattr(mfa_utils, 'get_mfa_config', AsyncMock(side_effect=lambda: mfa_config.value))

    async with makers[0]() as session:
        for uid, role in [('one', 'user'), ('two', 'user'), ('pending', 'pending')]:
            session.add(User(id=uid, name=uid, email=f'{uid}@local.example', role=role, oauth={}, created_at=1, updated_at=1, last_active_at=1))
            session.add(Auth(id=uid, email=f'{uid}@local.example', password='x', active=True, session_stamp=f'stamp-{uid}'))
        await session.commit()
    yield SimpleNamespace(clock=clock, makers=makers, mfa=mfa_config)
    for engine in engines:
        await engine.dispose()


def login_token(uid='one', *, now=None, lifetime=DAY, auth_method='password', mfa_verified=False, **extra):
    now = now or int(time.time())
    claims = {
        'id': uid,
        'typ': 'session',
        'auth_method': auth_method,
        'auth_time': now,
        'session_stamp': f'stamp-{uid}',
        'mfa_verified': mfa_verified,
        'exp': now + lifetime,
        **extra,
    }
    return auth_utils.create_token({k: v for k, v in claims.items() if v is not None})


async def open_login(uid='one', **kwargs):
    """A first-party login: access token plus refresh cookie value."""
    token = login_token(uid, **kwargs)
    response = Response()
    assert await sr.start_session_chain(response, token, user=SimpleNamespace(id=uid, role='admin' if uid == 'one' else 'user'))
    return token, cookie_value(response, sr.REFRESH_COOKIE)


def set_cookies(response):
    return [value.decode() for key, value in response.raw_headers if key == b'set-cookie']


def cookie_value(response, name):
    for header in set_cookies(response):
        if header.startswith(f'{name}='):
            return header.split(';', 1)[0].split('=', 1)[1].strip('"')
    return None


def make_request(token=None, refresh=None, *, origin=ORIGIN, extra_headers=(), cookies=None, path='/api/v1/auths/refresh'):
    headers = []
    if token is not None:
        headers.append((b'authorization', f'Bearer {token}'.encode()))
    if origin is not None:
        headers.append((b'origin', origin.encode()))
    jar = dict(cookies or {})
    if refresh is not None:
        jar[sr.REFRESH_COOKIE] = refresh
    if jar:
        headers.append((b'cookie', '; '.join(f'{k}={v}' for k, v in jar.items()).encode()))
    headers.extend((k.encode(), v.encode()) for k, v in extra_headers)
    scope = {
        'type': 'http',
        'method': 'POST',
        'path': path,
        'root_path': '',
        'scheme': 'https',
        'server': ('chat.example', 443),
        'query_string': b'',
        'headers': headers,
        'app': SimpleNamespace(state=SimpleNamespace(redis=None)),
    }
    return Request(scope)


async def refresh(token, refresh_cookie, **kwargs):
    response = Response()
    result = await sr.refresh_session(make_request(token, refresh_cookie, **kwargs), response)
    return result, response


async def status_of(token, refresh_cookie, **kwargs):
    try:
        await refresh(token, refresh_cookie, **kwargs)
    except HTTPException as error:
        return error.status_code
    return 200


async def valid(token):
    return await auth_utils.is_valid_token(auth_utils.decode_token(token))


async def lineage(token):
    claims = auth_utils.decode_token(token) or auth_utils.decode_token_for_revocation(token)
    return await SessionChains.get(claims.get('sid') or claims['jti'])


# --- policy -------------------------------------------------------------------------


def test_max_lifetime_parsing_is_strict_and_bounded():
    assert sr.parse_max_lifetime('30d') == MONTH
    assert sr.parse_max_lifetime('1h') == 3600
    assert sr.parse_max_lifetime('52w') == 52 * 7 * DAY
    for value in ['', '0', '-1', '59m', '366d', '30d1h', ' 3 0d', '1.5d', '30x', '1y', None]:
        assert sr.parse_max_lifetime(value) is None, value


@pytest.mark.anyio
async def test_policy_requires_finite_bounded_access_lifetime(env, monkeypatch):
    assert await sr.get_refresh_policy() == (DAY, MONTH)
    for value in ['-1', '0', '4m', '31d', 'nonsense']:
        FakeConfig.values['auth.jwt_expiry'] = value
        assert await sr.get_refresh_policy() is None, value
    FakeConfig.values['auth.jwt_expiry'] = '5m'
    assert await sr.get_refresh_policy() == (300, MONTH)
    monkeypatch.setattr(sr, 'ENABLE_SESSION_REFRESH', False)
    assert await sr.get_refresh_policy() is None


# --- lineage creation ---------------------------------------------------------------


@pytest.mark.anyio
async def test_browser_login_opens_lineage_with_strict_cookie(env, monkeypatch):
    from open_webui import events
    from open_webui.utils import access_control

    monkeypatch.setattr(events, 'publish_event', AsyncMock())
    monkeypatch.setattr(access_control, 'get_permissions', AsyncMock(return_value={}))
    monkeypatch.setattr(auth_utils, 'Config', FakeConfig)
    user = SimpleNamespace(id='one', role='user', email='one@local.example', name='one')
    auth = await mfa_utils.get_auth('one')
    request = make_request()

    response = Response()
    result = await auth_utils.create_session_response(request, user, response=response, set_cookie=True, auth=auth)
    chain = await lineage(result['token'])
    assert chain and chain.user_id == 'one' and chain.generation == 0 and chain.revoked_at is None
    assert chain.absolute_expires_at == chain.auth_time + MONTH
    header = next(h for h in set_cookies(response) if h.startswith(sr.REFRESH_COOKIE))
    lowered = header.lower()
    for attribute in ['httponly', 'samesite=strict', 'path=/api/v1/auths', 'max-age=']:
        assert attribute in lowered
    sid, secret = cookie_value(response, sr.REFRESH_COOKIE).split('.', 1)
    assert sid == auth_utils.decode_token(result['token'])['jti']
    assert digest(secret) == chain.current_secret and secret not in str(chain)

    # API-style issuance (token exchange, no cookie) and non-renewable browser flows never open one.
    for kwargs in [{'set_cookie': False}, {'set_cookie': True, 'renewable': False}]:
        response = Response()
        result = await auth_utils.create_session_response(request, user, response=response, auth=auth, **kwargs)
        assert await lineage(result['token']) is None
        assert cookie_value(response, sr.REFRESH_COOKIE) is None


@pytest.mark.anyio
async def test_only_complete_human_tokens_open_lineages(env, monkeypatch):
    human = SimpleNamespace(id='one', role='user')
    for token, user in [
        (login_token(auth_time=None), human),  # no auth_time (e.g. /auths/add tokens)
        (login_token(auth_method=None), human),
        (login_token(typ='automation'), human),
        (login_token(sid='other'), human),  # a renewed token can never start a new lineage
        (login_token('pending'), SimpleNamespace(id='pending', role='pending')),
        (login_token(), SimpleNamespace(id='two', role='user')),
    ]:
        response = Response()
        assert not await sr.start_session_chain(response, token, user=user)
        assert cookie_value(response, sr.REFRESH_COOKIE) is None
    FakeConfig.values['auth.jwt_expiry'] = '-1'
    assert not await sr.start_session_chain(Response(), login_token(), user=human)


def test_mfa_challenges_are_not_renewable_unless_marked():
    challenge = MfaChallenge(token_hash='h', type='verify', expires_at=1, auth_method='oauth', auth_time=1)
    assert challenge.renewable is False


@pytest.mark.anyio
async def test_signin_marks_only_cookie_logins_renewable(env, monkeypatch):
    start = AsyncMock(return_value={'next_step': 'verify'})
    monkeypatch.setattr(mfa_utils, 'start_mfa_login', start)
    env.mfa.value = mfa_utils.MfaConfigForm(ENABLE_MFA=True)
    user = SimpleNamespace(id='one', role='user')
    await auth_utils.create_signin_response(make_request(), user, source='oauth')  # token exchange
    await auth_utils.create_signin_response(make_request(), user, response=Response(), set_cookie=True)
    await auth_utils.create_signin_response(make_request(), user, source='oauth', renewable=True)  # callback
    assert [call.kwargs['renewable'] for call in start.await_args_list] == [False, True, True]


# --- refresh -------------------------------------------------------------------------


@pytest.mark.anyio
async def test_refresh_rotates_and_inherits_frozen_claims(env):
    t0, c0 = await open_login(mfa_verified=False)
    first = auth_utils.decode_token(t0)
    env.clock.now += DAY // 2
    result, response = await refresh(t0, c0)
    assert result['refreshed'] is True and result['server_time'] == env.clock.now
    renewed = auth_utils.decode_token(result['token'])
    assert renewed['sid'] == first['jti'] != renewed['jti']
    for key in ['id', 'typ', 'auth_method', 'auth_time', 'session_stamp', 'mfa_verified']:
        assert renewed[key] == first[key], key
    assert renewed['exp'] == result['expires_at'] == env.clock.now + DAY
    assert cookie_value(response, 'token') == result['token']
    token_header = next(h for h in set_cookies(response) if h.startswith('token='))
    assert 'httponly' in token_header.lower()
    c1 = cookie_value(response, sr.REFRESH_COOKIE)
    assert c1 and c1 != c0 and c1.split('.')[0] == first['jti']
    chain = await lineage(result['token'])
    assert chain.generation == 1 and chain.rotated_at == env.clock.now
    assert await valid(result['token']) and await valid(t0)  # old access token lives out its exp

    # The next generation chains on from the new pair.
    env.clock.now += DAY // 2
    result2, _ = await refresh(result['token'], c1)
    assert auth_utils.decode_token(result2['token'])['sid'] == first['jti']
    assert (await lineage(result2['token'])).generation == 2


@pytest.mark.anyio
async def test_refresh_waits_for_half_lifetime(env):
    t0, c0 = await open_login()
    env.clock.now += DAY // 2 - 10
    with pytest.raises(HTTPException) as error:
        await refresh(t0, c0)
    assert error.value.status_code == 429 and error.value.headers['Retry-After'] == '10'
    assert (await lineage(t0)).generation == 0


@pytest.mark.anyio
async def test_refresh_requires_strict_bearer_origin_and_cookie(env):
    t0, c0 = await open_login()
    env.clock.now += DAY // 2
    assert await status_of(None, c0, cookies={'token': t0}) == 401  # cookie-only
    for value in [f'Token {t0}', f'Bearer  {t0}', f'Bearer {t0} x', 'Bearer sk-abc', 'Bearer']:
        request = make_request(None, c0, extra_headers=[('authorization', value)])
        with pytest.raises(HTTPException) as error:
            await sr.refresh_session(request, Response())
        assert error.value.status_code == 401, value
    request = make_request(t0, c0, extra_headers=[('authorization', f'Bearer {t0}')])  # duplicate header
    with pytest.raises(HTTPException):
        await sr.refresh_session(request, Response())
    assert await status_of(t0, c0, origin=None) == 403
    assert await status_of(t0, c0, origin='https://evil.example') == 403
    assert await status_of(t0, c0, extra_headers=[('sec-fetch-site', 'same-site')]) == 403
    assert await status_of(t0, None) == 403  # access token alone never refreshes
    assert (await lineage(t0)).generation == 0
    assert await status_of(t0, c0, extra_headers=[('sec-fetch-site', 'same-origin')]) == 200


@pytest.mark.anyio
async def test_tokens_without_lineage_cannot_refresh(env):
    env.clock.now += DAY
    _, c0 = await open_login()
    for token in [
        login_token(),  # e.g. issued before deployment, by token exchange or /auths/add
        login_token(auth_time=None),
        login_token(typ='automation'),
        login_token(typ='subagent'),
    ]:
        assert await status_of(token, c0) in {401, 403}


@pytest.mark.anyio
async def test_other_login_cookie_or_unknown_secret_never_revokes(env):
    ta, ca = await open_login()
    tb, cb = await open_login()
    env.clock.now += DAY // 2
    assert await status_of(ta, cb) == 403  # browser holds another login's credential
    sid = ca.split('.')[0]
    assert await status_of(ta, f'{sid}.forged-secret') == 403
    assert (await lineage(ta)).revoked_at is None and (await lineage(tb)).revoked_at is None
    assert await valid(ta) and await valid(tb)


@pytest.mark.anyio
async def test_previous_secret_conflicts_in_grace_then_replay_revokes_lineage(env):
    t0, c0 = await open_login()
    env.clock.now += DAY // 2
    result, response = await refresh(t0, c0)
    t1, c1 = result['token'], cookie_value(response, sr.REFRESH_COOKIE)

    env.clock.now += sr.REUSE_GRACE_SECONDS
    assert await status_of(t0, c0) == 409  # concurrent tab / lost response, inside grace
    assert await status_of(t0, c1) == 409  # stale access token, current cookie
    assert (await lineage(t1)).revoked_at is None and await valid(t1)

    env.clock.now += 1
    assert await status_of(t0, c1) == 403  # stale access beyond grace: refused, not treated as theft
    assert (await lineage(t1)).revoked_at is None
    assert await status_of(t0, c0) == 401  # rotated-out refresh secret replayed: theft
    assert (await lineage(t1)).revoked_at == env.clock.now
    assert not await valid(t1) and not await valid(t0)
    assert await status_of(t1, c1) == 401


@pytest.mark.anyio
async def test_absolute_lifetime_is_frozen_and_caps_expiry(env):
    t, c = await open_login()
    chain = await lineage(t)
    for _ in range(59):
        env.clock.now += DAY // 2
        result, response = await refresh(t, c)
        if not result['refreshed']:
            break
        t, c = result['token'], cookie_value(response, sr.REFRESH_COOKIE)
        assert result['expires_at'] <= chain.absolute_expires_at
        # Raising the configured maximum later never extends an existing lineage.
        sr.MAX_LIFETIME_SECONDS = 365 * DAY
    assert auth_utils.decode_token(t)['exp'] == chain.absolute_expires_at
    env.clock.now = chain.absolute_expires_at - 100
    result, _ = await refresh(t, c)
    assert result == {
        'refreshed': False,
        'reason': 'max_lifetime',
        'expires_at': chain.absolute_expires_at,
        'server_time': env.clock.now,
    }


@pytest.mark.anyio
async def test_account_revocation_and_stamp_race_block_renewal(env):
    t0, c0 = await open_login()
    env.clock.now += DAY // 2
    chain = await lineage(t0)
    sid = auth_utils.decode_token(t0)['jti']
    # A revocation that lands between validation and rotation: the conditional update refuses.
    async with env.makers[0]() as session:
        await session.execute(update(Auth).where(Auth.id == 'one').values(session_stamp='rotated'))
        await session.commit()
    assert not await SessionChains.rotate(
        sid,
        user_id='one',
        session_stamp=chain.session_stamp,
        old_secret=chain.current_secret,
        old_jti=chain.current_jti,
        new_secret='n',
        new_jti='n',
        now=env.clock.now,
    )
    assert (await SessionChains.get(sid)).generation == 0
    assert await status_of(t0, c0) == 401  # access token itself now invalid


@pytest.mark.anyio
async def test_mfa_state_is_inherited_never_upgraded(env):
    env.mfa.value = mfa_utils.MfaConfigForm(ENABLE_MFA=True)
    async with env.makers[0]() as session:
        await session.execute(update(Auth).where(Auth.id == 'one').values(mfa={'secret': 'encrypted'}))
        await session.commit()
    t0, c0 = await open_login(mfa_verified=True)
    env.clock.now += DAY // 2
    result, _ = await refresh(t0, c0)
    assert auth_utils.decode_token(result['token'])['mfa_verified'] is True
    # A password token that never passed MFA is not valid, so it cannot be renewed either.
    env.clock.now += DAY
    unverified, cu = await open_login(mfa_verified=False)
    env.clock.now += DAY // 2
    assert await status_of(unverified, cu) == 401


@pytest.mark.anyio
async def test_concurrent_refresh_has_single_winner(env):
    t0, c0 = await open_login()
    env.clock.now += DAY // 2

    async def attempt():
        try:
            result, response = await refresh(t0, c0)
            return 200, result, cookie_value(response, sr.REFRESH_COOKIE)
        except HTTPException as error:
            return error.status_code, None, None

    outcomes = await asyncio.gather(*[attempt() for _ in range(4)])
    assert sorted(code for code, _, _ in outcomes) == [200, 409, 409, 409]
    winner = next(o for o in outcomes if o[0] == 200)
    chain = await lineage(winner[1]['token'])
    assert chain.generation == 1 and chain.revoked_at is None
    assert digest(winner[2].split('.', 1)[1]) == chain.current_secret
    # Three generations in a row from the winner's pair.
    t, c = winner[1]['token'], winner[2]
    for generation in [2, 3]:
        env.clock.now += DAY // 2
        result, response = await refresh(t, c)
        t, c = result['token'], cookie_value(response, sr.REFRESH_COOKIE)
        assert (await lineage(t)).generation == generation


# --- revocation and sign-out -----------------------------------------------------------


@pytest.mark.anyio
async def test_revocation_is_only_ever_extended(env):
    now = env.clock.now
    await SessionRevocations.revoke('x', now + 100)
    await SessionRevocations.revoke('x', now + 50)
    async with env.makers[0]() as session:
        assert (await session.get(RevokedSession, digest('x'))).expires_at == now + 100
    await SessionRevocations.revoke('x', None)
    await SessionRevocations.revoke('x', now + 10)
    async with env.makers[1]() as session:
        assert (await session.get(RevokedSession, digest('x'))).expires_at is None
    assert await SessionRevocations.any_revoked(['unrelated', 'x'])
    assert not await SessionRevocations.any_revoked(['unrelated', None])
    await asyncio.gather(*[SessionRevocations.revoke('y', now + n) for n in range(10, 60, 10)])
    async with env.makers[0]() as session:
        assert (await session.get(RevokedSession, digest('y'))).expires_at == now + 50


@pytest.mark.anyio
async def test_signout_revokes_every_generation_even_with_expired_token(env):
    t0, c0 = await open_login()
    env.clock.now += DAY // 2
    result, response = await refresh(t0, c0)
    t1, c1 = result['token'], cookie_value(response, sr.REFRESH_COOKIE)
    chain = await lineage(t1)
    expired = auth_utils.create_token({**auth_utils.decode_token(t1), 'exp': int(time.time()) - 5, 'jti': None})
    assert auth_utils.decode_token(expired) is None
    keep = await sr.revoke_for_signout(make_request(path='/api/v1/auths/signout'), expired)
    assert keep is False  # no refresh cookie on this request
    assert not await valid(t0) and not await valid(t1)
    async with env.makers[0]() as session:
        row = await session.get(RevokedSession, digest(auth_utils.decode_token(t0)['jti']))  # = sid
        assert row.expires_at == chain.absolute_expires_at
    assert await status_of(t1, c1) == 401


@pytest.mark.anyio
async def test_signout_by_cookie_requires_possession_and_spares_other_logins(env):
    ta, ca = await open_login()
    tb, cb = await open_login()
    # Header names login A while the shared cookie belongs to login B: only A ends.
    request = make_request(ta, cb, path='/api/v1/auths/signout')
    assert await sr.revoke_for_signout(request, ta) is False
    assert not await valid(ta) and await valid(tb)
    # Knowing a lineage id is not enough to end it.
    sid_b = cb.split('.')[0]
    assert await sr.revoke_for_signout(make_request(None, f'{sid_b}.guess'), None) is True
    assert await valid(tb)
    # The browser's own refresh cookie ends its lineage when no token is presented.
    assert await sr.revoke_for_signout(make_request(None, cb), None) is True
    assert not await valid(tb)


# --- HTTP surface ----------------------------------------------------------------------


@pytest.fixture
async def client(env, monkeypatch):
    from open_webui.routers import auths
    from open_webui.socket import main as socket_main

    monkeypatch.setattr(auths, 'Config', FakeConfig)
    monkeypatch.setattr(auths, 'get_permissions', AsyncMock(return_value={}))
    monkeypatch.setattr(auths, 'publish_event', AsyncMock())
    monkeypatch.setattr(socket_main, 'disconnect_user_sessions', AsyncMock())
    app = FastAPI()
    app.state.redis = None
    app.include_router(auths.router, prefix='/api/v1/auths')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=ORIGIN) as http:
        yield http


@pytest.mark.anyio
async def test_refresh_endpoint_headers(env, client):
    t0, c0 = await open_login()
    headers = {'Authorization': f'Bearer {t0}', 'Origin': ORIGIN, 'Cookie': f'{sr.REFRESH_COOKIE}={c0}'}
    early = await client.post('/api/v1/auths/refresh', headers=headers)
    assert early.status_code == 429 and early.headers['retry-after'] == str(DAY // 2)
    assert early.headers['cache-control'] == 'no-store'
    env.clock.now += DAY // 2
    ok = await client.post('/api/v1/auths/refresh', headers=headers)
    assert ok.status_code == 200 and ok.json()['refreshed'] is True
    assert ok.headers['cache-control'] == 'no-store'
    cookies = ok.headers.get_list('set-cookie')
    assert any(c.startswith('token=') for c in cookies) and any(c.startswith(sr.REFRESH_COOKIE) for c in cookies)


@pytest.mark.anyio
async def test_identity_lookup_never_overwrites_another_cookie(env, client):
    ta, _ = await open_login()
    tb, _ = await open_login('two')
    other = await client.get('/api/v1/auths/', headers={'Authorization': f'Bearer {ta}', 'Cookie': f'token={tb}'})
    assert other.status_code == 200 and other.json()['id'] == 'one'
    assert not any(c.startswith('token=') for c in other.headers.get_list('set-cookie'))
    same = await client.get('/api/v1/auths/', headers={'Authorization': f'Bearer {ta}', 'Cookie': f'token={ta}'})
    assert any(c.startswith('token=') for c in same.headers.get_list('set-cookie'))
    bare = await client.get('/api/v1/auths/', headers={'Authorization': f'Bearer {ta}'})
    assert any(c.startswith('token=') for c in bare.headers.get_list('set-cookie'))


@pytest.mark.anyio
async def test_signout_endpoint_ends_lineage_and_clears_cookie(env, client):
    t0, c0 = await open_login()
    response = await client.post(
        '/api/v1/auths/signout',
        headers={'Authorization': f'Bearer {t0}', 'Origin': ORIGIN, 'Cookie': f'{sr.REFRESH_COOKIE}={c0}'},
    )
    assert response.status_code == 200
    assert any(
        c.startswith(f'{sr.REFRESH_COOKIE}=') and 'max-age=0' in c.lower() for c in response.headers.get_list('set-cookie')
    )
    assert (await lineage(t0)).revoked_at is not None and not await valid(t0)


# --- socket ----------------------------------------------------------------------------


@pytest.mark.anyio
async def test_socket_token_swap_requires_same_login(env, monkeypatch):
    from open_webui.socket import main as socket_main

    t0, c0 = await open_login()
    env.clock.now += DAY // 2
    t1 = (await refresh(t0, c0))[0]['token']
    other, _ = await open_login()
    stranger, _ = await open_login('two')
    sessions = {'sid': {'user': {'id': 'one'}, 'token': t0}}
    monkeypatch.setattr(socket_main.sio, 'get_session', AsyncMock(side_effect=lambda sid: dict(sessions[sid])))
    monkeypatch.setattr(socket_main.sio, 'save_session', AsyncMock(side_effect=lambda sid, s: sessions.__setitem__(sid, s)))
    monkeypatch.setattr(socket_main.sio, 'get_environ', lambda sid: {})

    for token in [other, stranger, 'garbage', None]:
        assert await socket_main.session_token('sid', {'token': token}) == {'ok': False}
    assert sessions['sid']['token'] == t0
    assert await socket_main.session_token('sid', {'token': t1}) == {'ok': True}
    assert sessions['sid']['token'] == t1
    assert await socket_main.session_token('sid', {'token': t0}) == {'ok': False}  # never roll back


@pytest.mark.anyio
async def test_socket_join_never_switches_principal(env, monkeypatch):
    from open_webui.socket import main as socket_main

    stranger, _ = await open_login('two')
    monkeypatch.setattr(socket_main.sio, 'get_session', AsyncMock(return_value={'user': {'id': 'one'}, 'token': 'x'}))
    monkeypatch.setattr(socket_main.sio, 'get_environ', lambda sid: {})
    disconnect = AsyncMock()
    save = AsyncMock()
    enter = AsyncMock()
    monkeypatch.setattr(socket_main.sio, 'disconnect', disconnect)
    monkeypatch.setattr(socket_main.sio, 'save_session', save)
    monkeypatch.setattr(socket_main.sio, 'enter_room', enter)
    assert await socket_main.user_join('sid', {'auth': {'token': stranger}}) is None
    disconnect.assert_awaited_once_with('sid')
    save.assert_not_awaited()
    await socket_main.join_channel('sid', {'auth': {'token': stranger}})
    await socket_main.join_note('sid', {'auth': {'token': stranger}, 'note_id': 'n'})
    enter.assert_not_awaited()
