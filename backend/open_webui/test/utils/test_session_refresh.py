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
        for table in [User.__table__, Auth.__table__, OAuthSession.__table__, RevokedSession.__table__, SessionLineage.__table__, security.AuthHandoff.__table__]:
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


@pytest.fixture
def frozen_jwt(env, monkeypatch):
    """Token issuance (iat) and PyJWT's own exp/iat checks follow the test clock, with the real decoder."""
    import datetime as dt

    import jwt.api_jwt as api_jwt

    class FrozenDatetime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.fromtimestamp(env.clock.now, tz or dt.timezone.utc)

    monkeypatch.setattr(auth_utils, 'datetime', FrozenDatetime)
    monkeypatch.setattr(api_jwt, 'datetime', FrozenDatetime)
    return env.clock


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
    assert await sr.start_session_chain(None, response, token, user=SimpleNamespace(id=uid, role='admin' if uid == 'one' else 'user'))
    return token, refresh_cookie_of(response)


def set_cookies(response):
    return [value.decode() for key, value in response.raw_headers if key == b'set-cookie']


def cookie_value(response, name):
    for header in set_cookies(response):
        if header.startswith(f'{name}='):
            return header.split(';', 1)[0].split('=', 1)[1].strip('"')
    return None


def refresh_cookie_of(response):
    """`name=value` of the (per-lineage) refresh cookie a response issued, if any."""
    for header in set_cookies(response):
        pair = header.split(';', 1)[0]
        if pair.startswith(sr.REFRESH_COOKIE_PREFIX) and 'max-age=0' not in header.lower():
            return pair
    return None


def sid_of(refresh_cookie):
    return refresh_cookie.split('=', 1)[0][len(sr.REFRESH_COOKIE_PREFIX) :]


def secret_of(refresh_cookie):
    return refresh_cookie.split('=', 1)[1]


def cleared(headers):
    return {h.split('=', 1)[0] for h in headers if 'max-age=0' in h.lower()}


def make_request(token=None, refresh=None, *, origin=ORIGIN, extra_headers=(), cookies=None, path='/api/v1/auths/refresh'):
    headers = []
    if token is not None:
        headers.append((b'authorization', f'Bearer {token}'.encode()))
    if origin is not None:
        headers.append((b'origin', origin.encode()))
    jar = dict(cookies or {})
    if refresh is not None:
        name, _, value = refresh.partition('=')
        jar[name] = value
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
    decoded = auth_utils.decode_token(token)
    return bool(decoded) and await auth_utils.is_valid_token(decoded)


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
    header = next(h for h in set_cookies(response) if h.startswith(sr.REFRESH_COOKIE_PREFIX))
    lowered = header.lower()
    for attribute in ['httponly', 'samesite=strict', 'path=/api/v1/auths', 'max-age=']:
        assert attribute in lowered
    rc = refresh_cookie_of(response)
    sid, secret = sid_of(rc), secret_of(rc)
    assert sid == auth_utils.decode_token(result['token'])['jti']
    assert digest(secret) == chain.current_secret and secret not in str(chain)

    # API-style issuance (token exchange, no cookie) and non-renewable browser flows never open one.
    for kwargs in [{'set_cookie': False}, {'set_cookie': True, 'renewable': False}]:
        response = Response()
        result = await auth_utils.create_session_response(request, user, response=response, auth=auth, **kwargs)
        assert await lineage(result['token']) is None
        assert refresh_cookie_of(response) is None


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
        assert not await sr.start_session_chain(None, response, token, user=user)
        assert refresh_cookie_of(response) is None
    FakeConfig.values['auth.jwt_expiry'] = '-1'
    assert not await sr.start_session_chain(None, Response(), login_token(), user=human)


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
    c1 = refresh_cookie_of(response)
    assert c1 and c1 != c0 and sid_of(c1) == first['jti']
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
    # With a configured public URL, the request's own (Host-derived) origin is not trusted ...
    FakeConfig.values['webui.url'] = 'https://public.example'
    try:
        assert await status_of(t0, c0) == 403
        # ... except loopback access, e.g. through an SSH tunnel.
        local = make_request(t0, c0, origin='http://127.0.0.1:3000')
        local.scope['scheme'], local.scope['server'] = 'http', ('127.0.0.1', 3000)
        with pytest.raises(HTTPException) as error:
            await sr.require_first_party_origin(make_request(t0, c0, origin='https://chat.example'))
        assert error.value.status_code == 403
        await sr.require_first_party_origin(local)
        await sr.require_first_party_origin(make_request(t0, c0, origin='https://public.example'))
    finally:
        FakeConfig.values['webui.url'] = ORIGIN
    assert await status_of(t0, c0, extra_headers=[('sec-fetch-site', 'same-site')]) == 403
    assert await status_of(t0, None) == 403  # access token alone never refreshes
    sid = sid_of(c0)
    for malformed in [
        f'owui_sr_{sid}=ß√-non-ascii-secret',
        f'owui_sr_{sid}=',
        f'owui_sr_{sid}={"x" * 200}',
        f'owui_session_refresh={sid}.ß√-non-ascii-secret',
        'owui_sr_' + 'é' * 20 + '=' + 'x' * 20,
    ]:
        assert await status_of(t0, malformed) == 403, malformed
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
    sid = sid_of(ca)
    assert await status_of(ta, f'owui_sr_{sid}=forged-secret') == 403
    assert (await lineage(ta)).revoked_at is None and (await lineage(tb)).revoked_at is None
    assert await valid(ta) and await valid(tb)


@pytest.mark.anyio
async def test_previous_secret_conflicts_in_grace_then_replay_revokes_lineage(env):
    t0, c0 = await open_login()
    env.clock.now += DAY // 2
    result, response = await refresh(t0, c0)
    t1, c1 = result['token'], refresh_cookie_of(response)

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
        t, c = result['token'], refresh_cookie_of(response)
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
            return 200, result, refresh_cookie_of(response)
        except HTTPException as error:
            return error.status_code, None, None

    outcomes = await asyncio.gather(*[attempt() for _ in range(4)])
    assert sorted(code for code, _, _ in outcomes) == [200, 409, 409, 409]
    winner = next(o for o in outcomes if o[0] == 200)
    chain = await lineage(winner[1]['token'])
    assert chain.generation == 1 and chain.revoked_at is None
    assert digest(secret_of(winner[2])) == chain.current_secret
    # Three generations in a row from the winner's pair.
    t, c = winner[1]['token'], winner[2]
    for generation in [2, 3]:
        env.clock.now += DAY // 2
        result, response = await refresh(t, c)
        t, c = result['token'], refresh_cookie_of(response)
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
    await asyncio.gather(*[SessionRevocations.revoke('z', value) for value in [now + 5, None, now + 500, now + 1]])
    async with env.makers[1]() as session:
        assert (await session.get(RevokedSession, digest('z'))).expires_at is None
    # An expired revocation is replaced, not extended from the past.
    await SessionRevocations.revoke('w', now - 1)
    await SessionRevocations.revoke('w', now + 30)
    async with env.makers[0]() as session:
        assert (await session.get(RevokedSession, digest('w'))).expires_at == now + 30


@pytest.mark.anyio
async def test_signout_revokes_every_generation_even_with_expired_token(env):
    t0, c0 = await open_login()
    env.clock.now += DAY // 2
    result, response = await refresh(t0, c0)
    t1, c1 = result['token'], refresh_cookie_of(response)
    chain = await lineage(t1)
    expired = auth_utils.create_token({**auth_utils.decode_token(t1), 'exp': int(time.time()) - 5, 'jti': None})
    assert auth_utils.decode_token(expired) is None
    names = await sr.revoke_for_signout(make_request(path='/api/v1/auths/signout'), expired)
    assert names == []  # no refresh cookie on this request
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
    assert await sr.revoke_for_signout(request, ta) == []
    assert not await valid(ta) and await valid(tb)
    # Knowing a lineage id is not enough to end it.
    sid_b = sid_of(cb)
    assert await sr.revoke_for_signout(make_request(None, f'owui_sr_{sid_b}={"x" * 43}'), None) == [f'owui_sr_{sid_b}']
    assert await valid(tb)
    # An explicit but unusable credential never falls back to the shared cookie.
    for bogus in ['garbage', 'sk-abc', login_token(typ='automation', id=None), None]:
        assert await sr.revoke_for_signout(make_request(bogus, cb), bogus, explicit=True) == []
    assert await valid(tb)
    # The browser's own refresh cookie ends its lineage when no token is presented.
    assert await sr.revoke_for_signout(make_request(None, cb), None) == [cb.split('=', 1)[0]]
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
    from starlette.middleware.sessions import SessionMiddleware

    app = FastAPI()
    app.state.redis = None
    app.include_router(auths.router, prefix='/api/v1/auths')

    @app.get('/test/oauth-callback/{uid}')
    async def fake_oauth_callback(uid: str, request: Request, response: Response):
        """What the real callback does for a browser: one-time hand-over, then its lineage."""
        token = login_token(uid, auth_method='oauth', now=env.clock.now)
        assert await sr.issue_oauth_handoff(request, response, token)
        await sr.start_session_chain(request, response, token, user=SimpleNamespace(id=uid, role='user'))
        return {'token': token}

    app.add_middleware(SessionMiddleware, secret_key='session-tests-only')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=ORIGIN) as http:
        yield http


@pytest.mark.anyio
async def test_refresh_endpoint_headers(env, client):
    t0, c0 = await open_login()
    headers = {'Authorization': f'Bearer {t0}', 'Origin': ORIGIN, 'Cookie': c0}
    early = await client.post('/api/v1/auths/refresh', headers=headers)
    assert early.status_code == 429 and early.headers['retry-after'] == str(DAY // 2)
    assert early.headers['cache-control'] == 'no-store'
    env.clock.now += DAY // 2
    ok = await client.post('/api/v1/auths/refresh', headers=headers)
    assert ok.status_code == 200 and ok.json()['refreshed'] is True
    assert ok.headers['cache-control'] == 'no-store'
    cookies = ok.headers.get_list('set-cookie')
    assert any(c.startswith('token=') for c in cookies) and any(c.startswith(sr.REFRESH_COOKIE_PREFIX) for c in cookies)


@pytest.mark.anyio
async def test_token_cookie_only_moves_forward(env, client, frozen_jwt):
    t0, c0 = await open_login(now=env.clock.now)
    env.clock.now += DAY // 2
    result, response = await refresh(t0, c0)
    t1, c1 = result['token'], refresh_cookie_of(response)  # same login, newer generation
    env.clock.now += 10
    fresh_other_user, _ = await open_login('two', now=env.clock.now)
    fresh_same_user, _ = await open_login(now=env.clock.now)
    dead, _ = await open_login(now=env.clock.now)
    await sr.revoke_for_signout(make_request(dead, None), dead, explicit=True)

    async def writes_cookie(bearer, cookie):
        headers = {'Authorization': f'Bearer {bearer}', **({'Cookie': f'token={cookie}'} if cookie else {})}
        response = await client.get('/api/v1/auths/', headers=headers)
        assert response.status_code == 200
        return any(c.startswith('token=') for c in response.headers.get_list('set-cookie'))

    # An older generation or an older login never replaces the cookie (e.g. the OAuth hand-over).
    assert not await writes_cookie(t0, t1)
    assert not await writes_cookie(t1, fresh_other_user)
    assert not await writes_cookie(t1, fresh_same_user)
    # A newer login or generation moves it forward again, repairing a late overwrite.
    assert await writes_cookie(fresh_same_user, t1)
    assert await writes_cookie(t1, t0)
    # Absent, equal, revoked or garbage cookies are (re)written.
    for cookie in [None, t1, dead, 'garbage']:
        assert await writes_cookie(t1, cookie)

    # Two logins authenticated in the same second are not ordered: neither replaces the other.
    twin_a, _ = await open_login(now=env.clock.now)
    twin_b, _ = await open_login(now=env.clock.now)
    assert not await writes_cookie(twin_a, twin_b) and not await writes_cookie(twin_b, twin_a)
    # When the cookie's validity cannot be determined, it is left alone.
    original = auth_utils.is_valid_token

    async def unavailable(decoded, redis=None):
        if decoded.get('jti') == auth_utils.decode_token(dead)['jti']:
            raise HTTPException(503, 'Authentication is temporarily unavailable.')
        return await original(decoded, redis)

    auth_utils.is_valid_token = unavailable
    try:
        assert not await writes_cookie(t1, dead)
    finally:
        auth_utils.is_valid_token = original

    # A renewal of an older login keeps its own refresh cookie but leaves a newer login's token cookie.
    env.clock.now += DAY // 2
    _, response = await refresh(t1, c1, cookies={'token': fresh_same_user})
    assert cookie_value(response, 'token') is None and refresh_cookie_of(response)


@pytest.mark.anyio
async def test_signout_rejects_bad_authorization_without_side_effects(env, client):
    t0, c0 = await open_login()
    forged = t0[:-4] + ('AAAA' if not t0.endswith('AAAA') else 'BBBB')
    for header, status in [
        ('Bearer', 400),
        ('Basic Zm9vOmJhcg==', 400),
        ('Bearer a b', 400),
        ('Bearer sk-abc', 400),
        (f'Bearer {forged}', 401),
    ]:
        response = await client.post(
            '/api/v1/auths/signout',
            headers={'Authorization': header, 'Origin': ORIGIN, 'Cookie': f'{c0}; token={t0}'},
        )
        assert response.status_code == status, header
        assert response.headers.get_list('set-cookie') == []
    duplicate = await client.post(
        '/api/v1/auths/signout',
        headers=[('Authorization', f'Bearer {t0}'), ('Authorization', f'Bearer {t0}'), ('Origin', ORIGIN), ('Cookie', c0)],
    )
    assert duplicate.status_code == 400 and duplicate.headers.get_list('set-cookie') == []
    assert (await lineage(t0)).revoked_at is None
    assert await valid(t0)  # the cookie login was not touched in place of the bad header


@pytest.mark.anyio
async def test_signout_endpoint_ends_lineage_and_clears_cookie(env, client):
    t0, c0 = await open_login()
    response = await client.post(
        '/api/v1/auths/signout',
        headers={'Authorization': f'Bearer {t0}', 'Origin': ORIGIN, 'Cookie': f'{c0}; token={t0}'},
    )
    assert response.status_code == 200
    assert {c0.split('=', 1)[0], 'token'} <= cleared(response.headers.get_list('set-cookie'))
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


@pytest.mark.anyio
async def test_socket_room_joins_bind_unbound_connections(env, monkeypatch):
    from open_webui.socket import main as socket_main

    token, _ = await open_login()
    sessions = {'sid': {}}
    monkeypatch.setattr(socket_main.sio, 'get_session', AsyncMock(side_effect=lambda sid: dict(sessions[sid])))
    monkeypatch.setattr(socket_main.sio, 'save_session', AsyncMock(side_effect=lambda sid, s: sessions.__setitem__(sid, s)))
    monkeypatch.setattr(socket_main.sio, 'get_environ', lambda sid: {})
    monkeypatch.setattr(socket_main.sio, 'enter_room', AsyncMock())
    monkeypatch.setattr(socket_main, 'has_permission', AsyncMock(return_value=False))
    monkeypatch.setattr(socket_main, 'Config', FakeConfig)
    await socket_main.join_channel('sid', {'auth': {'token': token}})
    # The connection is now bound, so the 30 s re-validation covers it.
    assert sessions['sid']['user']['id'] == 'one' and sessions['sid']['token'] == token
    assert 'sid' in socket_main.LOCAL_AUTHENTICATED_SIDS
    socket_main.LOCAL_AUTHENTICATED_SIDS.discard('sid')
    socket_main.SESSION_POOL.pop('sid', None)

    monkeypatch.setattr(socket_main.sio, 'get_session', AsyncMock(side_effect=RuntimeError('gone')))
    enter = AsyncMock()
    monkeypatch.setattr(socket_main.sio, 'enter_room', enter)
    await socket_main.join_channel('sid', {'auth': {'token': token}})
    await socket_main.user_join('sid', {'auth': {'token': token}})
    enter.assert_not_awaited()


@pytest.mark.anyio
async def test_socket_user_join_never_rolls_token_back(env, monkeypatch):
    from open_webui.socket import main as socket_main

    t0, c0 = await open_login()
    env.clock.now += DAY // 2
    t1 = (await refresh(t0, c0))[0]['token']
    sessions = {'sid': {'user': {'id': 'one'}, 'token': t1}}
    monkeypatch.setattr(socket_main.sio, 'get_session', AsyncMock(side_effect=lambda sid: dict(sessions[sid])))
    monkeypatch.setattr(socket_main.sio, 'save_session', AsyncMock(side_effect=lambda sid, s: sessions.__setitem__(sid, s)))
    monkeypatch.setattr(socket_main.sio, 'get_environ', lambda sid: {})
    monkeypatch.setattr(socket_main.sio, 'enter_room', AsyncMock())
    monkeypatch.setattr(socket_main, 'has_permission', AsyncMock(return_value=False))
    monkeypatch.setattr(socket_main, 'Config', FakeConfig)
    disconnect = AsyncMock()
    monkeypatch.setattr(socket_main.sio, 'disconnect', disconnect)
    await socket_main.user_join('sid', {'auth': {'token': t0}})
    assert sessions['sid']['token'] == t1  # never rolled back to an older generation
    disconnect.assert_not_awaited()
    other_login, _ = await open_login()
    await socket_main.user_join('sid', {'auth': {'token': other_login}})
    assert sessions['sid']['token'] == t1  # another live login must reconnect instead
    disconnect.assert_awaited_once_with('sid')
    sessions['sid'] = {'user': {'id': 'one'}, 'token': 'expired-or-garbage'}
    await socket_main.user_join('sid', {'auth': {'token': t1}})
    assert disconnect.await_count == 2 and sessions['sid']['token'] == 'expired-or-garbage'
    socket_main.LOCAL_AUTHENTICATED_SIDS.discard('sid')
    socket_main.SESSION_POOL.pop('sid', None)


@pytest.mark.anyio
async def test_one_clock_drives_issue_rotation_and_jwt_expiry(env, frozen_jwt):
    """Issue, renew and expire on one clock, validated by the production decoder (PyJWT's own exp check)."""
    FakeConfig.values['auth.jwt_expiry'] = '5m'
    t0, c0 = await open_login(now=env.clock.now, lifetime=300)
    assert auth_utils.decode_token(t0)['iat'] == env.clock.now

    env.clock.now += 149
    assert await status_of(t0, c0) == 429
    env.clock.now += 1
    result, response = await refresh(t0, c0)
    t1, c1 = result['token'], refresh_cookie_of(response)
    claims = auth_utils.decode_token(t1)
    assert claims['iat'] == env.clock.now and claims['exp'] == env.clock.now + 300

    env.clock.now = claims['exp'] - 1
    assert auth_utils.decode_token(t1) is not None and await valid(t1)
    env.clock.now = claims['exp']
    assert auth_utils.decode_token(t1) is None  # PyJWT: expired at exp
    assert await status_of(t1, c1) == 401  # idle past the access lifetime: sign in again
    assert auth_utils.decode_token(t0) is None


# --- P1: one-time OAuth hand-over, per-login cookies, superseded logins -------------------


class Jar:
    """A minimal browser cookie jar: responses may be applied in any order (late responses)."""

    def __init__(self):
        self.cookies = {}

    def header(self):
        return '; '.join(f'{name}={value}' for name, value in self.cookies.items())

    def apply(self, set_cookie_headers):
        for header in set_cookie_headers:
            pair, _, attributes = header.partition(';')
            name, _, value = pair.strip().partition('=')
            if 'max-age=0' in attributes.lower():
                self.cookies.pop(name, None)
            else:
                self.cookies[name] = value.strip('"')

    def refresh_cookies(self):
        return {k: v for k, v in self.cookies.items() if k.startswith(sr.REFRESH_COOKIE_PREFIX)}


async def browser_login(jar, uid='one', *, now):
    """A first-party login answered to this browser: shared token cookie plus its own refresh cookie."""
    token = login_token(uid, now=now)
    response = Response()
    response.set_cookie('token', token, httponly=True)
    request = make_request(cookies=dict(jar.cookies), path='/api/v1/auths/signin')
    assert await sr.start_session_chain(request, response, token, user=SimpleNamespace(id=uid, role='user'))
    jar.apply(set_cookies(response))
    return token


@pytest.mark.anyio
async def test_new_login_supersedes_this_browsers_earlier_logins(env, frozen_jwt):
    jar = Jar()
    ta = await browser_login(jar, now=env.clock.now)
    env.clock.now += 5
    tb = await browser_login(jar, now=env.clock.now)
    assert set(jar.refresh_cookies()) == {sr.refresh_cookie_name(auth_utils.decode_token(tb)['jti'])}
    assert not await valid(ta) and (await lineage(ta)).revoked_at is not None
    assert await valid(tb)
    # Another browser's login is untouched: it never sent its refresh cookie here.
    other_browser = Jar()
    tc = await browser_login(other_browser, now=env.clock.now)
    assert await valid(tc) and await valid(tb)


@pytest.mark.anyio
@pytest.mark.parametrize('second_user', ['one', 'two'])
@pytest.mark.parametrize('processed_before_new_login', [True, False])
async def test_late_responses_of_an_earlier_login_never_take_over_the_browser(
    env, client, frozen_jwt, second_user, processed_before_new_login
):
    jar = Jar()
    ta = await browser_login(jar, now=env.clock.now)
    env.clock.now += DAY // 2
    snapshot = jar.header()  # what the earlier login's in-flight requests carried

    async def earlier_login_requests():
        late_get = await client.get('/api/v1/auths/', headers={'Authorization': f'Bearer {ta}', 'Cookie': snapshot})
        late_refresh = await client.post(
            '/api/v1/auths/refresh', headers={'Authorization': f'Bearer {ta}', 'Origin': ORIGIN, 'Cookie': snapshot}
        )
        return [late_get, late_refresh]

    late = await earlier_login_requests() if processed_before_new_login else None
    tb = await browser_login(jar, second_user, now=env.clock.now)
    refresh_b = jar.refresh_cookies()
    if not processed_before_new_login:
        late = await earlier_login_requests()
        # The earlier login was superseded: its requests fail without touching the browser's cookies.
        assert [r.status_code for r in late] == [401, 401]
        assert all(r.headers.get_list('set-cookie') == [] for r in late)
    for response in late:  # delivered after the new login, in this order
        jar.apply(response.headers.get_list('set-cookie'))

    # The new login keeps its own refresh credential; any other refresh cookie belongs to a dead login.
    assert {k: v for k, v in jar.refresh_cookies().items() if k in refresh_b} == refresh_b
    for name in set(jar.refresh_cookies()) - set(refresh_b):
        assert (await SessionChains.get(name[len(sr.REFRESH_COOKIE_PREFIX) :])).revoked_at is not None
    # A cookie-only request never acts as the earlier (now revoked) login.
    bare = await client.get('/api/v1/auths/', headers={'Cookie': jar.header()})
    jar.apply(bare.headers.get_list('set-cookie'))
    if bare.status_code == 200:
        assert bare.json()['token'] == tb
    else:
        assert bare.status_code == 401
    # The next request of the current login moves the shared cookie to it.
    current = await client.get('/api/v1/auths/', headers={'Authorization': f'Bearer {tb}', 'Cookie': jar.header()})
    jar.apply(current.headers.get_list('set-cookie'))
    bare = await client.get('/api/v1/auths/', headers={'Cookie': jar.header()})
    assert bare.status_code == 200 and bare.json()['id'] == second_user and bare.json()['token'] == tb
    # The current login can renew; signing it out leaves no credential in the browser.
    env.clock.now += DAY // 2
    renewed = await client.post(
        '/api/v1/auths/refresh', headers={'Authorization': f'Bearer {tb}', 'Origin': ORIGIN, 'Cookie': jar.header()}
    )
    assert renewed.status_code == 200
    jar.apply(renewed.headers.get_list('set-cookie'))
    tb1 = renewed.json()['token']
    out = await client.post(
        '/api/v1/auths/signout', headers={'Authorization': f'Bearer {tb1}', 'Origin': ORIGIN, 'Cookie': jar.header()}
    )
    assert out.status_code == 200
    jar.apply(out.headers.get_list('set-cookie'))
    assert 'token' not in jar.cookies and jar.refresh_cookies() == {}
    assert not await valid(ta) and not await valid(tb) and not await valid(tb1)


async def oauth_callback(client, jar, uid='one'):
    response = await client.get(f'/test/oauth-callback/{uid}', headers={'Cookie': jar.header()})
    jar.apply(response.headers.get_list('set-cookie'))
    return response.json()['token']


async def oauth_handoff(client, cookie_header, origin=ORIGIN):
    return await client.post('/api/v1/auths/oauth/handoff', headers={'Origin': origin, 'Cookie': cookie_header})


@pytest.mark.anyio
async def test_oauth_handoff_ticket_is_single_use_and_bound_to_the_browser(env, client, frozen_jwt):
    jar = Jar()
    old = await browser_login(jar, now=env.clock.now)
    old_refresh = set(jar.refresh_cookies())
    env.clock.now += 10
    fresh = await oauth_callback(client, jar)
    ticket = jar.cookies[sr.HANDOFF_COOKIE]
    # The browser holds only the ticket; the shared cookie still belongs to the earlier login.
    assert ticket != fresh and fresh not in jar.header() and jar.cookies['token'] != fresh

    # A refused origin neither signs in nor consumes the pending hand-over.
    refused = await oauth_handoff(client, jar.header(), origin='https://evil.example')
    assert refused.status_code == 403 and refused.headers.get_list('set-cookie') == []
    # The ticket alone, without this browser's server session, is useless and is not consumed either.
    stolen = await oauth_handoff(client, f'{sr.HANDOFF_COOKIE}={ticket}')
    assert stolen.status_code == 401
    assert (await oauth_handoff(client, 'token=x')).status_code == 404

    before = jar.header()
    response = await oauth_handoff(client, before)
    assert response.status_code == 200 and response.headers['cache-control'] == 'no-store'
    body = response.json()
    assert body['token'] == fresh and body['id'] == 'one' and body['expires_at'] == auth_utils.decode_token(fresh)['exp']
    headers = response.headers.get_list('set-cookie')
    jar.apply(headers)
    assert sr.HANDOFF_COOKIE not in jar.cookies and jar.cookies['token'] == fresh
    # The browser's earlier login ended and its refresh cookie is gone; the OAuth login's stays.
    assert not await valid(old) and not (old_refresh & set(jar.refresh_cookies()))
    assert sid_of(next(iter(jar.refresh_cookies())) + '=x') == auth_utils.decode_token(fresh)['jti']
    # Replaying the exact pre-hand-over cookies fails: the ticket was consumed.
    assert (await oauth_handoff(client, before)).status_code == 401


@pytest.mark.anyio
async def test_stale_oauth_handoff_never_ends_a_newer_login(env, client, frozen_jwt):
    jar = Jar()
    await oauth_callback(client, jar)  # sign-in A completes at the IdP, hand-over still pending
    pending = {k: v for k, v in jar.cookies.items() if not k.startswith(sr.REFRESH_COOKIE_PREFIX)}
    env.clock.now += 5
    newer_browser = Jar()
    newer = await browser_login(newer_browser, now=env.clock.now)  # a later sign-in B in this browser
    jar.cookies = {**pending, **newer_browser.cookies}
    before = jar.header()
    response = await oauth_handoff(client, before)
    assert response.status_code == 409 and 'token' not in response.text
    assert not any(h.startswith('token=') for h in response.headers.get_list('set-cookie'))
    jar.apply(response.headers.get_list('set-cookie'))
    assert jar.cookies['token'] == newer  # the browser keeps using the newer login
    assert await valid(newer) and (await lineage(newer)).revoked_at is None
    # The stale ticket is gone for good, even if the browser sends it again.
    assert (await oauth_handoff(client, before)).status_code == 401


@pytest.mark.anyio
async def test_handoff_ticket_expires(env, client, frozen_jwt):
    jar = Jar()
    await oauth_callback(client, jar)
    env.clock.now += sr.HANDOFF_MAX_AGE + 1
    assert (await oauth_handoff(client, jar.header())).status_code == 401


@pytest.mark.anyio
async def test_superseding_keeps_the_cookie_when_ending_fails(env, monkeypatch, frozen_jwt):
    jar = Jar()
    old = await browser_login(jar, now=env.clock.now)
    env.clock.now += 5
    monkeypatch.setattr(sr, 'revoke_lineage', AsyncMock(side_effect=RuntimeError('database busy')))
    await browser_login(jar, now=env.clock.now)
    assert sr.refresh_cookie_name(auth_utils.decode_token(old)['jti']) in jar.refresh_cookies()


@pytest.mark.anyio
async def test_renewable_login_uses_its_own_oauth_session(env, monkeypatch):
    from open_webui.utils import oauth as oauth_utils

    token = login_token(auth_method='oauth')
    assert await sr.start_session_chain(None, Response(), token, user=SimpleNamespace(id='one', role='user'), oauth_session_id='os-bound')
    get_token = AsyncMock(return_value=None)
    request = make_request(token, cookies={'oauth_session_id': 'os-other'})
    request.scope['app'].state.oauth_manager = SimpleNamespace(get_oauth_token=get_token)
    request.state.claims, request.state.auth_type = auth_utils.decode_token(token), 'jwt'
    monkeypatch.setattr(oauth_utils.OAuthSessions, 'get_sessions_by_user_id', AsyncMock(return_value=[SimpleNamespace(id='os-recent', provider='google', updated_at=1)]))
    assert await oauth_utils.get_system_oauth_token(request, SimpleNamespace(id='one')) is None
    get_token.assert_awaited_once_with('one', 'os-bound')  # no cookie, no "most recent session" fallback


@pytest.mark.anyio
async def test_legacy_and_new_cookie_names_for_one_login_are_both_settled(env):
    token = login_token()
    response = Response()
    assert await sr.start_session_chain(None, response, token, user=SimpleNamespace(id='one', role='user'))
    rc = refresh_cookie_of(response)
    env.clock.now += DAY // 2
    request = make_request(token, rc, cookies={sr.LEGACY_REFRESH_COOKIE: f'{sid_of(rc)}.{"y" * 43}'})
    renewed = Response()
    assert (await sr.refresh_session(request, renewed))['refreshed'] is True
    assert sr.LEGACY_REFRESH_COOKIE in cleared(set_cookies(renewed))


@pytest.mark.anyio
async def test_legacy_refresh_cookie_is_accepted_and_migrated(env):
    token = login_token()
    response = Response()
    assert await sr.start_session_chain(None, response, token, user=SimpleNamespace(id='one', role='user'))
    rc = refresh_cookie_of(response)
    legacy = f'{sr.LEGACY_REFRESH_COOKIE}={sid_of(rc)}.{secret_of(rc)}'
    env.clock.now += DAY // 2
    result, renewed = await refresh(token, legacy)
    assert result['refreshed'] is True
    assert sr.LEGACY_REFRESH_COOKIE in cleared(set_cookies(renewed))
    assert sid_of(refresh_cookie_of(renewed)) == sid_of(rc)


@pytest.mark.anyio
async def test_a_bad_bearer_never_clears_the_cookie_login(env, client):
    good, _ = await open_login()
    bad = good[:-4] + ('AAAA' if not good.endswith('AAAA') else 'BBBB')
    response = await client.get('/api/v1/auths/', headers={'Authorization': f'Bearer {bad}', 'Cookie': f'token={good}'})
    assert response.status_code == 401 and 'token' not in cleared(response.headers.get_list('set-cookie'))
    assert (await client.get('/api/v1/auths/', headers={'Cookie': f'token={good}'})).status_code == 200


@pytest.mark.anyio
async def test_signout_only_clears_shared_cookies_of_the_login_it_ends(env, client):
    ta, ca = await open_login()
    tb, cb = await open_login()
    # Named login A while the shared cookie holds live login B: A ends, B's cookies stay.
    response = await client.post(
        '/api/v1/auths/signout', headers={'Authorization': f'Bearer {ta}', 'Origin': ORIGIN, 'Cookie': f'{ca}; {cb}; token={tb}'}
    )
    assert response.status_code == 200
    gone = cleared(response.headers.get_list('set-cookie'))
    assert ca.split('=', 1)[0] in gone and cb.split('=', 1)[0] not in gone and 'token' not in gone
    assert not await valid(ta) and await valid(tb)
    # Header-less sign-out ends every login this browser can prove and clears everything.
    response = await client.post('/api/v1/auths/signout', headers={'Origin': ORIGIN, 'Cookie': f'{cb}; token={tb}'})
    assert {cb.split('=', 1)[0], 'token'} <= cleared(response.headers.get_list('set-cookie'))
    assert not await valid(tb)
