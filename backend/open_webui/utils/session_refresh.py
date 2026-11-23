"""Sliding sessions for first-party browser logins.

An access JWT alone can never be renewed: it is also handed to session-mode tools and
upstreams. Renewal additionally needs the rotating refresh secret that only the browser
holds, in an HttpOnly, SameSite=Strict cookie scoped to /api/v1/auths, bound to a lineage
row created at interactive login. Renewed tokens inherit the lineage's frozen
session_stamp, auth_method, auth_time and MFA state, never fresher values, and can never
outlive the absolute expiry frozen at login.
"""

from __future__ import annotations

import hmac
import logging
import re
import secrets
import time
from urllib.parse import urlsplit

from fastapi import HTTPException, Request, Response
from open_webui.config import ENABLE_SESSION_REFRESH, SESSION_MAX_LIFETIME
from open_webui.constants import ERROR_MESSAGES
from open_webui.env import (
    WEBUI_AUTH_COOKIE_SAME_SITE,
    WEBUI_AUTH_COOKIE_SECURE,
    WEBUI_AUTH_TRUSTED_EMAIL_HEADER,
)
from open_webui.models.config import Config
from open_webui.models.oauth_sessions import OAuthSessions
from open_webui.models.security import SessionChains, SessionRevocations, digest
from open_webui.models.users import Users
from open_webui.utils import auth as auth_utils
from open_webui.utils.misc import parse_duration

log = logging.getLogger(__name__)

# One refresh cookie per lineage (`owui_sr_<sid>`), so renewing one login can never overwrite the
# refresh credential of another login in the same browser. `owui_session_refresh` (`<sid>.<secret>`)
# is the single shared name used before; it is still accepted and migrated on use.
REFRESH_COOKIE_PREFIX = 'owui_sr_'
LEGACY_REFRESH_COOKIE = 'owui_session_refresh'
REFRESH_COOKIE_PATH = '/api/v1/auths'
# One-time OAuth sign-in hand-over: the callback parks the new token here (HttpOnly, path-scoped, short
# lived) instead of in the shared, JS-readable `token` cookie; POST /api/v1/auths/oauth/handoff consumes it.
HANDOFF_COOKIE = 'owui_oauth_handoff'
HANDOFF_COOKIE_PATH = '/api/v1/auths/oauth/handoff'
HANDOFF_MAX_AGE = 120
REUSE_GRACE_SECONDS = 120
MIN_REFRESH_INTERVAL_SECONDS = 30
MIN_ACCESS_LIFETIME_SECONDS = 300
CLOCK_SKEW_SECONDS = 30
HUMAN_ROLES = {'admin', 'user'}

UNAVAILABLE = 'session_refresh_unavailable'
CONFLICT = 'session_refresh_conflict'
TOO_SOON = 'session_refresh_too_soon'
STALE = 'session_refresh_stale_token'
REVOKED = 'Session has been revoked. Please sign in again.'

_DURATION = re.compile(r'(\d{1,6})(m|h|d|w)')
_UNIT_SECONDS = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800}


def parse_max_lifetime(value) -> int | None:
    """Strict, bounded absolute session lifetime (1h..365d); None when invalid."""
    match = _DURATION.fullmatch(str(value or '').strip())
    if not match:
        return None
    seconds = int(match.group(1)) * _UNIT_SECONDS[match.group(2)]
    return seconds if 3600 <= seconds <= 365 * 86400 else None


MAX_LIFETIME_SECONDS = parse_max_lifetime(SESSION_MAX_LIFETIME)
if ENABLE_SESSION_REFRESH and MAX_LIFETIME_SECONDS is None:
    log.error(
        'SESSION_MAX_LIFETIME=%r is invalid (use e.g. 30d, between 1h and 365d); session refresh is disabled.',
        SESSION_MAX_LIFETIME,
    )


def _int(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _str(value):
    return value if isinstance(value, str) and value else None


async def get_refresh_policy() -> tuple[int, int] | None:
    """(access lifetime, absolute lifetime) in seconds when renewal is usable, else None."""
    if not ENABLE_SESSION_REFRESH or MAX_LIFETIME_SECONDS is None:
        return None
    try:
        delta = parse_duration(str(await Config.get('auth.jwt_expiry')))
    except (TypeError, ValueError):
        return None
    if delta is None:
        return None
    lifetime = int(delta.total_seconds())
    if lifetime < MIN_ACCESS_LIFETIME_SECONDS or lifetime > MAX_LIFETIME_SECONDS:
        return None
    return lifetime, MAX_LIFETIME_SECONDS


def frozen_session_claims(claims) -> dict | None:
    """The claims of a complete interactive session token, or None when the token may not take part in renewal."""
    if not isinstance(claims, dict) or claims.get('typ') != 'session':
        return None
    values = {
        'id': _str(claims.get('id')),
        'jti': _str(claims.get('jti')),
        'session_stamp': _str(claims.get('session_stamp')),
        'auth_method': _str(claims.get('auth_method')),
        'iat': _int(claims.get('iat')),
        'exp': _int(claims.get('exp')),
        'auth_time': _int(claims.get('auth_time')),
    }
    if any(value is None for value in values.values()):
        return None
    if not isinstance(claims.get('mfa_verified'), bool) or len(values['auth_method']) > 32:
        return None
    sid = _str(claims.get('sid', values['jti']))
    if sid is None:
        return None
    if not values['auth_time'] <= values['iat'] + CLOCK_SKEW_SECONDS or values['iat'] >= values['exp']:
        return None
    return {**values, 'sid': sid, 'mfa_verified': claims['mfa_verified']}


_COOKIE_PART = re.compile(r'[A-Za-z0-9_-]{8,128}')


def refresh_cookie_name(sid: str) -> str:
    return f'{REFRESH_COOKIE_PREFIX}{sid}'


def browser_refresh_credentials(request: Request) -> dict[str, tuple[str, str]]:
    """sid -> (cookie name, secret) for every well-formed refresh credential this browser sent."""
    found = {}
    for name, value in request.cookies.items():
        if name.startswith(REFRESH_COOKIE_PREFIX):
            sid = name[len(REFRESH_COOKIE_PREFIX) :]
            if _COOKIE_PART.fullmatch(sid) and value and _COOKIE_PART.fullmatch(value):
                found[sid] = (name, value)
    legacy = request.cookies.get(LEGACY_REFRESH_COOKIE)
    if legacy and len(legacy) <= 256:
        sid, separator, secret = legacy.partition('.')
        if separator and _COOKIE_PART.fullmatch(sid) and _COOKIE_PART.fullmatch(secret):
            found.setdefault(sid, (LEGACY_REFRESH_COOKIE, secret))
    return found


def set_refresh_cookie(response: Response, sid: str, secret: str, max_age: int):
    response.set_cookie(
        refresh_cookie_name(sid),
        secret,
        max_age=max_age,
        path=REFRESH_COOKIE_PATH,
        httponly=True,
        secure=WEBUI_AUTH_COOKIE_SECURE,
        samesite='strict',
    )


def clear_refresh_cookie(response: Response, name: str):
    response.delete_cookie(
        name,
        path=REFRESH_COOKIE_PATH,
        httponly=True,
        secure=WEBUI_AUTH_COOKIE_SECURE,
        samesite='strict',
    )


def set_oauth_handoff(response: Response, token: str):
    response.set_cookie(
        HANDOFF_COOKIE,
        token,
        max_age=HANDOFF_MAX_AGE,
        path=HANDOFF_COOKIE_PATH,
        httponly=True,
        secure=WEBUI_AUTH_COOKIE_SECURE,
        samesite='lax',
    )


def clear_oauth_handoff(response: Response):
    response.delete_cookie(
        HANDOFF_COOKIE,
        path=HANDOFF_COOKIE_PATH,
        httponly=True,
        secure=WEBUI_AUTH_COOKIE_SECURE,
        samesite='lax',
    )


def _matches(candidate: str, expected: str | None) -> bool:
    return bool(expected) and hmac.compare_digest(candidate, expected)


def _holds(chain, secret: str) -> bool:
    secret_hash = digest(secret)
    return _matches(secret_hash, chain.current_secret) or _matches(secret_hash, chain.previous_secret)


def login_id(claims: dict) -> str | None:
    return _str(claims.get('sid')) or _str(claims.get('jti'))


async def supersede_browser_lineages(request: Request, response: Response, keep_sid: str | None) -> list[str]:
    """A new first-party login in this browser ends the browser's earlier renewable logins.

    Their refresh cookies (sent because they are scoped to /api/v1/auths) prove possession: those lineages
    are revoked and every other refresh cookie is cleared, so late responses of a previous login can neither
    renew it nor carry its identity back into the shared cookies. Never fails the login itself."""
    if request is None or response is None:
        return []
    ended = []
    now = int(time.time())
    for sid, (name, secret) in browser_refresh_credentials(request).items():
        if sid == keep_sid:
            continue
        try:
            chain = await SessionChains.get(sid)
            if chain and chain.revoked_at is None and _holds(chain, secret):
                await revoke_lineage(sid, now, reason='superseded by a new login in this browser')
                ended.append(sid)
        except Exception:
            log.exception('Could not end an earlier login of this browser; continuing with the new login.')
        clear_refresh_cookie(response, name)
    return ended


async def start_session_chain(request, response, token: str, *, user, oauth_session_id=None) -> bool:
    """Open a renewable lineage for a token just issued by a first-party interactive login."""
    if response is None or getattr(user, 'role', None) not in HUMAN_ROLES:
        return False
    policy = await get_refresh_policy()
    if policy is None:
        return False
    _, max_lifetime = policy
    claims = frozen_session_claims(auth_utils.decode_token(token))
    if claims is None or claims['sid'] != claims['jti'] or claims['id'] != user.id:
        return False
    now = int(time.time())
    absolute_expires_at = claims['auth_time'] + max_lifetime
    if claims['exp'] > absolute_expires_at or absolute_expires_at <= now:
        return False
    secret = secrets.token_urlsafe(32)
    try:
        await SessionChains.create(
            claims['sid'],
            user_id=claims['id'],
            session_stamp=claims['session_stamp'],
            auth_method=claims['auth_method'],
            auth_time=claims['auth_time'],
            mfa_verified=claims['mfa_verified'],
            absolute_expires_at=absolute_expires_at,
            secret_hash=digest(secret),
            jti_hash=digest(claims['jti']),
            now=now,
            oauth_session_id=oauth_session_id,
        )
    except Exception:
        log.exception('Could not start a renewable session; the login continues without sliding renewal.')
        return False
    await supersede_browser_lineages(request, response, keep_sid=claims['sid'])
    set_refresh_cookie(response, claims['sid'], secret, claims['exp'] - now)
    return True


LOOPBACK_HOSTS = {'localhost', '127.0.0.1', '::1'}


def _origin_of(url) -> tuple[str, str] | None:
    parts = urlsplit(str(url or ''))
    if not parts.scheme or not parts.netloc:
        return None
    return f'{parts.scheme}://{parts.netloc}', (parts.hostname or '')


async def require_first_party_origin(request: Request):
    """The configured public origin; the request's own origin only when none is configured or it is loopback."""
    allowed = set()
    configured = _origin_of(await Config.get('webui.url'))
    if configured:
        allowed.add(configured[0])
    own = _origin_of(request.base_url)
    if own and (configured is None or own[1] in LOOPBACK_HOSTS):
        allowed.add(own[0])
    if request.headers.get('origin') not in allowed:
        raise HTTPException(403, 'Invalid request origin.')
    site = request.headers.get('sec-fetch-site')
    if site is not None and site != 'same-origin':
        raise HTTPException(403, 'Invalid request origin.')


def strict_bearer_token(request: Request) -> str:
    """Exactly one `Authorization: Bearer <jwt>`; no cookie, state or API-key fallback."""
    values = request.headers.getlist('authorization')
    if len(values) != 1:
        raise HTTPException(401, 'Not authenticated')
    scheme, separator, credential = values[0].partition(' ')
    if (
        not separator
        or scheme.lower() != 'bearer'
        or not credential
        or credential != credential.strip()
        or ' ' in credential
        or credential.startswith('sk-')
    ):
        raise HTTPException(401, 'Not authenticated')
    return credential


async def revoke_lineage(sid: str, now: int | None = None, *, reason: str) -> bool:
    """Revoke every past and future token of one lineage until its frozen absolute expiry.

    The durable token revocation is written first: once it commits, every generation is rejected
    by is_valid_token (and therefore cannot refresh), even if marking the lineage then fails."""
    now = now or int(time.time())
    chain = await SessionChains.get(sid)
    if chain is None:
        return False
    await SessionRevocations.revoke(sid, chain.absolute_expires_at)
    chain = await SessionChains.mark_revoked(sid, now) or chain
    if reason == 'replay':
        log.warning(
            'Session refresh secret replay detected for user %s (lineage %s); lineage revoked.',
            chain.user_id,
            chain.id[:12],
        )
    else:
        log.info('Session lineage %s of user %s revoked (%s).', chain.id[:12], chain.user_id, reason)
    return True


async def refresh_session(request: Request, response: Response) -> dict:
    policy = await get_refresh_policy()
    if policy is None:
        raise HTTPException(403, UNAVAILABLE)
    lifetime, _ = policy
    await require_first_party_origin(request)

    token = strict_bearer_token(request)
    decoded = auth_utils.decode_token(token)
    if (
        not isinstance(decoded, dict)
        or not _str(decoded.get('id'))
        or not await auth_utils.is_valid_token(decoded, getattr(request.app.state, 'redis', None))
    ):
        raise HTTPException(401, ERROR_MESSAGES.INVALID_TOKEN)
    claims = frozen_session_claims(decoded)
    if claims is None:
        raise HTTPException(403, UNAVAILABLE)
    user = await Users.get_user_by_id(claims['id'])
    if user is None:
        raise HTTPException(401, ERROR_MESSAGES.INVALID_TOKEN)
    if WEBUI_AUTH_TRUSTED_EMAIL_HEADER:
        trusted_email = request.headers.get(WEBUI_AUTH_TRUSTED_EMAIL_HEADER, '').lower()
        if trusted_email and user.email != trusted_email:
            raise HTTPException(401, 'User mismatch. Please sign in again.')
    if user.role not in HUMAN_ROLES:
        raise HTTPException(403, UNAVAILABLE)

    credential = browser_refresh_credentials(request).get(claims['sid'])
    if credential is None:
        # No refresh credential for this login in this browser (missing, malformed, other logins only).
        raise HTTPException(403, UNAVAILABLE)
    cookie_name, presented_secret = credential
    chain = await SessionChains.get(claims['sid'])
    if chain is None:
        raise HTTPException(403, UNAVAILABLE)
    now = int(time.time())
    if chain.revoked_at is not None:
        raise HTTPException(401, REVOKED)
    frozen = (chain.user_id, chain.session_stamp, chain.auth_method, chain.auth_time, chain.mfa_verified)
    presented = (
        claims['id'],
        claims['session_stamp'],
        claims['auth_method'],
        claims['auth_time'],
        claims['mfa_verified'],
    )
    if frozen != presented:
        log.warning('Session refresh rejected: lineage does not match the token of user %s.', claims['id'])
        raise HTTPException(403, UNAVAILABLE)

    secret_hash = digest(presented_secret)
    if not _matches(secret_hash, chain.current_secret):
        if _matches(secret_hash, chain.previous_secret):
            if now - chain.rotated_at <= REUSE_GRACE_SECONDS:
                raise HTTPException(409, CONFLICT)
            await revoke_lineage(claims['sid'], now, reason='replay')
            raise HTTPException(401, REVOKED)
        raise HTTPException(403, UNAVAILABLE)

    jti_hash = digest(claims['jti'])
    if not _matches(jti_hash, chain.current_jti):
        if _matches(jti_hash, chain.previous_jti) and now - chain.rotated_at <= REUSE_GRACE_SECONDS:
            raise HTTPException(409, CONFLICT)
        raise HTTPException(403, STALE)

    wait = chain.rotated_at + max(MIN_REFRESH_INTERVAL_SECONDS, lifetime // 2) - now
    if wait > 0:
        raise HTTPException(429, TOO_SOON, headers={'Retry-After': str(wait)})

    new_exp = min(now + lifetime, chain.absolute_expires_at)
    if new_exp <= claims['exp']:
        return {'refreshed': False, 'reason': 'max_lifetime', 'expires_at': claims['exp'], 'server_time': now}

    new_token = auth_utils.create_token(
        {
            'id': chain.user_id,
            'typ': 'session',
            'sid': claims['sid'],
            'auth_method': chain.auth_method,
            'auth_time': chain.auth_time,
            'session_stamp': chain.session_stamp,
            'mfa_verified': chain.mfa_verified,
            'exp': new_exp,
        }
    )
    new_jti = auth_utils.decode_token(new_token)['jti']
    new_secret = secrets.token_urlsafe(32)
    rotated = await SessionChains.rotate(
        claims['sid'],
        user_id=chain.user_id,
        session_stamp=chain.session_stamp,
        old_secret=chain.current_secret,
        old_jti=chain.current_jti,
        new_secret=digest(new_secret),
        new_jti=digest(new_jti),
        now=now,
    )
    if not rotated:
        # Lost a race, or the account/lineage changed meanwhile: nothing was issued.
        raise HTTPException(409, CONFLICT)

    max_age = new_exp - now
    # Same forward-only rule as identity lookups: never replace a newer login's cookie.
    if await may_write_token_cookie(request, new_token):
        response.set_cookie(
            'token',
            new_token,
            httponly=True,
            secure=WEBUI_AUTH_COOKIE_SECURE,
            samesite=WEBUI_AUTH_COOKIE_SAME_SITE,
            max_age=max_age,
        )
    set_refresh_cookie(response, claims['sid'], new_secret, max_age)
    if cookie_name == LEGACY_REFRESH_COOKIE:
        clear_refresh_cookie(response, LEGACY_REFRESH_COOKIE)
    # Only extend this login's own OAuth session cookie, never replace another login's.
    if chain.oauth_session_id and request.cookies.get('oauth_session_id') in (None, chain.oauth_session_id):
        oauth_session = await OAuthSessions.get_session_by_id(chain.oauth_session_id)
        if oauth_session and oauth_session.user_id == chain.user_id:
            response.set_cookie(
                'oauth_session_id',
                oauth_session.id,
                httponly=True,
                secure=WEBUI_AUTH_COOKIE_SECURE,
                samesite=WEBUI_AUTH_COOKIE_SAME_SITE,
                max_age=max_age,
            )
    log.info('Session renewed for user %s (lineage %s, generation %s).', chain.user_id, chain.id[:12], chain.generation + 1)
    return {
        'token': new_token,
        'token_type': 'Bearer',
        'expires_at': new_exp,
        'server_time': now,
        'refreshed': True,
    }


async def revoke_for_signout(request: Request, token: str | None, *, explicit: bool = False) -> list[str]:
    """End the lineage named by a signed (possibly expired) session token; return refresh cookies to clear.

    `explicit` means the caller sent an Authorization header: only that login is ended and only its own
    refresh cookie is cleared. Without any token the browser signs out of every lineage it can prove
    possession of with the refresh cookies it sent."""
    now = int(time.time())
    credentials = browser_refresh_credentials(request)
    token_sid = None
    if token and not token.startswith('sk-'):
        claims = auth_utils.decode_token_for_revocation(token)
        if isinstance(claims, dict) and claims.get('typ', 'session') == 'session' and _str(claims.get('id')):
            token_sid = login_id(claims)
            if token_sid:
                chain = await SessionChains.get(token_sid)
                if chain and chain.user_id == claims['id'] and chain.revoked_at is None:
                    await revoke_lineage(token_sid, now, reason='sign-out')
    if token_sid is not None:
        # This login's refresh cookie goes, and so do cookies of logins that are already over (revoked,
        # past their absolute expiry, unknown); live cookies of other logins in this browser stay.
        names = []
        for sid, (name, _) in credentials.items():
            if sid == token_sid:
                names.append(name)
                continue
            chain = await SessionChains.get(sid)
            if chain is None or chain.revoked_at is not None or chain.absolute_expires_at <= now:
                names.append(name)
        return names
    if explicit or token:
        return []  # A credential that names no lineage never falls back to the browser's cookies.
    names = []
    for sid, (name, secret) in credentials.items():
        chain = await SessionChains.get(sid)
        if chain and chain.revoked_at is None and _holds(chain, secret):
            await revoke_lineage(sid, now, reason='sign-out')
        names.append(name)
    return names


def signout_bearer(auth_header: str) -> str:
    """The credential of a sent Authorization header, validated before sign-out has any side effect."""
    scheme, separator, credential = auth_header.partition(' ')
    if (
        not separator
        or scheme.lower() != 'bearer'
        or not credential
        or credential != credential.strip()
        or ' ' in credential
    ):
        raise HTTPException(400, 'Invalid Authorization header.')
    if not credential.startswith('sk-') and auth_utils.decode_token_for_revocation(credential) is None:
        raise HTTPException(401, ERROR_MESSAGES.INVALID_TOKEN)
    return credential


async def browser_cookies_belong_to(request: Request, token: str | None, explicit: bool) -> bool:
    """Whether sign-out may clear the browser's shared cookies (token, OAuth session, server session).

    Yes without an explicit credential, or when the cookie is this same login, absent, or no longer valid.
    No when the caller named one login while the cookie holds another live login of this browser."""
    if not explicit:
        return True
    cookie_token = request.cookies.get('token')
    if not cookie_token or cookie_token == token:
        return True
    target = auth_utils.decode_token_for_revocation(token) if token and not token.startswith('sk-') else None
    cookie = auth_utils.decode_token_for_revocation(cookie_token)
    if isinstance(target, dict) and isinstance(cookie, dict) and login_id(target) == login_id(cookie):
        return True
    live = auth_utils.decode_token(cookie_token)
    if not isinstance(live, dict) or not _str(live.get('id')):
        return True
    try:
        return not await auth_utils.is_valid_token(live, getattr(request.app.state, 'redis', None))
    except HTTPException as error:
        return error.status_code == 401
    except Exception:
        return False


async def complete_oauth_handoff(request: Request, response: Response):
    """Consume the one-time OAuth hand-over cookie: (user, token, claims) of the login it carries."""
    await require_first_party_origin(request)
    raw = request.cookies.get(HANDOFF_COOKIE)
    if not raw:
        raise HTTPException(404, 'No pending sign-in.')
    clear_oauth_handoff(response)
    decoded = auth_utils.decode_token(raw)
    if (
        not isinstance(decoded, dict)
        or decoded.get('typ', 'session') != 'session'
        or not _str(decoded.get('id'))
        or not await auth_utils.is_valid_token(decoded, getattr(request.app.state, 'redis', None))
    ):
        raise HTTPException(401, ERROR_MESSAGES.INVALID_TOKEN)
    issued_at = _int(decoded.get('iat'))
    if issued_at is None or int(time.time()) - issued_at > HANDOFF_MAX_AGE + CLOCK_SKEW_SECONDS:
        raise HTTPException(401, 'Sign-in expired. Please sign in again.')
    user = await Users.get_user_by_id(decoded['id'])
    if user is None:
        raise HTTPException(401, ERROR_MESSAGES.INVALID_TOKEN)
    await supersede_browser_lineages(request, response, keep_sid=login_id(decoded))
    if await may_write_token_cookie(request, raw):
        response.set_cookie(
            'token',
            raw,
            httponly=True,
            secure=WEBUI_AUTH_COOKIE_SECURE,
            samesite=WEBUI_AUTH_COOKIE_SAME_SITE,
            max_age=max(0, (_int(decoded.get('exp')) or 0) - int(time.time())),
        )
    return user, raw, decoded


def _is_newer(presented: dict, cookie: dict) -> bool:
    """A later generation of the same login, or a login authenticated strictly later.

    Logins authenticated in the same second are not ordered, so neither replaces the other."""
    if (presented.get('sid') or presented.get('jti')) == (cookie.get('sid') or cookie.get('jti')):
        return (_int(presented.get('iat')) or 0) > (_int(cookie.get('iat')) or 0)
    presented_auth = _int(presented.get('auth_time')) or _int(presented.get('iat')) or 0
    cookie_auth = _int(cookie.get('auth_time')) or _int(cookie.get('iat')) or 0
    return presented_auth > cookie_auth


async def may_write_token_cookie(request: Request, token: str) -> bool:
    """The shared token cookie only ever moves forward.

    It is (re)written when absent, equal to `token`, confirmed invalid, or older than `token` - an earlier
    generation of the same login, or a login authenticated earlier. A newer login (e.g. the readable cookie
    an OAuth callback hands to the auth page) is never replaced by an older one, so a late response from an
    older login cannot take the cookie over for good: the next request with the newer Bearer moves it back.
    When validity cannot be determined (e.g. the database is unavailable) the cookie is left alone."""
    cookie_token = request.cookies.get('token')
    if not cookie_token or cookie_token == token:
        return True
    cookie = auth_utils.decode_token(cookie_token)
    if not isinstance(cookie, dict) or not _str(cookie.get('id')):
        return True
    try:
        if not await auth_utils.is_valid_token(cookie, getattr(request.app.state, 'redis', None)):
            return True
    except HTTPException as error:
        return error.status_code == 401  # 401: the cookie's account is gone or disabled.
    except Exception:
        return False
    presented = auth_utils.decode_token(token)
    return isinstance(presented, dict) and _is_newer(presented, cookie)
