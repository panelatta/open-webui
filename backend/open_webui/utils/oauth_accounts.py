"""Explicit Google identity linking; never merge accounts by email."""
import logging
import secrets
import time
from urllib.parse import urlsplit

from fastapi import BackgroundTasks, HTTPException, Response
from fastapi.responses import RedirectResponse
from sqlalchemy import delete, update
from sqlalchemy.exc import IntegrityError

from open_webui.internal.db import get_async_db
from open_webui.models.auths import Auth, Auths
from open_webui.models.config import Config
from open_webui.models.oauth_sessions import OAuthSession
from open_webui.models.users import User, Users
from open_webui.utils.auth import decode_token, get_current_user, get_human_user, verify_password

log = logging.getLogger(__name__)
LINK_PREFIX = 'owui-link-'
LINK_SESSION = 'google_account_link'
LINK_SECONDS = 300


def require_recent_human(request, user):
    claims = getattr(request.state, 'claims', {})
    age = int(time.time()) - claims.get('auth_time', 0)
    if (
        getattr(request.state, 'auth_type', None) != 'jwt'
        or claims.get('typ', 'session') != 'session'
        or user.role not in {'user', 'admin'}
    ):
        raise HTTPException(403, 'A human session is required.')
    if age < -30 or age > LINK_SECONDS:
        raise HTTPException(403, 'reauthentication_required')
    return claims


async def require_same_origin(request):
    configured = await Config.get('webui.url')
    parsed = urlsplit(configured or str(request.base_url))
    if request.headers.get('origin') != f'{parsed.scheme}://{parsed.netloc}':
        raise HTTPException(403, 'Invalid request origin.')


async def start_link(request, user):
    await require_same_origin(request)
    claims = require_recent_human(request, user)
    # The callback must return to the same authenticated browser, not merely a bearer caller.
    cookie = decode_token(request.cookies.get('token', '')) or {}
    if not claims.get('jti') or cookie.get('jti') != claims['jti']:
        raise HTTPException(403, 'reauthentication_required')
    current = await Users.get_user_by_id(user.id)
    if (current.oauth or {}).get('google'):
        raise HTTPException(409, 'Unlink the current Google account first.')
    from open_webui.utils.oauth import get_oauth_runtime_config
    settings = await get_oauth_runtime_config()
    client = request.app.state.oauth_manager.get_client('google')
    if not settings.ENABLE_OAUTH or client is None:
        raise HTTPException(404, 'Google sign-in is not configured.')
    state = LINK_PREFIX + secrets.token_urlsafe(32)
    request.session[LINK_SESSION] = {
        'state': state, 'user_id': user.id, 'session_stamp': claims['session_stamp'],
        'jti': claims['jti'], 'expires_at': int(time.time()) + LINK_SECONDS,
    }
    redirect_uri = (client.server_metadata or {}).get('redirect_uri') or str(
        request.url_for('oauth_login_callback', provider='google')
    )
    redirect = await client.authorize_redirect(
        request, redirect_uri, state=state, prompt='select_account'
    )
    return {'url': redirect.headers['location']}


async def change_google_binding(user_id, expected_stamp, *, identity=None, password=None):
    """Rotate the account stamp and change the binding in one transaction.

    A unique DB index on Google's subject also covers normal OAuth signup, so
    concurrent callbacks cannot assign the same identity to different accounts.
    """
    try:
        async with get_async_db() as db:
            locked = await db.execute(
                update(Auth).where(Auth.id == user_id, Auth.session_stamp == expected_stamp, Auth.active == True)
                .values(session_stamp=Auth.session_stamp)
            )
            if locked.rowcount != 1:
                raise HTTPException(409, 'Authentication changed. Please sign in again.')
            auth = await db.get(Auth, user_id)
            user = await db.get(User, user_id)
            if user is None or user.role not in {'user', 'admin'}:
                raise HTTPException(403, 'Account is not active.')
            oauth = dict(user.oauth or {})
            if identity is not None:
                if oauth.get('google'):
                    raise HTTPException(409, 'Unlink the current Google account first.')
                owner = await Users.get_user_by_oauth_sub('google', identity['sub'], db=db)
                if owner and owner.id != user_id:
                    raise HTTPException(409, 'This Google account is already linked to another account.')
                oauth['google'] = identity
            else:
                # Prove a usable fallback before removing a login method.
                if not password or not await verify_password(password, auth.password):
                    raise HTTPException(400, 'Your current password is incorrect.')
                if 'google' not in oauth:
                    raise HTTPException(409, 'No Google account is linked.')
                oauth.pop('google')
            user.oauth = oauth
            await db.flush()
            await Auths.revoke_sessions_by_user_id(user_id, db=db)
            await db.execute(delete(OAuthSession).where(
                OAuthSession.user_id == user_id, OAuthSession.provider == 'google'
            ))
            await db.commit()
    except IntegrityError:
        raise HTTPException(409, 'This Google account is already linked to another account.') from None


async def disconnect_after_change(user_id):
    from open_webui.socket.main import disconnect_user_sessions
    try:
        await disconnect_user_sessions(user_id)
    except Exception:
        log.warning('Could not disconnect sockets after Google binding change for user %s', user_id)


async def finish_link(request):
    """Separate from sign-in: a failed/expired link never creates or switches users."""
    result = 'failed'
    try:
        state = request.query_params.get('state', '')
        intent = request.session.pop(LINK_SESSION, None)
        if not intent or intent.get('state') != state or intent.get('expires_at', 0) <= int(time.time()):
            raise HTTPException(400, 'Link request expired.')
        response = Response()
        user = await get_current_user(request, response, BackgroundTasks(), auth_token=None)
        await get_human_user(request, user)
        claims = require_recent_human(request, user)
        if (
            user.id != intent['user_id']
            or claims.get('session_stamp') != intent['session_stamp']
            or claims.get('jti') != intent['jti']
        ):
            raise HTTPException(403, 'The signed-in account changed.')
        from open_webui.utils.oauth import get_oauth_runtime_config
        settings = await get_oauth_runtime_config()
        client = request.app.state.oauth_manager.get_client('google')
        if not settings.ENABLE_OAUTH or client is None:
            raise HTTPException(404, 'Google sign-in is not configured.')
        token = await client.authorize_access_token(request)
        # Authlib validates the Google ID token's issuer, audience, signature and nonce.
        info = token.get('userinfo')
        if not info or info.get('email_verified') is not True or not info.get('sub') or not info.get('email'):
            raise HTTPException(400, 'A verified Google email is required.')
        email = info['email'].lower()
        if '*' not in settings.OAUTH_ALLOWED_DOMAINS and email.split('@')[-1] not in settings.OAUTH_ALLOWED_DOMAINS:
            raise HTTPException(403, 'Email domain is not allowed.')
        await change_google_binding(user.id, intent['session_stamp'], identity={'sub': str(info['sub']), 'email': email})
        result = 'linked'
        log.info('Google identity linked for user %s', user.id)
        await disconnect_after_change(user.id)
    except HTTPException as error:
        result = 'in_use' if error.detail == 'This Google account is already linked to another account.' else 'failed'
        log.info('Google linking rejected: HTTP %s', error.status_code)
    except Exception:
        # Do not expose OAuth codes, tokens, or provider responses in logs/URLs.
        log.warning('Google linking did not complete')
    return RedirectResponse('/auth/google-link?result=' + result, status_code=303,
                            headers={'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer'})
