"""Human-session endpoints for managing linked Google identities."""
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from open_webui.config import ENABLE_PASSWORD_AUTH
from open_webui.models.config import Config
from open_webui.models.users import Users
from open_webui.utils.auth import get_human_user
from open_webui.utils.oauth_accounts import (
    change_google_binding, disconnect_after_change, require_recent_human,
    require_same_origin, start_link,
)
from open_webui.utils.rate_limit import RateLimiter

router = APIRouter()
limiter = RateLimiter(limit=5, window=300)


class UnlinkForm(BaseModel):
    password: str = Field(min_length=1, max_length=1024)


@router.get('/google')
async def status(request: Request, user=Depends(get_human_user)):
    current = await Users.get_user_by_id(user.id)
    binding = (current.oauth or {}).get('google') or {}
    return {'linked': bool(binding.get('sub')), 'email': binding.get('email'),
            'password_email': current.email}


@router.post('/google/link')
async def link(request: Request, user=Depends(get_human_user)):
    return await start_link(request, user)


@router.post('/google/unlink')
async def unlink(request: Request, form: UnlinkForm, user=Depends(get_human_user)):
    await require_same_origin(request)
    claims = require_recent_human(request, user)
    if not ENABLE_PASSWORD_AUTH or not await Config.get('ui.enable_login_form'):
        raise HTTPException(400, 'Password sign-in must be enabled before unlinking Google.')
    if await limiter.is_limited(getattr(request.app.state, 'redis', None), 'google-unlink:' + user.id):
        raise HTTPException(429, 'Too many attempts. Please try again later.')
    await change_google_binding(user.id, claims['session_stamp'], password=form.password)
    await disconnect_after_change(user.id)
    return {'success': True}
