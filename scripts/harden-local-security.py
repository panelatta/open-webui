"""Apply local deployment security policy inside the running container.

Run once without --activate-mfa before smoke tests, then with --activate-mfa
after verification. The latter invalidates existing sessions; users enroll
their own authenticator at their next password login. No MFA secrets are created.
"""

import argparse
import asyncio
import copy
import uuid

from sqlalchemy import select
from open_webui.internal.db import get_async_db
from open_webui.models.auths import Auth
from open_webui.models.config import Config
from open_webui.models.groups import Group


async def main(activate_mfa=False):
    await Config.upsert(
        {
            "auth.jwt_expiry": "24h",
            "auth.admin.show": False,
            "ui.default_user_role": "pending",
        }
    )
    async with get_async_db() as db:
        groups = (await db.execute(select(Group))).scalars().all()
        for group in groups:
            permissions = copy.deepcopy(group.permissions or {})
            workspace = permissions.setdefault("workspace", {})
            # Administrators retain tool authoring; membership alone must not grant server code execution.
            workspace.update(tools=False, tools_import=False, tools_export=False)
            group.permissions = permissions
        await db.commit()
    if activate_mfa and not await Config.get("auth.mfa.enable"):
        async with get_async_db() as db:
            for auth in (await db.execute(select(Auth))).scalars().all():
                auth.session_stamp = str(uuid.uuid4())
            await db.commit()
        await Config.upsert(
            {
                "auth.mfa.enable": True,
                "auth.mfa.allow_oauth_bypass": False,
                "auth.mfa.allow_trusted_header_bypass": False,
            }
        )
    print(
        "Security policy applied; MFA required:",
        bool(await Config.get("auth.mfa.enable")),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activate-mfa", action="store_true")
    asyncio.run(main(parser.parse_args().activate_mfa))
