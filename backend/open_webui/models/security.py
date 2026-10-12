"""Durable session revocation and ownership of opaque provider/cache identifiers."""

import hashlib
import time

from open_webui.internal.db import Base, get_async_db_context
from pydantic import BaseModel, ConfigDict
from sqlalchemy import BigInteger, Column, String, Text, delete, select
from sqlalchemy.exc import IntegrityError


class RevokedSession(Base):
    __tablename__ = "revoked_session"
    id = Column(String(64), primary_key=True)
    expires_at = Column(BigInteger, nullable=True, index=True)


class SessionRevocations:
    @staticmethod
    async def revoke(jti, expires_at=None, db=None):
        key = hashlib.sha256(jti.encode()).hexdigest()
        async with get_async_db_context(db) as session:
            await session.execute(
                delete(RevokedSession).where(
                    RevokedSession.expires_at <= int(time.time())
                )
            )
            if not await session.get(RevokedSession, key):
                session.add(RevokedSession(id=key, expires_at=expires_at))
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()  # Concurrent sign-outs are idempotent.

    @staticmethod
    async def is_revoked(jti, db=None):
        async with get_async_db_context(db) as session:
            row = await session.get(
                RevokedSession, hashlib.sha256(jti.encode()).hexdigest()
            )
            return row is not None and (
                row.expires_at is None or row.expires_at > int(time.time())
            )


class SecurityArtifact(Base):
    __tablename__ = "security_artifact"
    id = Column(String(64), primary_key=True)
    kind = Column(String(32), nullable=False)
    scope = Column(String(64), nullable=False)
    external_id = Column(Text, nullable=False)
    user_id = Column(Text, nullable=False, index=True)
    local_file_id = Column(Text, nullable=True)
    chat_id = Column(Text, nullable=True)
    model_id = Column(Text, nullable=True)
    created_at = Column(BigInteger, nullable=False)


class SecurityArtifactModel(BaseModel):
    user_id: str
    local_file_id: str | None = None
    chat_id: str | None = None
    model_id: str | None = None
    model_config = ConfigDict(from_attributes=True)


class SecurityArtifacts:
    @staticmethod
    def key(kind, scope, external_id):
        return hashlib.sha256(f"{kind}\0{scope}\0{external_id}".encode()).hexdigest()

    @classmethod
    async def get(cls, kind, scope, external_id, db=None):
        async with get_async_db_context(db) as session:
            row = await session.get(SecurityArtifact, cls.key(kind, scope, external_id))
            return SecurityArtifactModel.model_validate(row) if row else None

    @classmethod
    async def record(
        cls,
        kind,
        scope,
        external_id,
        user_id,
        *,
        local_file_id=None,
        chat_id=None,
        model_id=None,
        db=None,
    ):
        from fastapi import HTTPException

        if (
            not isinstance(external_id, str)
            or not external_id
            or len(external_id) > 2048
        ):
            raise HTTPException(502, "Invalid artifact identifier from provider.")
        key = cls.key(kind, scope, external_id)
        async with get_async_db_context(db) as session:
            row = await session.get(SecurityArtifact, key)
            if row is None:
                row = SecurityArtifact(
                    id=key,
                    kind=kind,
                    scope=scope,
                    external_id=external_id,
                    user_id=user_id,
                    local_file_id=local_file_id,
                    chat_id=chat_id,
                    model_id=model_id,
                    created_at=int(time.time()),
                )
                session.add(row)
                try:
                    await session.commit()
                except IntegrityError:
                    await session.rollback()
                    row = await session.get(SecurityArtifact, key)
            # Never allow a collision or a shared upstream key to transfer ownership.
            if (
                row is None
                or row.user_id != user_id
                or row.local_file_id != local_file_id
            ):
                raise HTTPException(403, "Artifact ownership conflict.")
