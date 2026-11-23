"""Durable session revocation and ownership of opaque provider/cache identifiers."""

import hashlib
import time

from open_webui.internal.db import Base, get_async_db_context
from pydantic import BaseModel, ConfigDict
from sqlalchemy import (
    BigInteger,
    Boolean,
    case,
    Column,
    Integer,
    String,
    Text,
    delete,
    exists,
    or_,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class RevokedSession(Base):
    __tablename__ = "revoked_session"
    id = Column(String(64), primary_key=True)
    expires_at = Column(BigInteger, nullable=True, index=True)


class SessionRevocations:
    @staticmethod
    async def revoke(jti, expires_at=None, db=None):
        """Revoke an identifier; an existing revocation is only ever extended (None = permanent)."""
        key = digest(jti)
        async with get_async_db_context(db) as session:
            dialect = session.get_bind().dialect.name
        if dialect in {"sqlite", "postgresql"}:
            await SessionRevocations._upsert(key, expires_at, dialect, db)
            return
        for _ in range(3):
            async with get_async_db_context(db) as session:
                try:
                    await session.execute(
                        delete(RevokedSession).where(
                            RevokedSession.expires_at <= int(time.time())
                        )
                    )
                    extend = update(RevokedSession).where(RevokedSession.id == key)
                    if expires_at is not None:
                        extend = extend.where(
                            RevokedSession.expires_at.is_not(None),
                            RevokedSession.expires_at < expires_at,
                        )
                    await session.execute(extend.values(expires_at=expires_at))
                    present = await session.execute(
                        select(RevokedSession.id).where(RevokedSession.id == key)
                    )
                    if present.scalar_one_or_none() is None:
                        session.add(RevokedSession(id=key, expires_at=expires_at))
                    await session.commit()
                    return
                except IntegrityError:
                    # A concurrent sign-out inserted the row first; retry as an extension.
                    await session.rollback()
        raise RuntimeError("Session revocation could not be persisted.")

    @staticmethod
    async def _upsert(key, expires_at, dialect, db=None):
        """One atomic statement: insert, or keep the longer expiry (NULL = permanent wins)."""
        if dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert
        else:
            from sqlalchemy.dialects.sqlite import insert

        statement = insert(RevokedSession).values(id=key, expires_at=expires_at)
        incoming = statement.excluded.expires_at
        statement = statement.on_conflict_do_update(
            index_elements=[RevokedSession.id],
            set_={
                "expires_at": case(
                    (
                        or_(
                            RevokedSession.expires_at.is_(None),
                            incoming.is_(None),
                        ),
                        None,
                    ),
                    (incoming > RevokedSession.expires_at, incoming),
                    else_=RevokedSession.expires_at,
                )
            },
        )
        async with get_async_db_context(db) as session:
            await session.execute(
                delete(RevokedSession).where(
                    RevokedSession.expires_at <= int(time.time())
                )
            )
            await session.execute(statement)
            await session.commit()

    @staticmethod
    async def is_revoked(jti, db=None):
        return await SessionRevocations.any_revoked([jti], db=db)

    @staticmethod
    async def any_revoked(identifiers, db=None):
        keys = {digest(value) for value in identifiers if value}
        if not keys:
            return False
        async with get_async_db_context(db) as session:
            row = await session.execute(
                select(RevokedSession.id)
                .where(
                    RevokedSession.id.in_(keys),
                    or_(
                        RevokedSession.expires_at.is_(None),
                        RevokedSession.expires_at > int(time.time()),
                    ),
                )
                .limit(1)
            )
            return row.first() is not None


class SessionLineage(Base):
    """One first-party interactive login whose access tokens may be renewed by a rotating refresh secret."""

    __tablename__ = "session_lineage"
    id = Column(String(64), primary_key=True)  # sha256(sid); sid is the first access token's jti
    user_id = Column(Text, nullable=False, index=True)
    session_stamp = Column(Text, nullable=False)
    auth_method = Column(String(32), nullable=False)
    auth_time = Column(BigInteger, nullable=False)
    mfa_verified = Column(Boolean, nullable=False)
    absolute_expires_at = Column(BigInteger, nullable=False, index=True)
    generation = Column(Integer, nullable=False)
    current_secret = Column(String(64), nullable=False)
    previous_secret = Column(String(64), nullable=True)
    current_jti = Column(String(64), nullable=False)
    previous_jti = Column(String(64), nullable=True)
    rotated_at = Column(BigInteger, nullable=False)
    oauth_session_id = Column(Text, nullable=True)
    revoked_at = Column(BigInteger, nullable=True)


class SessionLineageModel(BaseModel):
    id: str
    user_id: str
    session_stamp: str
    auth_method: str
    auth_time: int
    mfa_verified: bool
    absolute_expires_at: int
    generation: int
    current_secret: str
    previous_secret: str | None = None
    current_jti: str
    previous_jti: str | None = None
    rotated_at: int
    oauth_session_id: str | None = None
    revoked_at: int | None = None
    model_config = ConfigDict(from_attributes=True)


class SessionChains:
    """Persistence for session lineages. Every call uses its own short transaction."""

    CLEANUP_BATCH = 100

    @staticmethod
    async def create(
        sid,
        *,
        user_id,
        session_stamp,
        auth_method,
        auth_time,
        mfa_verified,
        absolute_expires_at,
        secret_hash,
        jti_hash,
        now,
        oauth_session_id=None,
    ):
        async with get_async_db_context() as session:
            expired = await session.execute(
                select(SessionLineage.id)
                .where(SessionLineage.absolute_expires_at < now)
                .limit(SessionChains.CLEANUP_BATCH)
            )
            expired_ids = list(expired.scalars())
            if expired_ids:
                await session.execute(
                    delete(SessionLineage).where(SessionLineage.id.in_(expired_ids))
                )
            session.add(
                SessionLineage(
                    id=digest(sid),
                    user_id=user_id,
                    session_stamp=session_stamp,
                    auth_method=auth_method,
                    auth_time=auth_time,
                    mfa_verified=mfa_verified,
                    absolute_expires_at=absolute_expires_at,
                    generation=0,
                    current_secret=secret_hash,
                    previous_secret=None,
                    current_jti=jti_hash,
                    previous_jti=None,
                    rotated_at=now,
                    oauth_session_id=oauth_session_id,
                    revoked_at=None,
                )
            )
            await session.commit()

    @staticmethod
    async def get(sid) -> SessionLineageModel | None:
        async with get_async_db_context() as session:
            row = await session.execute(
                select(SessionLineage).where(SessionLineage.id == digest(sid))
            )
            row = row.scalar_one_or_none()
            return SessionLineageModel.model_validate(row) if row else None

    @staticmethod
    async def rotate(
        sid,
        *,
        user_id,
        session_stamp,
        old_secret,
        old_jti,
        new_secret,
        new_jti,
        now,
    ) -> bool:
        """Advance the lineage only if it is still at the presented generation and the account is unchanged."""
        from open_webui.models.auths import Auth

        account_unchanged = exists().where(
            Auth.id == user_id,
            Auth.session_stamp == session_stamp,
            Auth.active == True,  # noqa: E712
        )
        async with get_async_db_context() as session:
            result = await session.execute(
                update(SessionLineage)
                .where(
                    SessionLineage.id == digest(sid),
                    SessionLineage.user_id == user_id,
                    SessionLineage.session_stamp == session_stamp,
                    SessionLineage.current_secret == old_secret,
                    SessionLineage.current_jti == old_jti,
                    SessionLineage.revoked_at.is_(None),
                    SessionLineage.absolute_expires_at > now,
                    account_unchanged,
                )
                .values(
                    current_secret=new_secret,
                    previous_secret=old_secret,
                    current_jti=new_jti,
                    previous_jti=old_jti,
                    generation=SessionLineage.generation + 1,
                    rotated_at=now,
                )
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                await session.rollback()
                return False
            await session.commit()
            return True

    @staticmethod
    async def mark_revoked(sid, now) -> SessionLineageModel | None:
        async with get_async_db_context() as session:
            await session.execute(
                update(SessionLineage)
                .where(
                    SessionLineage.id == digest(sid),
                    SessionLineage.revoked_at.is_(None),
                )
                .values(revoked_at=now)
                .execution_options(synchronize_session=False)
            )
            await session.commit()
            row = await session.execute(
                select(SessionLineage).where(SessionLineage.id == digest(sid))
            )
            row = row.scalar_one_or_none()
            return SessionLineageModel.model_validate(row) if row else None

    @staticmethod
    async def unexpired_user_ids(now) -> list[str]:
        """Users with any lineage not past its absolute expiry, revoked ones included: code that
        predates lineages only checks `jti`, so all of them must be ended before a rollback."""
        async with get_async_db_context() as session:
            rows = await session.execute(
                select(SessionLineage.user_id)
                .where(SessionLineage.absolute_expires_at > now)
                .distinct()
            )
            return list(rows.scalars())


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


class AuthHandoff(Base):
    """A pending OAuth sign-in hand-over: consumed exactly once by the browser that completed the callback."""

    __tablename__ = "auth_handoff"
    id = Column(String(64), primary_key=True)  # sha256(ticket)
    user_id = Column(Text, nullable=False)
    auth_time = Column(BigInteger, nullable=False)
    token = Column(Text, nullable=False)  # Fernet ciphertext of the session token
    oauth_session_id = Column(Text, nullable=True)
    id_token = Column(Text, nullable=True)  # Fernet ciphertext of the IdP id_token, if kept as a cookie
    expires_at = Column(BigInteger, nullable=False, index=True)


class AuthHandoffs:
    CLEANUP_BATCH = 100

    @staticmethod
    async def create(
        ticket_hash,
        *,
        user_id,
        auth_time,
        token_ciphertext,
        expires_at,
        now,
        oauth_session_id=None,
        id_token_ciphertext=None,
    ):
        async with get_async_db_context() as session:
            expired = await session.execute(
                select(AuthHandoff.id)
                .where(AuthHandoff.expires_at <= now)
                .limit(AuthHandoffs.CLEANUP_BATCH)
            )
            expired_ids = list(expired.scalars())
            if expired_ids:
                await session.execute(
                    delete(AuthHandoff).where(AuthHandoff.id.in_(expired_ids))
                )
            session.add(
                AuthHandoff(
                    id=ticket_hash,
                    user_id=user_id,
                    auth_time=auth_time,
                    token=token_ciphertext,
                    oauth_session_id=oauth_session_id,
                    id_token=id_token_ciphertext,
                    expires_at=expires_at,
                )
            )
            await session.commit()

    @staticmethod
    async def consume(ticket_hash, now):
        """Return the pending hand-over once; None if unknown, expired or already used."""
        async with get_async_db_context() as session:
            row = await session.execute(
                select(
                    AuthHandoff.user_id,
                    AuthHandoff.auth_time,
                    AuthHandoff.token,
                    AuthHandoff.oauth_session_id,
                    AuthHandoff.id_token,
                    AuthHandoff.expires_at,
                ).where(AuthHandoff.id == ticket_hash)
            )
            row = row.first()
            if row is None:
                return None
            result = await session.execute(
                delete(AuthHandoff)
                .where(AuthHandoff.id == ticket_hash)
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                await session.rollback()
                return None  # A concurrent request consumed it first.
            await session.commit()
            if row.expires_at <= now:
                return None
            return row
