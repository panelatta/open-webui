# Multi-user security hardening

This fork retains Open WebUI 0.12.0 functionality while enforcing ownership of
hosted attachments, Responses identifiers and cached media.

## Authorization

- Hosted uploads check the authenticated user's file ownership or current read
  grants before opening, uploading, reusing or updating the file. Both normal
  chat completions and the direct OpenAI-compatible endpoint use this check.
- Provider response, item, conversation, container and file IDs have a durable ownership
  registry. The registry is scoped to provider URL and credential/config identity,
  not a mutable connection index. Unknown or foreign references fail closed.
  Hosted file references recheck current local file grants, including revocation.
  Hosted tool file/container references are checked too; unknown vector-store
  references are rejected (this deployment keeps non-memory embeddings disabled).
  Response references can be delegated through their original live chat ACL,
  never through an untrusted client-supplied chat ID or a copied share snapshot.
- Streaming records returned IDs before dispatching complete SSE events. JSON
  responses, native Responses requests and background retrieval/resume use the
  same checks. Disabled provider connections cannot be selected by route index.
- Legacy unregistered upstream state is not guessed or assigned to a caller.
  Saved local chat history remains available; starting a new request with that
  history obtains a fresh registered response. Existing hosted files are
  registered when safely uploaded/reused through their authenticated local file.
- Audio output names are per-user. The cache endpoint checks ownership, resolves
  symlinks before checking containment, and denies unregistered legacy cache
  objects to non-admins. Generated images already use the normal file ACL.
- Individual JWT sign-outs are stored in the database, independent of Redis and
  worker restarts. Other sessions remain valid; legacy JWTs without a session ID
  use account-wide revocation. Password-change/session-stamp revocation remains.

The additive migration creates security_artifact and revoked_session.
It does not change stored chat, model, memory or uploaded-file contents.

## Local production policy

Use a durable secret key. Configure exact CORS origins, secure session/auth
cookies, a loopback-only published port and a read-only backend source mount.
Set STATIC_DIR=/app/backend/data/static for writable branding assets.

Enable password validation with a minimum of 12 characters; bcrypt's existing
72-byte maximum remains. Empty/whitespace passwords are always rejected.
Signup has process-local/Redis-compatible limits of 5 attempts per source per
hour and 20 total per hour. Trust forwarded IP headers only from the local proxy.
The global limit also bounds attempts with spoofed source addresses.

scripts/harden-local-security.py sets a 24-hour JWT lifetime, hides admin
contact details, preserves pending registration, and removes server-code
authoring rights from group membership. Admins retain Tools and Functions.
With --activate-mfa, it invalidates existing sessions and requires authenticator
enrollment at the next login. It never creates or records the user's MFA secret
or recovery codes. Keep recovery codes after enrollment.

Registration remains open with manual approval for personally known users.
Email delivery/verification is deliberately not configured; the user chose this
scope after discussing email providers. The approval UI explicitly says that
password-signup email ownership is unverified and identity needs separate
confirmation through a trusted channel.

## Regression tests

Run the existing backend suites plus test_multi_user_security.py. These cover
file ownership/sharing/revocation, native references, scope changes, SSE ownership,
session revocation and cache traversal/isolation.

scripts/test-multi-user-security.py is a real HTTP/Socket.IO regression harness.
It refuses to run without the documented synthetic test key and /tmp/security-*
data directory. Run it against a fresh v0.12.0 container with this backend,
--network none, an empty temporary database, RAG_EMBEDDING_ENGINE=openai,
both provider APIs initially disabled, password validation enabled, secure
cookies, and CORS_ALLOW_ORIGIN=https://openwebui.panelatta.de. Enable background
resume only in this test container with model allowlist audit-public.
The suite runs a mock provider on loopback and never invokes embeddings.

Before deployment, also verify the local frontend build, browser attachment
uploads with process=false, real hosted file reading/stream completion and
memory-only embedding. Compare database integrity/counts and vector collections
before/after. Activate production MFA after smoke tests; enrollment then belongs
to the human account owner.

The deployment environment/Compose changes are reproducible with
scripts/configure-local-security.py. Pass the existing deployment directory,
public HTTPS origin, Git build hash, published port and actual Docker proxy
gateway. Back up .env and docker-compose.yml first. The script retains other
settings and secrets, adds only the security settings, and keeps explicit
localhost origins for local access. Copy existing package static assets to the
data/static directory before starting with the read-only backend mount.
Run harden-local-security.py in the container with PYTHONPATH=/app/backend.
