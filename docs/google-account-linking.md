# Google sign-in and explicit account linking

Settings → Account → Google account lets an approved user attach a Google identity
to their existing account without changing their password sign-in email. A local
email and a different Google email can therefore authenticate the same user ID,
with the same conversations and permissions. Start from the existing account's
settings; using the ordinary Google sign-in button first creates a separate
pending account when the identity is not already linked.

## Authentication and linking

- Google signup and password signup/login are enabled together. New users still
  require administrator approval. Automatic merging by email remains disabled.
- Native MFA remains enabled for password sign-in. The OAuth bypass is enabled;
  currently Google is the only OAuth provider. Google may enforce its own MFA.
- Linking requires a human session authenticated within five minutes, a
  same-origin POST, and a matching browser session cookie. Google authorization
  uses an explicit account chooser, a nonce and Authlib's signed-ID-token checks.
- The callback is bound to the initiating user, JWT ID and account session stamp.
  Expired, cancelled, mismatched or replayed link callbacks never fall through
  into ordinary signup. Verified Google email and configured domain restrictions
  are checked. A failed link does not switch accounts.
- A unique database index on Google's subject prevents one Google identity from
  belonging to multiple users, including concurrent linking and normal signup.
- Linking and unlinking atomically rotate the account's session stamp, remove
  its Google OAuth sessions and disconnect sockets. Sign in again afterwards.
- Unlinking also verifies the current local password with rate limiting, and
  requires password login to be enabled. A Google-only user without a known
  local password must first arrange a password through the administrator.
- Keep OAUTH_UPDATE_EMAIL_ON_LOGIN=false to retain the local login email.

## Deployment

Run scripts/configure-google-login.py with --deployment-dir, --credentials and
--origin (the HTTPS Cloudflare Tunnel public origin). It writes a mode-0600
google-oauth.env and references it from Docker Compose. Do not commit that file,
the credential JSON, or production database values. Back up the deployment
configuration and database before applying changes.

Persist webui.url, ui.enable_signup=true, ui.enable_login_form=true,
auth.mfa.enable=true and auth.mfa.allow_oauth_bypass=true. Apply MFA policy changes
with session revocation, preserving authenticator secrets and recovery codes.

Migration 62fbe170a903 adds uq_user_google_sub. Existing duplicate Google
subjects cause migration failure instead of silently merging users. SQLite and
PostgreSQL expressions normalize subjects to text. No user, chat or MFA data is
rewritten by this migration.

Build the entire frontend locally with Node 22, 8 GiB RAM, existing 8 GiB swap
and BUILD_MAX_OLD_SPACE_MB=6144 npm run build:safe. Mount the resulting build and
backend source, retaining the v0.12.0 runtime image and all fork customizations.

## Validation

The account-link tests cover distinct emails, identity uniqueness, stale stamps,
password verification, expired/mismatched OAuth state, wrong users, verified
Google email, domain restrictions, and password/OAuth MFA policy separation.
Run them together with embedding/file-policy and multi-user security regressions.
An actual Google account authorization remains a user-driven final check.
