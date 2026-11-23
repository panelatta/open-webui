// Client-side timing helpers for sliding sessions. Claims read here are never trusted for
// authorization; they only decide *when* to ask the server and whether two tokens are the same login.

export type TokenClaims = {
	id: string | null;
	jti: string | null;
	chain: string | null; // lineage id: `sid` on renewed tokens, the first token's `jti` otherwise
	iat: number | null;
	exp: number | null;
};

export const ACTIVITY_WINDOW_SECONDS = 5 * 60;
// Retries after a network error stay well inside the server's 120 s reuse grace window.
export const NETWORK_RETRY_DELAYS_SECONDS = [5, 15, 30];
// localStorage key: the jti this browser must never try to renew again (shared by all tabs).
export const REFRESH_HALT_KEY = 'session-refresh-halt';
export const REFRESH_LOCK_NAME = 'open-webui-session-refresh';

const decodeBase64Url = (segment: string): string => {
	const base64 = segment.replace(/-/g, '+').replace(/_/g, '/');
	const padded = base64 + '='.repeat((4 - (base64.length % 4)) % 4);
	const binary = atob(padded);
	const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
	return new TextDecoder().decode(bytes);
};

const asNumber = (value: unknown): number | null =>
	typeof value === 'number' && Number.isFinite(value) ? value : null;

const asString = (value: unknown): string | null =>
	typeof value === 'string' && value.length > 0 ? value : null;

export const decodeJwtClaims = (token: string | null | undefined): TokenClaims | null => {
	if (typeof token !== 'string' || !token) {
		return null;
	}
	const parts = token.split('.');
	if (parts.length !== 3) {
		return null;
	}
	try {
		const payload = JSON.parse(decodeBase64Url(parts[1]));
		if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
			return null;
		}
		const jti = asString(payload.jti);
		return {
			id: asString(payload.id),
			jti,
			chain: asString(payload.sid) ?? jti,
			iat: asNumber(payload.iat),
			exp: asNumber(payload.exp)
		};
	} catch {
		return null;
	}
};

/** Same user and same login lineage (a renewed token of the session already shown). */
export const sameLogin = (a: TokenClaims | null, b: TokenClaims | null): boolean =>
	!!a && !!b && a.id !== null && a.id === b.id && a.chain !== null && a.chain === b.chain;

/** Seconds before `exp` at which the UI treats the session as over; scaled down for short lifetimes. */
export const expiryBuffer = (claims: TokenClaims | null): number => {
	if (!claims || claims.iat === null || claims.exp === null || claims.exp <= claims.iat) {
		return 60;
	}
	return Math.max(5, Math.min(60, Math.floor((claims.exp - claims.iat) * 0.1)));
};

/**
 * Renew only past half of the token's lifetime, while there is still time left, and only if the
 * person used this page recently (trusted pointer/keyboard input, not visibility or page load).
 */
export const shouldRefresh = ({
	claims,
	now,
	idleSeconds,
	activityWindowSeconds = ACTIVITY_WINDOW_SECONDS
}: {
	claims: TokenClaims | null;
	now: number;
	idleSeconds: number;
	activityWindowSeconds?: number;
}): boolean => {
	if (!claims || claims.jti === null || claims.iat === null || claims.exp === null) {
		return false;
	}
	if (claims.exp <= claims.iat) {
		return false;
	}
	const lifetime = claims.exp - claims.iat;
	return (
		claims.exp - now > expiryBuffer(claims) &&
		now - claims.iat >= lifetime / 2 &&
		idleSeconds >= 0 &&
		idleSeconds <= activityWindowSeconds
	);
};

/** Delay before retry number `failures` (1-based) after network errors, or null to give up. */
export const networkRetryDelay = (failures: number): number | null =>
	failures >= 1 && failures <= NETWORK_RETRY_DELAYS_SECONDS.length
		? NETWORK_RETRY_DELAYS_SECONDS[failures - 1]
		: null;
