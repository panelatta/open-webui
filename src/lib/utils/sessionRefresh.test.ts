import { describe, expect, it } from 'vitest';
import {
	decodeJwtClaims,
	expiryBuffer,
	networkRetryDelay,
	sameLogin,
	shouldRefresh
} from './sessionRefresh';

const encode = (value: unknown) =>
	Buffer.from(JSON.stringify(value)).toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
const jwt = (payload: Record<string, unknown>) => `${encode({ alg: 'HS256' })}.${encode(payload)}.sig`;

describe('session refresh claims', () => {
	it('reads first-generation and renewed tokens', () => {
		const first = decodeJwtClaims(jwt({ id: 'u1', jti: 'a', iat: 100, exp: 200 }));
		expect(first).toEqual({ id: 'u1', jti: 'a', chain: 'a', iat: 100, exp: 200 });
		const renewed = decodeJwtClaims(jwt({ id: 'u1', jti: 'b', sid: 'a', iat: 150, exp: 300 }));
		expect(renewed?.chain).toBe('a');
		expect(sameLogin(first, renewed)).toBe(true);
	});

	it('handles base64url payloads with non-ASCII text', () => {
		const claims = decodeJwtClaims(jwt({ id: '用户~?>', jti: 'j', iat: 1, exp: 2 }));
		expect(claims?.id).toBe('用户~?>');
	});

	it('rejects malformed tokens and mismatched logins', () => {
		for (const token of [null, '', 'a.b', 'a.@@@.c', `${encode({})}.${encode([1])}.x`, 'sk-abc']) {
			expect(decodeJwtClaims(token as string)).toBeNull();
		}
		const a = decodeJwtClaims(jwt({ id: 'u1', jti: 'a', iat: 1, exp: 2 }));
		expect(sameLogin(a, decodeJwtClaims(jwt({ id: 'u2', jti: 'x', sid: 'a', iat: 1, exp: 2 })))).toBe(false);
		expect(sameLogin(a, decodeJwtClaims(jwt({ id: 'u1', jti: 'c', iat: 1, exp: 2 })))).toBe(false);
		expect(sameLogin(a, null)).toBe(false);
		expect(sameLogin(decodeJwtClaims(jwt({ jti: 'a' })), decodeJwtClaims(jwt({ jti: 'a' })))).toBe(false);
	});
});

describe('session refresh timing', () => {
	const day = decodeJwtClaims(jwt({ id: 'u', jti: 'j', iat: 0, exp: 86400 }));

	it('waits for half the lifetime and recent activity', () => {
		expect(shouldRefresh({ claims: day, now: 43199, idleSeconds: 0 })).toBe(false);
		expect(shouldRefresh({ claims: day, now: 43200, idleSeconds: 0 })).toBe(true);
		expect(shouldRefresh({ claims: day, now: 43200, idleSeconds: 300 })).toBe(true);
		expect(shouldRefresh({ claims: day, now: 43200, idleSeconds: 301 })).toBe(false);
		expect(shouldRefresh({ claims: day, now: 43200, idleSeconds: -5 })).toBe(false);
	});

	it('stops inside the expiry buffer', () => {
		expect(expiryBuffer(day)).toBe(60);
		expect(shouldRefresh({ claims: day, now: 86339, idleSeconds: 0 })).toBe(true);
		expect(shouldRefresh({ claims: day, now: 86340, idleSeconds: 0 })).toBe(false);
	});

	it('scales the buffer so short lifetimes still have a renewal window', () => {
		const short = decodeJwtClaims(jwt({ id: 'u', jti: 'j', iat: 0, exp: 300 }));
		expect(expiryBuffer(short)).toBe(30);
		expect(shouldRefresh({ claims: short, now: 150, idleSeconds: 0 })).toBe(true);
		expect(shouldRefresh({ claims: short, now: 270, idleSeconds: 0 })).toBe(false);
		expect(expiryBuffer(decodeJwtClaims(jwt({ iat: 0, exp: 20 })))).toBe(5);
		expect(expiryBuffer(null)).toBe(60);
	});

	it('never renews incomplete tokens', () => {
		expect(shouldRefresh({ claims: decodeJwtClaims(jwt({ id: 'u', iat: 0, exp: 10 })), now: 6, idleSeconds: 0 })).toBe(false);
		expect(shouldRefresh({ claims: decodeJwtClaims(jwt({ id: 'u', jti: 'j', exp: 10 })), now: 6, idleSeconds: 0 })).toBe(false);
		expect(shouldRefresh({ claims: decodeJwtClaims(jwt({ id: 'u', jti: 'j', iat: 10, exp: 10 })), now: 10, idleSeconds: 0 })).toBe(false);
	});

	it('bounds network retries', () => {
		expect([0, 1, 2, 3, 4].map(networkRetryDelay)).toEqual([null, 5, 15, 30, null]);
	});
});
