import { beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from 'svelte/store';
import { toast } from 'svelte-sonner';
import { updateUserSettings } from '$lib/apis/users';
import { changeLanguage } from '$lib/i18n';
import { applyAccountLanguage, saveAccountLanguage, savingAccountLanguage } from './account';

vi.mock('$lib/apis/users', () => ({ updateUserSettings: vi.fn() }));
vi.mock('svelte-sonner', () => ({ toast: { error: vi.fn() } }));
vi.mock('$lib/i18n', async () => {
	const { writable } = await import('svelte/store');
	return {
		default: writable({ t: (key: string) => key }),
		changeLanguage: vi.fn(),
		getLanguages: async () => [{ code: 'en-US' }, { code: 'zh-CN' }, { code: 'fr-FR' }]
	};
});

beforeEach(() => {
	vi.resetAllMocks();
	savingAccountLanguage.set(false);
});

describe('account language preference', () => {
	it('applies the account preference when entering the app without writing settings', async () => {
		await applyAccountLanguage('zh-CN');
		expect(changeLanguage).toHaveBeenCalledWith('zh-CN');
		expect(updateUserSettings).not.toHaveBeenCalled();
	});

	it('keeps the browser fallback for accounts without a supported preference', async () => {
		for (const value of [undefined, null, '', {}, 'invalid-locale']) {
			await applyAccountLanguage(value);
		}
		expect(changeLanguage).not.toHaveBeenCalled();
	});

	it('patches only the account language, preserving unrelated settings', async () => {
		vi.mocked(updateUserSettings).mockResolvedValue({
			language: 'fr-FR',
			ui: { system: 'Keep me' }
		});
		expect(await saveAccountLanguage('session-token', 'fr-FR')).toBe(true);
		expect(updateUserSettings).toHaveBeenCalledWith('session-token', { language: 'fr-FR' });
		expect(changeLanguage).toHaveBeenCalledWith('fr-FR');
		expect(get(savingAccountLanguage)).toBe(false);
	});

	it.each([null, { language: 'en-US' }])(
		'does not switch if saving is not confirmed: %j',
		async (response) => {
			vi.mocked(updateUserSettings).mockResolvedValue(response);
			expect(await saveAccountLanguage('session-token', 'zh-CN')).toBe(false);
			expect(changeLanguage).not.toHaveBeenCalled();
			expect(toast.error).toHaveBeenCalled();
			expect(get(savingAccountLanguage)).toBe(false);
		}
	);

	it('reports denied or failed requests and keeps the current language', async () => {
		vi.mocked(updateUserSettings).mockRejectedValue('Access prohibited');
		expect(await saveAccountLanguage('session-token', 'zh-CN')).toBe(false);
		expect(changeLanguage).not.toHaveBeenCalled();
		expect(toast.error).toHaveBeenCalled();
		expect(get(savingAccountLanguage)).toBe(false);
	});

	it('waits for persistence and prevents overlapping changes from both controls', async () => {
		let complete: (value: object) => void;
		vi.mocked(updateUserSettings).mockReturnValue(new Promise((resolve) => (complete = resolve)));
		const pending = saveAccountLanguage('session-token', 'zh-CN');
		expect(get(savingAccountLanguage)).toBe(true);
		expect(changeLanguage).not.toHaveBeenCalled();
		expect(await saveAccountLanguage('session-token', 'en-US')).toBe(false);
		expect(updateUserSettings).toHaveBeenCalledTimes(1);
		complete!({ language: 'zh-CN' });
		expect(await pending).toBe(true);
		expect(get(savingAccountLanguage)).toBe(false);
	});
});
