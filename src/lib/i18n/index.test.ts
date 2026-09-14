import { beforeAll, describe, expect, it } from 'vitest';
import i18next from 'i18next';
import { initI18n } from './index';

beforeAll(async () => {
	await initI18n('zh-CN');
});

describe('switching from a cached Chinese locale', () => {
	it.each(['en-US', 'en-GB'])('renders English interface labels for %s', async (locale) => {
		await i18next.changeLanguage('zh-CN');
		expect(i18next.t('Language')).toBe('语言');
		await i18next.changeLanguage(locale);
		expect(i18next.t('Language')).toBe('Language');
		expect(i18next.t('Settings')).toBe('Settings');
		expect(i18next.t('Save')).toBe('Save');
	});
});
