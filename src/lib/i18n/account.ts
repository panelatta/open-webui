import { get, writable } from 'svelte/store';
import { toast } from 'svelte-sonner';
import { updateUserSettings } from '$lib/apis/users';
import i18n, { changeLanguage, getLanguages } from '$lib/i18n';

export const savingAccountLanguage = writable(false);

export const applyAccountLanguage = async (language: unknown) => {
	if (typeof language === 'string' && (await getLanguages()).some((l) => l.code === language)) {
		await changeLanguage(language);
	}
};

export const saveAccountLanguage = async (token: string, language: string) => {
	if (get(savingAccountLanguage)) return false;
	savingAccountLanguage.set(true);
	try {
		// A top-level patch preserves UI settings, including changes from other devices.
		const saved = await updateUserSettings(token, { language });
		if (saved?.language !== language) throw new Error('Language preference was not saved');
		await changeLanguage(language);
		return true;
	} catch {
		toast.error(get(i18n).t('Failed to save language preference. Please try again.'));
		return false;
	} finally {
		savingAccountLanguage.set(false);
	}
};
