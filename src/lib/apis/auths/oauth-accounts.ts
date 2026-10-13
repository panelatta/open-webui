import { WEBUI_API_BASE_URL } from '$lib/constants';

export const googleAccountRequest = async (
	action: '' | '/link' | '/unlink',
	token: string,
	body?: { password: string }
) => {
	const response = await fetch(`${WEBUI_API_BASE_URL}/auths/oauth/accounts/google${action}`, {
		method: action ? 'POST' : 'GET',
		credentials: 'same-origin',
		cache: 'no-store',
		headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
		...(body ? { body: JSON.stringify(body) } : {})
	});
	const result = await response.json();
	if (!response.ok) throw new Error(result.detail || 'Google account operation failed.');
	return result;
};
