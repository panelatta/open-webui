import { WEBUI_API_BASE_URL } from '$lib/constants';

export type ManagedChat = {
	id: string;
	title: string;
	updated_at: number;
	created_at: number;
	archived: boolean;
	pinned: boolean;
	folder_id: string | null;
};
export type BulkAction = 'move' | 'archive' | 'unarchive' | 'delete';
export type BulkResult = { id: string; success: boolean; error: string | null };

async function request(token: string, path: string, init: RequestInit = {}) {
	const response = await fetch(`${WEBUI_API_BASE_URL}/chats/${path}`, {
		...init,
		headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` }
	});
	const data = await response.json();
	if (!response.ok) {
		throw new Error(typeof data.detail === 'string' ? data.detail : response.statusText);
	}
	return data;
}

export async function getManagedChats(
	token: string,
	filter: { query: string; page: number; archived: string; folder: string }
): Promise<{ items: ManagedChat[]; total: number }> {
	const params = new URLSearchParams({ query: filter.query, page: String(filter.page) });
	if (filter.archived !== 'all') params.set('archived', filter.archived);
	if (filter.folder !== 'all') params.set('folder_id', filter.folder);
	return request(token, `manage?${params}`);
}

export async function bulkManageChats(
	token: string,
	ids: string[],
	action: BulkAction,
	folder_id: string | null = null
): Promise<BulkResult[]> {
	return request(token, 'bulk', {
		method: 'POST',
		body: JSON.stringify({ ids, action, folder_id })
	});
}
