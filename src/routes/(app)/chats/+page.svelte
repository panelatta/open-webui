<script lang="ts">
	import { getContext, onMount, onDestroy } from 'svelte';
	import { beforeNavigate } from '$app/navigation';
	import { toast } from 'svelte-sonner';
	import { config, mobile, showSidebar, user, WEBUI_NAME } from '$lib/stores';
	import { refreshSidebar } from '$lib/stores/chatList';
	import { getFolders } from '$lib/apis/folders';
	import {
		bulkManageChats,
		getManagedChats,
		type BulkAction,
		type ManagedChat
	} from '$lib/apis/chats/manage';
	import ConfirmDialog from '$lib/components/common/ConfirmDialog.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import SidebarIcon from '$lib/components/icons/Sidebar.svelte';

	const i18n: any = getContext('i18n');
	let items: ManagedChat[] = [];
	let folderList: { id: string; name: string; parent_id: string | null }[] = [];
	let selected = new Set<string>();
	let query = '';
	let archived = 'false';
	let folder = 'all';
	let targetFolder = '';
	let page = 1;
	let total = 0;
	let loading = true;
	let busy = false;
	let error = '';
	let folderError = '';
	let notice = '';
	let failures: { title: string; error: string }[] = [];
	let showConfirm = false;
	let pendingAction: BulkAction = 'delete';
	let requestId = 0;
	let timer: ReturnType<typeof setTimeout>;
	let disposed = false;

	$: canDelete = $user?.role === 'admin' || ($user?.permissions?.chat?.delete ?? true);
	$: canMove =
		($config?.features as { enable_folders?: boolean } | undefined)?.enable_folders &&
		($user?.role === 'admin' || ($user?.permissions?.features?.folders ?? true));
	$: pages = Math.max(1, Math.ceil(total / 50));
	$: allSelected = items.length > 0 && items.every((item) => selected.has(item.id));
	$: actionLabel =
		pendingAction === 'move'
			? $i18n.t('Move')
			: pendingAction === 'archive'
				? $i18n.t('Archive')
				: pendingAction === 'unarchive'
					? $i18n.t('Unarchive')
					: $i18n.t('Delete');

	function folderName(id: string | null): string {
		if (!id) return $i18n.t('No folder');
		const parts: string[] = [];
		const seen = new Set<string>();
		let current = folderList.find((item) => item.id === id);
		while (current && !seen.has(current.id)) {
			seen.add(current.id);
			parts.unshift(current.name);
			current = folderList.find((item) => item.id === current?.parent_id);
		}
		return parts.join(' / ') || $i18n.t('Folder');
	}

	async function load(keep = new Set<string>()) {
		const id = ++requestId;
		loading = true;
		error = '';
		selected = new Set();
		try {
			const result = await getManagedChats(localStorage.token, { query, archived, folder, page });
			if (disposed || id !== requestId) return;
			total = result.total;
			if (page > Math.max(1, Math.ceil(total / 50))) {
				page = Math.max(1, Math.ceil(total / 50));
				await load(keep);
				return;
			}
			items = result.items;
			selected = new Set(items.filter((item) => keep.has(item.id)).map((item) => item.id));
		} catch (e) {
			if (disposed || id !== requestId) return;
			items = [];
			error = String(e instanceof Error ? e.message : e);
		} finally {
			if (!disposed && id === requestId) loading = false;
		}
	}

	function filterChanged(debounce = false) {
		clearTimeout(timer);
		requestId++;
		page = 1;
		selected = new Set();
		loading = true;
		notice = '';
		failures = [];
		if (debounce) timer = setTimeout(() => load(), 250);
		else load();
	}

	function toggle(id: string) {
		const next = new Set(selected);
		if (next.has(id)) next.delete(id);
		else next.add(id);
		selected = next;
	}

	function confirm(action: BulkAction) {
		if (busy || loading || !selected.size) return;
		pendingAction = action;
		showConfirm = true;
	}

	async function run() {
		if (busy || !selected.size) return;
		const ids = [...selected];
		const titles = new Map(items.map((item) => [item.id, item.title]));
		busy = true;
		notice = '';
		failures = [];
		try {
			const results = await bulkManageChats(
				localStorage.token,
				ids,
				pendingAction,
				targetFolder || null
			);
			const failed = results.filter((item) => !item.success);
			notice = $i18n.t('chatManager.result', {
				succeeded: results.length - failed.length,
				failed: failed.length
			});
			failures = failed.map((item) => ({
				title: titles.get(item.id) || item.id,
				error: item.error || $i18n.t('Something went wrong')
			}));
			await load(new Set(failed.map((item) => item.id)));
			await refreshSidebar(localStorage.token).catch(() =>
				toast.error($i18n.t('chatManager.refreshFailed'))
			);
		} catch (e) {
			notice = $i18n.t('chatManager.uncertain');
			failures = [{ title: actionLabel, error: String(e instanceof Error ? e.message : e) }];
			await load();
			await refreshSidebar(localStorage.token).catch(() => {});
		} finally {
			busy = false;
		}
	}

	beforeNavigate((navigation) => {
		if (busy) {
			navigation.cancel();
			if (!navigation.willUnload) toast.info($i18n.t('chatManager.wait'));
		}
	});

	onMount(() => {
		if ($mobile) showSidebar.set(false);
		load();
		if (canMove) {
			getFolders(localStorage.token)
				.then((value) => {
					if (!disposed) folderList = value;
				})
				.catch((e) => {
					if (!disposed) folderError = String(e);
				});
		}
	});
	onDestroy(() => {
		disposed = true;
		requestId++;
		clearTimeout(timer);
	});
</script>

<svelte:head><title>{$i18n.t('Manage Chats')} · {$WEBUI_NAME}</title></svelte:head>

<ConfirmDialog
	bind:show={showConfirm}
	title={$i18n.t('chatManager.confirmTitle', { action: actionLabel, count: selected.size })}
	confirmLabel={actionLabel}
	on:confirm={run}
>
	<div class="space-y-3 text-sm text-gray-600 dark:text-gray-300">
		<p>{$i18n.t('chatManager.onlySelected', { count: selected.size })}</p>
		{#if pendingAction === 'delete'}
			<p class="text-red-600 dark:text-red-400">{$i18n.t('chatManager.deleteWarning')}</p>
		{:else if pendingAction === 'move'}
			<p>{$i18n.t('chatManager.destination', { folder: folderName(targetFolder || null) })}</p>
			<p>{$i18n.t('chatManager.moveWarning')}</p>
		{:else if pendingAction === 'archive'}
			<p>{$i18n.t('chatManager.archiveWarning')}</p>
		{:else if pendingAction === 'unarchive'}
			<p>{$i18n.t('chatManager.unarchiveWarning')}</p>
		{/if}
	</div>
</ConfirmDialog>

<div
	class="flex h-screen max-h-[100dvh] w-full min-w-0 flex-col {$showSidebar
		? 'md:max-w-[calc(100%-var(--sidebar-width))]'
		: ''}"
>
	<header class="flex items-center gap-3 border-b border-gray-100 dark:border-gray-850 px-4 py-3">
		<button
			class="rounded-lg p-1.5 hover:bg-gray-100 dark:hover:bg-gray-850"
			aria-label={$i18n.t('Toggle Sidebar')}
			on:click={() => showSidebar.set(!$showSidebar)}
		>
			<SidebarIcon className="size-5" />
		</button>
		<h1 class="text-lg font-semibold">{$i18n.t('Manage Chats')}</h1>
	</header>
	<section
		data-testid="chat-manager"
		aria-label={$i18n.t('Manage Chats')}
		class="flex-1 overflow-y-auto p-4 sm:p-6"
	>
		<div class="mx-auto max-w-6xl space-y-5">
			<p class="text-sm text-gray-500 dark:text-gray-400">{$i18n.t('chatManager.description')}</p>
			<div class="flex flex-wrap gap-3">
				<label class="min-w-48 flex-1">
					<span class="mb-1 block text-xs text-gray-500">{$i18n.t('Search')}</span>
					<input
						type="search"
						maxlength="500"
						bind:value={query}
						on:input={() => filterChanged(true)}
						disabled={busy}
						placeholder={$i18n.t('chatManager.search')}
						class="w-full rounded-xl border border-gray-200 bg-transparent px-3 py-2 text-sm dark:border-gray-700"
					/>
				</label>
				<label>
					<span class="mb-1 block text-xs text-gray-500">{$i18n.t('Status')}</span>
					<select
						aria-label={$i18n.t('Status')}
						bind:value={archived}
						on:change={() => filterChanged()}
						disabled={busy}
						class="rounded-xl border border-gray-200 bg-white pl-3 pr-8 py-2 text-sm dark:border-gray-700 dark:bg-gray-900"
					>
						<option value="false">{$i18n.t('chatManager.unarchived')}</option>
						<option value="true">{$i18n.t('Archived')}</option>
						<option value="all">{$i18n.t('All')}</option>
					</select>
				</label>
				{#if canMove}
					<label class="max-w-full">
						<span class="mb-1 block text-xs text-gray-500">{$i18n.t('Folder')}</span>
						<select
							aria-label={$i18n.t('Folder')}
							bind:value={folder}
							on:change={() => filterChanged()}
							disabled={busy || !!folderError}
							class="max-w-full rounded-xl border border-gray-200 bg-white pl-3 pr-8 py-2 text-sm dark:border-gray-700 dark:bg-gray-900"
						>
							<option value="all">{$i18n.t('All folders')}</option>
							<option value="">{$i18n.t('No folder')}</option>
							{#each folderList as entry}<option value={entry.id}>{folderName(entry.id)}</option
								>{/each}
						</select>
					</label>
				{/if}
				<button
					class="self-end rounded-xl border border-gray-200 px-3 py-2 text-sm disabled:opacity-40 dark:border-gray-700"
					on:click={() => load()}
					disabled={busy || loading}>{$i18n.t('Refresh')}</button
				>
			</div>
			{#if folderError}<p role="alert" class="text-sm text-red-600">{folderError}</p>{/if}
			<div
				class="sticky top-0 z-10 flex flex-wrap items-center gap-2 rounded-2xl bg-gray-50 p-3 dark:bg-gray-900"
				aria-busy={busy}
			>
				<span class="mr-auto text-sm font-medium" aria-live="polite"
					>{$i18n.t('chatManager.selected', { count: selected.size })}</span
				>
				{#if busy}<Spinner className="size-4" /><span class="text-sm"
						>{$i18n.t('chatManager.wait')}</span
					>{/if}
				{#if canMove}
					<select
						aria-label={$i18n.t('chatManager.moveTo')}
						bind:value={targetFolder}
						disabled={busy || !!folderError}
						class="max-w-full rounded-lg border border-gray-200 bg-white pl-2 pr-8 py-1.5 text-sm dark:border-gray-700 dark:bg-gray-850"
					>
						<option value="">{$i18n.t('No folder')}</option>
						{#each folderList as entry}<option value={entry.id}>{folderName(entry.id)}</option
							>{/each}
					</select>
					<button
						class="action"
						disabled={busy || loading || !selected.size || !!folderError}
						on:click={() => confirm('move')}>{$i18n.t('Move')}</button
					>
				{/if}
				<button
					class="action"
					disabled={busy || loading || !selected.size}
					on:click={() => confirm('archive')}>{$i18n.t('Archive')}</button
				>
				<button
					class="action"
					disabled={busy || loading || !selected.size}
					on:click={() => confirm('unarchive')}>{$i18n.t('Unarchive')}</button
				>
				{#if canDelete}<button
						class="action text-red-600 dark:text-red-400"
						disabled={busy || loading || !selected.size}
						on:click={() => confirm('delete')}>{$i18n.t('Delete')}</button
					>{/if}
			</div>
			{#if notice}
				<div
					role="status"
					class="rounded-xl border border-gray-200 p-3 text-sm dark:border-gray-700"
				>
					<p>{notice}</p>
					{#if failures.length}<ul class="mt-2 space-y-1 text-red-600 dark:text-red-400">
							{#each failures as failure}<li class="break-words">
									{failure.title}: {failure.error}
								</li>{/each}
						</ul>{/if}
				</div>
			{/if}
			{#if error}
				<div role="alert" class="rounded-xl border border-red-200 p-4 text-sm text-red-600">
					{error} <button class="ml-2 underline" on:click={() => load()}>{$i18n.t('Retry')}</button>
				</div>
			{:else if loading}
				<div class="flex justify-center gap-2 p-12" role="status">
					<Spinner className="size-5" />{$i18n.t('Loading...')}
				</div>
			{:else}
				<div class="overflow-x-auto rounded-2xl border border-gray-200 dark:border-gray-800">
					<table class="w-full text-left text-sm">
						<thead class="bg-gray-50 text-xs text-gray-500 dark:bg-gray-900">
							<tr>
								<th class="w-10 p-3"
									><input
										type="checkbox"
										checked={allSelected}
										indeterminate={selected.size > 0 && !allSelected}
										disabled={busy || !items.length}
										aria-label={$i18n.t('chatManager.selectPage')}
										on:change={() =>
											(selected = allSelected ? new Set() : new Set(items.map((item) => item.id)))}
									/></th
								>
								<th class="p-3">{$i18n.t('Title')}</th><th class="hidden p-3 sm:table-cell"
									>{$i18n.t('Folder')}</th
								><th class="hidden whitespace-nowrap p-3 md:table-cell">{$i18n.t('Updated at')}</th>
							</tr>
						</thead>
						<tbody>
							{#each items as item (item.id)}
								<tr
									class="border-t border-gray-100 dark:border-gray-800 {selected.has(item.id)
										? 'bg-blue-50/60 dark:bg-blue-950/20'
										: ''}"
								>
									<td class="p-3"
										><input
											type="checkbox"
											checked={selected.has(item.id)}
											disabled={busy}
											aria-label={$i18n.t('chatManager.selectChat', { title: item.title })}
											on:change={() => toggle(item.id)}
										/></td
									>
									<td class="max-w-xs p-3">
										<a
											href="/c/{item.id}"
											class="line-clamp-2 break-words font-medium hover:underline"
											>{item.title || $i18n.t('New Chat')}</a
										>
										<div class="mt-1 flex flex-wrap gap-2 text-xs text-gray-500">
											{#if item.archived}<span>{$i18n.t('Archived')}</span>{/if}
											{#if item.pinned}<span>{$i18n.t('Pinned')}</span>{/if}
											<span class="sm:hidden">{folderName(item.folder_id)}</span>
										</div>
									</td>
									<td class="hidden max-w-48 break-words p-3 text-gray-500 sm:table-cell"
										>{folderName(item.folder_id)}</td
									>
									<td class="hidden whitespace-nowrap p-3 text-gray-500 md:table-cell"
										>{new Date(item.updated_at * 1000).toLocaleString($i18n.language)}</td
									>
								</tr>
							{:else}<tr
									><td colspan="4" class="p-12 text-center text-gray-500"
										>{$i18n.t('No results found')}</td
									></tr
								>{/each}
						</tbody>
					</table>
				</div>
			{/if}
			<div class="flex flex-wrap items-center justify-between gap-3 text-sm text-gray-500">
				<p>{$i18n.t('chatManager.page', { page, pages, total })}</p>
				<div class="flex gap-2">
					<button
						class="action"
						disabled={busy || loading || page <= 1}
						on:click={() => {
							page--;
							load();
						}}>{$i18n.t('Previous')}</button
					>
					<button
						class="action"
						disabled={busy || loading || page >= pages}
						on:click={() => {
							page++;
							load();
						}}>{$i18n.t('Next')}</button
					>
				</div>
			</div>
			<p class="text-xs text-gray-500">{$i18n.t('chatManager.selectionHint')}</p>
		</div>
	</section>
</div>

<style>
	.action {
		border: 1px solid color-mix(in srgb, currentColor 20%, transparent);
		border-radius: 0.65rem;
		padding: 0.4rem 0.75rem;
		font-size: 0.875rem;
	}
	.action:hover:not(:disabled) {
		background: color-mix(in srgb, currentColor 7%, transparent);
	}
	.action:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}
	input[type='checkbox'] {
		width: 1rem;
		height: 1rem;
		accent-color: #2563eb;
	}
</style>
