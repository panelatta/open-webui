<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { googleAccountRequest } from '$lib/apis/auths/oauth-accounts';
	const i18n: any = getContext('i18n');
	let status: { linked: boolean; email: string | null; password_email: string } | null = null;
	let error = '';
	let busy = false;
	let showUnlink = false;
	let password = '';
	let reauthenticate = false;
	const buttonClass =
		'text-xs text-gray-500 hover:text-gray-900 disabled:opacity-40 dark:hover:text-white';
	onMount(async () => {
		try {
			status = await googleAccountRequest('', localStorage.token);
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	});
	async function act(unlink: boolean) {
		if (busy) return;
		busy = true;
		error = '';
		try {
			const result = await googleAccountRequest(
				unlink ? '/unlink' : '/link',
				localStorage.token,
				unlink ? { password } : undefined
			);
			password = '';
			if (unlink) {
				localStorage.removeItem('token');
				window.location.assign('/auth/google-link?result=unlinked');
			} else {
				window.location.assign(result.url);
			}
		} catch (e) {
			const message = e instanceof Error ? e.message : String(e);
			reauthenticate = message === 'reauthentication_required';
			error = reauthenticate
				? $i18n.t('Sign in again before managing linked accounts.')
				: $i18n.t(message);
		} finally {
			busy = false;
			password = '';
		}
	}
</script>

<div class="space-y-2.5 text-xs">
	{#if status}
		<p class="text-gray-600 dark:text-gray-400">
			{$i18n.t('Password sign-in email')}: {status.password_email}
		</p>
		<div class="flex items-center justify-between gap-3">
			<span
				>{status.linked
					? status.email || $i18n.t('Google account linked')
					: $i18n.t('No Google account linked')}</span
			>
			{#if status.linked}
				<button
					type="button"
					class={buttonClass}
					disabled={busy}
					on:click={() => (showUnlink = !showUnlink)}
				>
					{$i18n.t('Unlink Google account')}
				</button>
			{:else}
				<button type="button" class={buttonClass} disabled={busy} on:click={() => act(false)}>
					{$i18n.t('Link Google account')}
				</button>
			{/if}
		</div>
		<p class="text-[0.6875rem] text-gray-500">
			{$i18n.t(
				'Use a different Google email with this account. Linking does not change your password sign-in email.'
			)}
		</p>
		<p class="text-[0.6875rem] text-gray-500">
			{$i18n.t('Linking or unlinking signs out all devices. Sign in again to continue.')}
		</p>
		{#if showUnlink}
			<div class="space-y-2">
				<p>{$i18n.t('Confirm your current password so you can still sign in after unlinking.')}</p>
				<label class="block">
					{$i18n.t('Current Password')}
					<input
						type="password"
						autocomplete="current-password"
						bind:value={password}
						class="mt-1 w-full rounded-lg bg-gray-50 p-2 dark:bg-gray-900"
					/>
				</label>
				<button
					type="button"
					class={buttonClass}
					disabled={busy || !password}
					on:click={() => act(true)}
				>
					{$i18n.t('Confirm unlink')}
				</button>
				<button
					type="button"
					class={buttonClass}
					on:click={() => {
						showUnlink = false;
						password = '';
					}}
				>
					{$i18n.t('Cancel')}
				</button>
			</div>
		{/if}
	{/if}
	{#if error}<p role="alert" class="text-xs text-red-500">{error}</p>{/if}
	{#if reauthenticate}
		<a class={buttonClass} href="/auth?state=logout&form=signin">{$i18n.t('Sign in again')}</a>
	{/if}
</div>
