<script lang="ts">
	import { getContext } from 'svelte';
	import { page } from '$app/stores';
	const i18n: any = getContext('i18n');
	$: result = $page.url.searchParams.get('result');
	$: message =
		result === 'linked'
			? 'Google account linked. Sign in again to continue.'
			: result === 'unlinked'
				? 'Google account unlinked. Sign in with your email and password.'
				: result === 'in_use'
					? 'This Google account is already linked. Unlink it from its current account first.'
					: 'Google account linking failed or expired. Return to account settings and try again.';
</script>

<div class="flex min-h-screen items-center justify-center p-6">
	<div class="max-w-md space-y-5 text-center">
		<h1 class="text-xl font-medium">{$i18n.t('Google account')}</h1>
		<p role="status">{$i18n.t(message)}</p>
		<a
			class="block rounded-lg bg-black px-4 py-2 text-white dark:bg-white dark:text-black"
			href="/auth?state=logout&form=signin">{$i18n.t('Sign in again')}</a
		>
		<a class="block text-sm text-gray-500" href="/">{$i18n.t('Back to OpenWebUI')}</a>
	</div>
</div>
