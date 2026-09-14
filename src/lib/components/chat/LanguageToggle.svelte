<script lang="ts">
	import { getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import type { i18n as I18n } from 'i18next';
	import { changeLanguage } from '$lib/i18n';
	import Tooltip from '$lib/components/common/Tooltip.svelte';

	const i18n = getContext<Writable<I18n>>('i18n');

	$: isChinese = $i18n.language?.startsWith('zh');
	$: label = isChinese
		? $i18n.t('Switch interface language to English')
		: $i18n.t('Switch interface language to Simplified Chinese');
</script>

<Tooltip content={label} placement="bottom" touch={false}>
	<button
		id="chat-language-toggle"
		type="button"
		class="flex h-11 shrink-0 cursor-pointer items-center justify-center gap-1 rounded-lg px-2 text-xs text-gray-500 transition hover:bg-gray-50/40 hover:text-gray-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-400 dark:text-gray-400 dark:hover:bg-gray-800/40 dark:hover:text-gray-200 sm:h-8"
		aria-label={label}
		on:click={() => changeLanguage(isChinese ? 'en-US' : 'zh-CN')}
	>
		<span
			aria-hidden="true"
			class={isChinese ? 'font-semibold text-gray-800 dark:text-gray-100' : ''}>中</span
		>
		<span aria-hidden="true" class="text-gray-300 dark:text-gray-600">/</span>
		<span
			aria-hidden="true"
			class={!isChinese ? 'font-semibold text-gray-800 dark:text-gray-100' : ''}>EN</span
		>
	</button>
</Tooltip>

<style>
	@media (pointer: coarse) {
		button {
			min-height: 2.75rem;
		}
	}
</style>
