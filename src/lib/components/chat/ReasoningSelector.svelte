<script lang="ts">
	import { getContext } from 'svelte';
	import { models, type Model } from '$lib/stores';
	import type { ReasoningConfig } from '$lib/utils/reasoning';
	import { reasoningLabel, reasoningSelection } from '$lib/utils/reasoning';
	const i18n = getContext<import('svelte/store').Readable<import('i18next').i18n>>('i18n');
	export let selectedModelIds: string[] = [];
	export let params: any = {};
	$: available = selectedModelIds
		.map((id) => $models.find((m) => m.id === id))
		.filter((m): m is Model & { reasoning_effort_config: ReasoningConfig } =>
			Boolean(m?.reasoning_effort_config?.enabled && m.reasoning_effort_config.options?.length)
		);
</script>

{#if available.length}
	<div
		class="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 pb-1"
		data-testid="reasoning-selectors"
	>
		{#each available as model (model.id)}
			{@const config = model.reasoning_effort_config}
			<label
				class="reasoning-control flex min-w-0 max-w-full items-center gap-1 text-xs text-gray-500 dark:text-gray-400"
			>
				<span class="min-w-0 max-w-48 truncate"
					>{$i18n.t('Thinking effort')}{available.length > 1 ? ` · ${model.name}` : ''}</span
				>
				<select
					class="min-w-0 max-w-40 truncate rounded-lg border-0 bg-transparent py-1 pl-1 pr-5 text-xs text-gray-700 hover:bg-gray-100 focus-visible:outline focus-visible:outline-2 dark:text-gray-200 dark:hover:bg-gray-800"
					aria-label={`${$i18n.t('Thinking effort')} · ${model.name}`}
					value={reasoningSelection(config, params.reasoning_effort_levels?.[model.id])}
					on:change={(event) => {
						params = {
							...params,
							reasoning_effort_levels: {
								...params.reasoning_effort_levels,
								[model.id]: event.currentTarget.value
							}
						};
					}}
				>
					<option value="">{$i18n.t('Model default')}</option>
					{#each config.options as option (option.id)}
						<option value={option.id}>{reasoningLabel(option, $i18n.language)}</option>
					{/each}
				</select>
			</label>
		{/each}
	</div>
{/if}

<style>
	@media (pointer: coarse), (max-width: 639px) {
		.reasoning-control select {
			min-height: 44px;
		}
	}
</style>
