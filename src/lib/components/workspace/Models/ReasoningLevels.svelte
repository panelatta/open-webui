<script lang="ts">
	import { getContext } from 'svelte';
	import {
		reasoningFields,
		reasoningTemplate,
		reasoningLabel,
		type ReasoningConfig
	} from '$lib/utils/reasoning';
	const i18n = getContext<import('svelte/store').Readable<import('i18next').i18n>>('i18n');
	export let config: ReasoningConfig | null = null;
	export let inherited: ReasoningConfig | null = null;
	let locale = 'en-US';
	let errors: Record<number, string> = {};
	$: mode = config === null ? 'inherit' : config.enabled ? 'custom' : 'disabled';
	const template = (field: string) => {
		config = reasoningTemplate(field);
		errors = {};
	};
	const move = (index: number, offset: number) => {
		if (!config) return;
		const options = [...config.options];
		[options[index], options[index + offset]] = [options[index + offset], options[index]];
		config = { ...config, options };
		errors = {};
	};
	export function validate() {
		if (!config?.enabled) return true;
		const ids = config.options.map((o) => o.id);
		return (
			Object.keys(errors).length === 0 &&
			ids.length > 0 &&
			ids.length <= 32 &&
			ids.every((id) => /^[a-zA-Z0-9_-]{1,64}$/.test(id)) &&
			new Set(ids).size === ids.length &&
			(!config.default_id || ids.includes(config.default_id))
		);
	}
	function editValue(index: number, text: string) {
		if (!config) return;
		try {
			config.options[index].value = JSON.parse(text);
		} catch {
			config.options[index].value = text;
		}
		config = config;
	}
</script>

<section
	class="my-3 space-y-3 rounded-xl border border-gray-100 p-3 text-xs dark:border-gray-800"
	data-testid="reasoning-editor"
>
	<div class="flex flex-wrap items-center justify-between gap-2">
		<span class="font-medium">{$i18n.t('Thinking effort levels')}</span>
		<select
			class="rounded-lg bg-gray-50 p-2 dark:bg-gray-850"
			aria-label={$i18n.t('Thinking effort levels')}
			value={mode}
			on:change={(e) => {
				config =
					e.currentTarget.value === 'inherit'
						? null
						: e.currentTarget.value === 'disabled'
							? { ...reasoningTemplate(), enabled: false }
							: structuredClone(inherited ?? reasoningTemplate());
				if (e.currentTarget.value === 'custom' && config) config.enabled = true;
				errors = {};
			}}
		>
			<option value="inherit">{$i18n.t('Inherit from base model')}</option>
			<option value="custom">{$i18n.t('Custom')}</option>
			<option value="disabled">{$i18n.t('Disabled')}</option>
		</select>
	</div>
	<p class="text-gray-500">
		{$i18n.t(
			'Configure chat effort choices for the actual upstream connection. Model default keeps existing advanced parameters.'
		)}
	</p>
	{#if config?.enabled}
		<div class="flex flex-wrap gap-2">
			{#each [['OpenAI', 'reasoning_effort'], ['OpenRouter', 'reasoning.effort'], ['Claude effort', 'output_config.effort'], ['Claude budget', 'thinking.budget_tokens']] as [label, field]}
				<button
					type="button"
					class="rounded-lg bg-gray-100 px-2 py-2 dark:bg-gray-800"
					on:click={() => template(field)}>{$i18n.t(label)}</button
				>
			{/each}
		</div>
		<div class="grid gap-3 sm:grid-cols-2">
			<label
				>{$i18n.t('Request field')}<select
					class="mt-1 w-full rounded-lg bg-gray-50 p-2 dark:bg-gray-850"
					bind:value={config.field}
				>
					{#each reasoningFields as field}<option value={field}>{field}</option>{/each}
				</select></label
			>
			<label
				>{$i18n.t('Default level')}<select
					class="mt-1 w-full rounded-lg bg-gray-50 p-2 dark:bg-gray-850"
					bind:value={config.default_id}
				>
					<option value="">{$i18n.t('Model default')}</option>
					{#each config.options as option}<option value={option.id}
							>{reasoningLabel(option, $i18n.language)}</option
						>{/each}
				</select></label
			>
		</div>
		<label class="block"
			>{$i18n.t('Label language (locale code)')}
			<input
				class="mt-1 w-full rounded-lg bg-gray-50 p-2 dark:bg-gray-850"
				bind:value={locale}
				placeholder="en-US / zh-CN / ja-JP"
				list="reasoning-locales"
			/>
			<datalist id="reasoning-locales"
				><option value="en-US"></option><option value="zh-CN"></option><option value="zh-TW"
				></option><option value="ja-JP"></option><option value="fr-FR"></option></datalist
			>
		</label>
		{#each config.options as option, index}
			<div class="space-y-2 rounded-lg bg-gray-50 p-2 dark:bg-gray-850">
				<div class="grid grid-cols-1 gap-2 sm:grid-cols-3">
					<label
						>{$i18n.t('Level ID')}<input
							class="mt-1 w-full rounded-lg bg-white p-2 dark:bg-gray-900"
							required
							pattern={'[a-zA-Z0-9_\\-]{1,64}'}
							bind:value={option.id}
						/></label
					>
					<label
						>{$i18n.t('Translated label')} · {locale}<input
							class="mt-1 w-full rounded-lg bg-white p-2 dark:bg-gray-900"
							maxlength="120"
							value={option.labels[locale] ?? ''}
							placeholder={option.id}
							on:input={(e) => {
								if (locale.trim()) {
									if (e.currentTarget.value.trim())
										option.labels[locale.trim()] = e.currentTarget.value;
									else delete option.labels[locale.trim()];
									config = config;
								}
							}}
						/></label
					>
					<label
						>{$i18n.t('Request value (JSON or text)')}<input
							class="mt-1 w-full rounded-lg bg-white p-2 dark:bg-gray-900"
							required
							value={typeof option.value === 'string' ? option.value : JSON.stringify(option.value)}
							on:change={(e) => editValue(index, e.currentTarget.value)}
						/></label
					>
				</div>
				<details>
					<summary class="cursor-pointer py-1"
						>{$i18n.t('Additional field bindings (JSON)')}</summary
					>
					<textarea
						class="mt-1 w-full rounded-lg bg-white p-2 font-mono dark:bg-gray-900"
						aria-label={$i18n.t('Additional field bindings (JSON)')}
						rows="2"
						value={JSON.stringify(option.bindings, null, 2)}
						on:input={(e) => {
							try {
								const value = JSON.parse(e.currentTarget.value);
								if (
									!value ||
									Array.isArray(value) ||
									typeof value !== 'object' ||
									Object.keys(value).some(
										(key) => !reasoningFields.includes(key) || key === config?.field
									)
								)
									throw Error();
								option.bindings = value;
								delete errors[index];
								e.currentTarget.setCustomValidity('');
							} catch {
								errors[index] = $i18n.t('Invalid reasoning configuration');
								e.currentTarget.setCustomValidity(errors[index]);
							}
							errors = errors;
							config = config;
						}}
					></textarea>
					<p class="text-gray-500">
						{$i18n.t(
							'Use dotted fields, for example: {"thinking.type":"enabled"}. A null value removes a field.'
						)}
					</p>
				</details>
				{#if errors[index]}<p class="text-red-500" role="alert">{errors[index]}</p>{/if}
				<div class="flex justify-end gap-2">
					<button
						class="rounded p-2 disabled:opacity-30"
						type="button"
						disabled={index === 0}
						aria-label={$i18n.t('Move up')}
						on:click={() => move(index, -1)}>↑</button
					>
					<button
						class="rounded p-2 disabled:opacity-30"
						type="button"
						disabled={index === config.options.length - 1}
						aria-label={$i18n.t('Move down')}
						on:click={() => move(index, 1)}>↓</button
					>
					<button
						class="rounded p-2"
						type="button"
						on:click={() => {
							if (!config) return;
							config.options = config.options.filter((_, i) => i !== index);
							if (config.default_id === option.id) config.default_id = '';
							config = config;
							errors = {};
						}}>{$i18n.t('Delete')}</button
					>
				</div>
			</div>
		{/each}
		<button
			type="button"
			class="rounded-lg bg-gray-100 px-3 py-2 dark:bg-gray-800"
			disabled={config.options.length >= 32}
			on:click={() => {
				if (!config) return;
				config.options = [
					...config.options,
					{ id: `level-${Date.now()}`, labels: {}, value: 'high', bindings: {} }
				];
				config = config;
			}}>{$i18n.t('Add effort level')}</button
		>
		<details>
			<summary class="cursor-pointer py-2">{$i18n.t('Request mapping preview')}</summary>
			<pre class="overflow-x-auto text-xs">{JSON.stringify(
					config.options.map((o) => ({
						id: o.id,
						fields: { [config!.field]: o.value, ...o.bindings }
					})),
					null,
					2
				)}</pre>
		</details>
	{/if}
</section>
