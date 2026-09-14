export type ReasoningLevel = {
	id: string;
	labels: Record<string, string>;
	value: string | number | boolean | null;
	bindings: Record<string, string | number | boolean | null>;
};
export type ReasoningConfig = {
	enabled: boolean;
	field: string;
	default_id: string;
	options: ReasoningLevel[];
};
export const reasoningFields = [
	'reasoning_effort',
	'reasoning.effort',
	'reasoning.max_tokens',
	'reasoning.enabled',
	'reasoning.exclude',
	'thinking.type',
	'thinking.budget_tokens',
	'output_config.effort'
];
export function reasoningLabel(level: ReasoningLevel, language: string): string {
	const labels = level.labels ?? {};
	const locale = language.toLowerCase();
	const entries = Object.entries(labels).filter(([, value]) => value?.trim());
	return (
		entries.find(([key]) => key.toLowerCase() === locale)?.[1] ??
		entries.find(([key]) => key.toLowerCase() === locale.split('-')[0])?.[1] ??
		entries.find(([key]) => key.toLowerCase().split('-')[0] === locale.split('-')[0])?.[1] ??
		labels['en-US'] ??
		labels.en ??
		level.id
	);
}
export function reasoningSelection(config: ReasoningConfig, selected?: string): string {
	return selected === '' || config.options.some((o) => o.id === selected)
		? selected!
		: config.default_id;
}
export function reasoningRequest(model: any, params: any) {
	const config: ReasoningConfig = model?.reasoning_effort_config;
	if (!config?.enabled || !config.options?.length) return {};
	return {
		reasoning_effort_level: reasoningSelection(config, params?.reasoning_effort_levels?.[model.id])
	};
}
export function reasoningTemplate(field = 'reasoning_effort'): ReasoningConfig {
	const budget = field === 'thinking.budget_tokens';
	return {
		enabled: true,
		field,
		default_id: '',
		options: ['low', 'medium', 'high'].map(
			(id, i): ReasoningLevel => ({
				id,
				labels: { 'en-US': ['Low', 'Medium', 'High'][i], 'zh-CN': ['轻量', '标准', '深入'][i] },
				value: budget ? [1024, 4096, 8192][i] : id,
				bindings: budget ? { 'thinking.type': 'enabled' } : {}
			})
		)
	};
}
