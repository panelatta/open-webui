import { describe, expect, it } from 'vitest';
import {
	reasoningLabel,
	reasoningRequest,
	reasoningSelection,
	reasoningTemplate
} from './reasoning';
describe('reasoning levels', () => {
	it('resolves exact, regional, English and ID label fallbacks', () => {
		const option = reasoningTemplate().options[0];
		expect(reasoningLabel(option, 'zh-CN')).toBe('轻量');
		expect(reasoningLabel(option, 'zh-SG')).toBe('轻量');
		expect(reasoningLabel(option, 'fr-FR')).toBe('Low');
		expect(reasoningLabel({ ...option, labels: {} }, 'ja')).toBe('low');
	});
	it('keeps model default distinct from configured default and recovers deleted IDs', () => {
		const config = { ...reasoningTemplate(), default_id: 'medium' };
		expect(reasoningSelection(config, '')).toBe('');
		expect(reasoningSelection(config, 'deleted')).toBe('medium');
		expect(reasoningSelection(config, 'high')).toBe('high');
	});
	it('sends only the current alias selection, never mappings or another model selection', () => {
		const model = { id: 'custom-alias', reasoning_effort_config: reasoningTemplate() };
		const params = { reasoning_effort_levels: { 'custom-alias': 'low', other: 'high' } };
		expect(reasoningRequest(model, params)).toEqual({ reasoning_effort_level: 'low' });
		expect(reasoningRequest({ ...model, id: 'other' }, params)).toEqual({
			reasoning_effort_level: 'high'
		});
		expect(reasoningRequest({ ...model, reasoning_effort_config: null }, params)).toEqual({});
	});
	it('preserves provider-specific choices without imposing a three-level ladder', () => {
		for (const levels of [['low', 'medium', 'high', 'xhigh', 'max'], ['low', 'high', 'max']]) {
			const config = {
				...reasoningTemplate(),
				options: levels.map((id) => ({ id, value: id, labels: {}, bindings: {} }))
			};
			const model = { id: 'alias', reasoning_effort_config: config };
			for (const level of levels) {
				expect(reasoningRequest(model, { reasoning_effort_levels: { alias: level } })).toEqual({
					reasoning_effort_level: level
				});
			}
		}
	});
});
