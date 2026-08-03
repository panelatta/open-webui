import { describe, expect, it } from 'vitest';
import { parseGeneratedTitle } from './title';

describe('parseGeneratedTitle', () => {
	it('parses the expected JSON response', () => {
		expect(
			parseGeneratedTitle({
				choices: [{ message: { content: '{"title":"EMR 日志接入"}' } }]
			})
		).toBe('EMR 日志接入');
	});

	it('accepts JSON returned in reasoning_content', () => {
		expect(
			parseGeneratedTitle({
				choices: [{ message: { content: '', reasoning_content: '{"title":"任务标题"}' } }]
			})
		).toBe('任务标题');
	});

	it('accepts a short plain-text title from compatible providers', () => {
		expect(
			parseGeneratedTitle({
				choices: [{ message: { content: 'EMR 新架构接入' } }]
			})
		).toBe('EMR 新架构接入');
	});

	it('rejects empty or explanatory responses', () => {
		expect(parseGeneratedTitle({ choices: [{ message: { content: '' } }] })).toBeNull();
		expect(
			parseGeneratedTitle({
				choices: [{ message: { content: 'Here is the title:\nEMR 新架构接入' } }]
			})
		).toBeNull();
	});
});
