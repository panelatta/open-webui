import { describe, expect, it } from 'vitest';
import { mergeStatus } from './status';

describe('generation status', () => {
 it('updates one display row in place while preserving tool statuses', () => {
  const pending = { action: 'generation_config', id: 'generation-config:a', done: false };
  const search = { action: 'web_search', done: true };
  const done = { ...pending, done: true, reasoning_effort: 'medium' };
  const history = [pending, search];
  expect(mergeStatus(history, done)).toEqual([done, search]);
  expect(history).toEqual([pending, search]);
 });
 it('retains append behavior for other statuses and separate messages', () => {
  const item = { action: 'web_search', done: true };
  expect(mergeStatus([item], item)).toEqual([item, item]);
  expect(mergeStatus([], { action: 'generation_config', id: 'new' })).toHaveLength(1);
 });
});
