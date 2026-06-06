import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  parseSSEBlock,
  parseSSEBlocks,
  sseStateToLabel,
  buildNotificationFromSidechannel,
  removeViolation,
  apiAcceptImplication,
  apiIgnoreImplication,
  apiAcceptInference,
  apiIgnoreInference,
  apiGenerateInferences,
  apiRevalidateInferences,
  apiDeleteInference,
  apiPatchInferenceStatus,
  apiCreateFact,
  apiPatchFactMutability,
  apiPatchFactCategory,
  apiPromoteInference,
} from '../../src/memories/frontend/chat.js';

// ---------------------------------------------------------------------------
// parseSSEBlock
// ---------------------------------------------------------------------------

describe('parseSSEBlock', () => {
  it('parses event and data from a standard block', () => {
    const block = 'event: status\ndata: {"state":"generating"}';
    const result = parseSSEBlock(block);
    expect(result).toEqual({ event: 'status', data: '{"state":"generating"}' });
  });

  it('returns null when the block has no event line', () => {
    expect(parseSSEBlock('data: {"orphan":true}')).toBeNull();
  });

  it('returns null for an empty block', () => {
    expect(parseSSEBlock('')).toBeNull();
  });

  it('handles a block with extra whitespace lines', () => {
    const block = '\nevent: done\ndata: {}\n';
    const result = parseSSEBlock(block);
    expect(result?.event).toBe('done');
  });
});

// ---------------------------------------------------------------------------
// parseSSEBlocks
// ---------------------------------------------------------------------------

describe('parseSSEBlocks', () => {
  it('parses a single complete event block', () => {
    const text = 'event: status\ndata: {"state":"generating"}\n\n';
    const blocks = parseSSEBlocks(text);
    expect(blocks).toHaveLength(1);
    expect(blocks[0].event).toBe('status');
    expect(blocks[0].data).toBe('{"state":"generating"}');
  });

  it('parses multiple event blocks from one response', () => {
    const text =
      'event: status\ndata: {"state":"generating"}\n\n' +
      'event: message\ndata: {"content":"hi","turn_id":1}\n\n' +
      'event: done\ndata: {}\n\n';
    const blocks = parseSSEBlocks(text);
    expect(blocks).toHaveLength(3);
    expect(blocks.map(b => b.event)).toEqual(['status', 'message', 'done']);
  });

  it('returns empty array for empty input', () => {
    expect(parseSSEBlocks('')).toEqual([]);
  });

  it('filters out blocks that have no event field', () => {
    const text = 'data: {"orphan":true}\n\n';
    expect(parseSSEBlocks(text)).toEqual([]);
  });

  it('parses a trailing block that has no double-newline terminator', () => {
    // parseSSEBlocks processes whatever text it receives; buffer management
    // (popping the trailing incomplete chunk) is done by the streaming reader.
    const text =
      'event: status\ndata: {"state":"generating"}\n\n' +
      'event: done\ndata: {}';
    const blocks = parseSSEBlocks(text);
    expect(blocks).toHaveLength(2);
    expect(blocks[1].event).toBe('done');
  });
});

// ---------------------------------------------------------------------------
// sseStateToLabel
// ---------------------------------------------------------------------------

describe('sseStateToLabel', () => {
  it('returns the generating label', () => {
    expect(sseStateToLabel('generating')).toBe('Generating response…');
  });

  it('returns the reviewing label', () => {
    expect(sseStateToLabel('reviewing')).toBe('Reviewing against facts…');
  });

  it('returns the regenerating label', () => {
    expect(sseStateToLabel('regenerating')).toBe('Regenerating (contradiction found)…');
  });

  it('returns empty string for an unknown state', () => {
    expect(sseStateToLabel('unknown_state')).toBe('');
  });
});

// ---------------------------------------------------------------------------
// buildNotificationFromSidechannel
// ---------------------------------------------------------------------------

describe('buildNotificationFromSidechannel', () => {
  it('builds an implication notification with correct role and scType', () => {
    const payload = {
      type: 'implication',
      turn_id: 3,
      violations: [
        { type: 'implication', description: 'implied a sibling',
          suggested_fact: { key: 'siblings', value: 'one sister' } },
      ],
      new_inferences: [],
    };
    const notif = buildNotificationFromSidechannel(payload);
    expect(notif.role).toBe('notification');
    expect(notif.scType).toBe('implication');
    expect(notif.turn_id).toBe(3);
    expect(notif._loading).toBe(false);
  });

  it('initialises _editValue from suggested_fact.value', () => {
    const payload = {
      type: 'implication',
      turn_id: 1,
      violations: [
        { type: 'implication', description: 'd',
          suggested_fact: { key: 'k', value: 'expected-value' } },
      ],
      new_inferences: [],
    };
    const notif = buildNotificationFromSidechannel(payload);
    expect(notif.violations[0]._editValue).toBe('expected-value');
  });

  it('initialises _loading: false on each violation', () => {
    const payload = {
      type: 'implication',
      turn_id: 1,
      violations: [
        { type: 'implication', description: 'd1', suggested_fact: { key: 'k1', value: 'v1' } },
        { type: 'implication', description: 'd2', suggested_fact: { key: 'k2', value: 'v2' } },
      ],
      new_inferences: [],
    };
    const notif = buildNotificationFromSidechannel(payload);
    expect(notif.violations[0]._loading).toBe(false);
    expect(notif.violations[1]._loading).toBe(false);
  });

  it('initialises _editValue to empty string when suggested_fact is null', () => {
    const payload = {
      type: 'implication',
      turn_id: 1,
      violations: [{ type: 'implication', description: 'd', suggested_fact: null }],
      new_inferences: [],
    };
    const notif = buildNotificationFromSidechannel(payload);
    expect(notif.violations[0]._editValue).toBe('');
  });

  it('builds a probabilistic inference notification with correct structure', () => {
    const payload = {
      type: 'new_inference_probabilistic',
      turn_id: 2,
      violations: [],
      new_inferences: [
        { inference_type: 'probabilistic', statement: 'works long hours',
          derivation: 'occupation=surgeon', source_fact_ids: [1], source_inference_ids: [] },
      ],
    };
    const notif = buildNotificationFromSidechannel(payload);
    expect(notif.scType).toBe('new_inference_probabilistic');
    expect(notif.turn_id).toBe(2);
    expect(notif.new_inferences).toHaveLength(1);
    expect(notif._loading).toBe(false);
  });

  it('adds _loading: false to each inference item', () => {
    const payload = {
      type: 'new_inference_probabilistic',
      turn_id: 2,
      violations: [],
      new_inferences: [
        { inference_type: 'probabilistic', statement: 'A', derivation: 'x' },
        { inference_type: 'probabilistic', statement: 'B', derivation: 'y' },
      ],
    };
    const notif = buildNotificationFromSidechannel(payload);
    expect(notif.new_inferences[0]._loading).toBe(false);
    expect(notif.new_inferences[1]._loading).toBe(false);
  });

  it('builds a contradiction notification with iteration and description', () => {
    const payload = { type: 'contradiction', iteration: 2, description: 'character said London' };
    const notif = buildNotificationFromSidechannel(payload);
    expect(notif.scType).toBe('contradiction');
    expect(notif.iteration).toBe(2);
    expect(notif.description).toBe('character said London');
  });

  it('returns null for an unrecognised sidechannel type', () => {
    expect(buildNotificationFromSidechannel({ type: 'unknown' })).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// removeViolation
// ---------------------------------------------------------------------------

describe('removeViolation', () => {
  function makeNotif(...keys) {
    return {
      violations: keys.map(k => ({ suggested_fact: { key: k, value: 'v' }, _editValue: 'v' })),
    };
  }

  it('removes the target violation from the array', () => {
    const notif = makeNotif('age', 'hometown');
    const v = notif.violations[0];
    removeViolation(notif, v);
    expect(notif.violations).toHaveLength(1);
    expect(notif.violations[0].suggested_fact.key).toBe('hometown');
  });

  it('returns false when other violations still remain', () => {
    const notif = makeNotif('age', 'hometown');
    const result = removeViolation(notif, notif.violations[0]);
    expect(result).toBe(false);
  });

  it('returns true when the last violation is removed', () => {
    const notif = makeNotif('age');
    const result = removeViolation(notif, notif.violations[0]);
    expect(result).toBe(true);
    expect(notif.violations).toHaveLength(0);
  });

  it('is a no-op and returns true when the violation is not in the array', () => {
    const notif = makeNotif();
    const result = removeViolation(notif, { suggested_fact: { key: 'x', value: 'y' } });
    expect(result).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

describe('API helpers', () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({ ok: true });
  });

  it('apiAcceptImplication POSTs to the correct URL with key, value, and regenerate=true', async () => {
    await apiAcceptImplication(5, 3, 'siblings', 'one sister');
    expect(fetch).toHaveBeenCalledWith(
      '/api/sessions/5/turns/3/accept-implication',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ key: 'siblings', value: 'one sister', regenerate: true }),
      })
    );
  });

  it('apiAcceptImplication sends regenerate=false when explicitly passed', async () => {
    await apiAcceptImplication(5, 3, 'siblings', 'one sister', false);
    expect(fetch).toHaveBeenCalledWith(
      '/api/sessions/5/turns/3/accept-implication',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ key: 'siblings', value: 'one sister', regenerate: false }),
      })
    );
  });

  it('apiIgnoreImplication POSTs to the correct URL', async () => {
    await apiIgnoreImplication(5, 3);
    expect(fetch).toHaveBeenCalledWith(
      '/api/sessions/5/turns/3/ignore-implication',
      expect.objectContaining({ method: 'POST' })
    );
  });

  it('apiAcceptInference POSTs statement, derivation, and type to the correct URL', async () => {
    const inf = {
      statement: 'works long hours',
      derivation: 'occupation=surgeon',
      source_fact_ids: [1],
      inference_type: 'probabilistic',
    };
    await apiAcceptInference(5, 3, inf);
    expect(fetch).toHaveBeenCalledWith(
      '/api/sessions/5/turns/3/accept-inference',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          statement: 'works long hours',
          derivation: 'occupation=surgeon',
          source_fact_ids: [1],
          inference_type: 'probabilistic',
        }),
      })
    );
  });

  it('apiAcceptInference defaults source_fact_ids to [] and inference_type to probabilistic', async () => {
    await apiAcceptInference(1, 1, { statement: 's', derivation: 'd' });
    const body = JSON.parse(fetch.mock.calls[0][1].body);
    expect(body.source_fact_ids).toEqual([]);
    expect(body.inference_type).toBe('probabilistic');
  });

  it('apiIgnoreInference POSTs to the correct URL', async () => {
    await apiIgnoreInference(5, 3);
    expect(fetch).toHaveBeenCalledWith(
      '/api/sessions/5/turns/3/ignore-inference',
      expect.objectContaining({ method: 'POST' })
    );
  });
});

// ---------------------------------------------------------------------------
// Phase 3 — new inference management API helpers
// ---------------------------------------------------------------------------

describe('Phase 3 inference API helpers', () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({ ok: true });
  });

  it('apiGenerateInferences_posts_to_correct_url', async () => {
    await apiGenerateInferences(7);
    expect(fetch).toHaveBeenCalledWith(
      '/api/characters/7/inferences/generate',
      expect.objectContaining({ method: 'POST' })
    );
  });

  it('apiGenerateInferences_uses_post_method', async () => {
    await apiGenerateInferences(7);
    const [, opts] = fetch.mock.calls[0];
    expect(opts.method).toBe('POST');
  });

  it('apiRevalidateInferences_posts_to_correct_url', async () => {
    await apiRevalidateInferences(7, 42);
    expect(fetch).toHaveBeenCalledWith(
      '/api/characters/7/inferences/revalidate',
      expect.objectContaining({ method: 'POST' })
    );
  });

  it('apiRevalidateInferences_sends_changed_fact_id_in_body', async () => {
    await apiRevalidateInferences(7, 42);
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body.changed_fact_id).toBe(42);
  });

  it('apiDeleteInference_sends_delete_to_correct_url', async () => {
    await apiDeleteInference(7, 99);
    expect(fetch).toHaveBeenCalledWith(
      '/api/characters/7/inferences/99',
      expect.objectContaining({ method: 'DELETE' })
    );
  });

  it('apiPatchInferenceStatus_sends_patch_to_correct_url', async () => {
    await apiPatchInferenceStatus(7, 99, 'active');
    expect(fetch).toHaveBeenCalledWith(
      '/api/characters/7/inferences/99',
      expect.objectContaining({ method: 'PATCH' })
    );
  });

  it('apiPatchInferenceStatus_sends_status_in_body', async () => {
    await apiPatchInferenceStatus(7, 99, 'stale');
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body.status).toBe('stale');
  });
});

// ---------------------------------------------------------------------------
// Phase 4 — fact category/mutability and inference promotion API helpers
// ---------------------------------------------------------------------------

describe('Phase 4 fact API helpers', () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({ ok: true });
  });

  it('apiCreateFact_posts_to_correct_url', async () => {
    await apiCreateFact(7, 'mood', 'cheerful');
    expect(fetch).toHaveBeenCalledWith(
      '/api/characters/7/facts',
      expect.objectContaining({ method: 'POST' })
    );
  });

  it('apiCreateFact_sends_key_value_category_mutability_in_body', async () => {
    await apiCreateFact(7, 'mood', 'cheerful', 'character', 'high');
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body).toHaveProperty('key', 'mood');
    expect(body).toHaveProperty('value', 'cheerful');
    expect(body).toHaveProperty('category');
    expect(body).toHaveProperty('mutability');
  });

  it('apiCreateFact_uses_default_category_character', async () => {
    await apiCreateFact(7, 'mood', 'cheerful');
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body.category).toBe('character');
  });

  it('apiCreateFact_uses_default_mutability_immutable', async () => {
    await apiCreateFact(7, 'mood', 'cheerful');
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body.mutability).toBe('immutable');
  });

  it('apiCreateFact_accepts_custom_category', async () => {
    await apiCreateFact(7, 'mood', 'cheerful', 'user', 'high');
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body.category).toBe('user');
  });

  it('apiCreateFact_accepts_custom_mutability', async () => {
    await apiCreateFact(7, 'mood', 'cheerful', 'user', 'high');
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body.mutability).toBe('high');
  });

  it('apiPatchFactMutability_sends_patch_to_correct_url', async () => {
    await apiPatchFactMutability(7, 42, 'high');
    expect(fetch).toHaveBeenCalledWith(
      '/api/characters/7/facts/42',
      expect.objectContaining({ method: 'PATCH' })
    );
  });

  it('apiPatchFactMutability_sends_mutability_in_body', async () => {
    await apiPatchFactMutability(7, 42, 'high');
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body.mutability).toBe('high');
  });

  it('apiPatchFactMutability_does_not_send_category_field', async () => {
    await apiPatchFactMutability(7, 42, 'high');
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body).not.toHaveProperty('category');
  });

  it('apiPatchFactCategory_sends_patch_to_correct_url', async () => {
    await apiPatchFactCategory(7, 42, 'setting');
    expect(fetch).toHaveBeenCalledWith(
      '/api/characters/7/facts/42',
      expect.objectContaining({ method: 'PATCH' })
    );
  });

  it('apiPatchFactCategory_sends_category_in_body', async () => {
    await apiPatchFactCategory(7, 42, 'setting');
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body.category).toBe('setting');
  });

  it('apiPatchFactCategory_does_not_send_mutability_field', async () => {
    await apiPatchFactCategory(7, 42, 'setting');
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body).not.toHaveProperty('mutability');
  });

  it('apiPatchFactMutability_uses_integer_fact_id_in_url', async () => {
    await apiPatchFactMutability(7, 99, 'low');
    expect(fetch).toHaveBeenCalledWith(
      '/api/characters/7/facts/99',
      expect.anything()
    );
  });
});

describe('Phase 4 inference promotion API helper', () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({ ok: true });
  });

  it('apiPromoteInference_posts_to_correct_url', async () => {
    await apiPromoteInference(7, 42, 'birth_year', '1993');
    expect(fetch).toHaveBeenCalledWith(
      '/api/characters/7/inferences/42/promote',
      expect.objectContaining({ method: 'POST' })
    );
  });

  it('apiPromoteInference_sends_key_value_category_mutability_in_body', async () => {
    await apiPromoteInference(7, 42, 'birth_year', '1993', 'character', 'immutable');
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body).toHaveProperty('key', 'birth_year');
    expect(body).toHaveProperty('value', '1993');
    expect(body).toHaveProperty('category');
    expect(body).toHaveProperty('mutability');
  });

  it('apiPromoteInference_uses_default_category_character', async () => {
    await apiPromoteInference(7, 42, 'birth_year', '1993');
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body.category).toBe('character');
  });

  it('apiPromoteInference_uses_default_mutability_immutable', async () => {
    await apiPromoteInference(7, 42, 'birth_year', '1993');
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body.mutability).toBe('immutable');
  });

  it('apiPromoteInference_accepts_custom_category', async () => {
    await apiPromoteInference(7, 42, 'location', 'Chicago', 'setting', 'low');
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body.category).toBe('setting');
  });

  it('apiPromoteInference_accepts_custom_mutability', async () => {
    await apiPromoteInference(7, 42, 'location', 'Chicago', 'setting', 'low');
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body.mutability).toBe('low');
  });
});

describe('Phase 4 buildNotificationFromSidechannel — mutability implication', () => {
  it('buildNotificationFromSidechannel_still_handles_implication_for_mutability_change', () => {
    const payload = {
      type: 'implication',
      turn_id: 5,
      violations: [
        {
          type: 'implication',
          description: "Mood appears to have shifted from 'cheerful' to 'anxious' (high-mutability fact)",
          suggested_fact: { key: 'mood', value: 'anxious' },
        },
      ],
      new_inferences: [],
    };
    const notif = buildNotificationFromSidechannel(payload);
    expect(notif).not.toBeNull();
    expect(notif.scType).toBe('implication');
    expect(notif.turn_id).toBe(5);
    expect(notif.violations[0]._editValue).toBe('anxious');
    expect(notif._loading).toBe(false);
  });
});
