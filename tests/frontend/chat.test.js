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
  apiEndSession,
  apiCreateExperience,
  apiListExperiences,
  apiDeleteExperience,
  sortExperiences,
  buildScoreMap,
  buildProposalList,
  removeContradictedExperiences,
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

  it('apiAcceptImplication POSTs to the correct URL with key, value, regenerate=true, and default category', async () => {
    await apiAcceptImplication(5, 3, 'siblings', 'one sister');
    expect(fetch).toHaveBeenCalledWith(
      '/api/sessions/5/turns/3/accept-implication',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ key: 'siblings', value: 'one sister', regenerate: true, category: 'character' }),
      })
    );
  });

  it('apiAcceptImplication sends regenerate=false when explicitly passed', async () => {
    await apiAcceptImplication(5, 3, 'siblings', 'one sister', false);
    expect(fetch).toHaveBeenCalledWith(
      '/api/sessions/5/turns/3/accept-implication',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ key: 'siblings', value: 'one sister', regenerate: false, category: 'character' }),
      })
    );
  });

  it('apiAcceptImplication forwards the supplied category to the backend', async () => {
    await apiAcceptImplication(5, 3, 'jacket_colour', 'blue', true, 'user');
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body.category).toBe('user');
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

// ---------------------------------------------------------------------------
// Phase 5 — experience_update sidechannel notification (tests 182–186)
// ---------------------------------------------------------------------------

describe('Phase 5 buildNotificationFromSidechannel — experience_update', () => {
  it('buildNotificationFromSidechannel_experience_update_has_scType', () => {
    const payload = { type: 'experience_update', turn_id: 4, experience_updates: [] };
    const notif = buildNotificationFromSidechannel(payload);
    expect(notif.scType).toBe('experience_update');
  });

  it('buildNotificationFromSidechannel_experience_update_has_turn_id', () => {
    const payload = { type: 'experience_update', turn_id: 7, experience_updates: [] };
    const notif = buildNotificationFromSidechannel(payload);
    expect(notif.turn_id).toBe(7);
  });

  it('buildNotificationFromSidechannel_experience_update_has_experience_updates_array', () => {
    const updates = [
      { contradicted_experience_id: 5, description: 'now in New York' },
      { contradicted_experience_id: 8, description: 'changed job' },
    ];
    const payload = { type: 'experience_update', turn_id: 4, experience_updates: updates };
    const notif = buildNotificationFromSidechannel(payload);
    expect(notif.experience_updates).toEqual(updates);
  });

  it('buildNotificationFromSidechannel_experience_update_empty_updates_array', () => {
    const payload = { type: 'experience_update', turn_id: 4, experience_updates: [] };
    const notif = buildNotificationFromSidechannel(payload);
    expect(notif.experience_updates).toEqual([]);
  });

  it('buildNotificationFromSidechannel_experience_update_has_loading_false', () => {
    const payload = { type: 'experience_update', turn_id: 4, experience_updates: [] };
    const notif = buildNotificationFromSidechannel(payload);
    expect(notif._loading).toBe(false);
  });

  it('buildNotificationFromSidechannel_experience_update_missing_field_defaults_to_empty_array', () => {
    // payload has no experience_updates key — the || [] fallback must fire
    const payload = { type: 'experience_update', turn_id: 4 };
    const notif = buildNotificationFromSidechannel(payload);
    expect(notif.experience_updates).toEqual([]);
  });

  it('buildNotificationFromSidechannel_experience_update_clones_items', () => {
    const original = { contradicted_experience_id: 5, description: 'now in New York' };
    const payload = { type: 'experience_update', turn_id: 4, experience_updates: [original] };
    const notif = buildNotificationFromSidechannel(payload);
    // Items must be copies so Vue mutations don't alias back to the SSE object.
    expect(notif.experience_updates[0]).not.toBe(original);
    expect(notif.experience_updates[0]).toEqual(original);
  });
});

// ---------------------------------------------------------------------------
// Phase 5 API helpers — apiEndSession (tests 187–189)
// ---------------------------------------------------------------------------

describe('Phase 5 apiEndSession', () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({ ok: true });
  });

  it('apiEndSession_posts_to_correct_url', async () => {
    await apiEndSession(7);
    expect(fetch).toHaveBeenCalledWith(
      '/api/sessions/7/end',
      expect.objectContaining({ method: 'POST' })
    );
  });

  it('apiEndSession_sends_no_body', async () => {
    await apiEndSession(7);
    const [, opts] = fetch.mock.calls[0];
    expect(opts).not.toHaveProperty('body');
  });
});

// ---------------------------------------------------------------------------
// Phase 5 API helpers — apiCreateExperience (tests 190–195)
// ---------------------------------------------------------------------------

describe('Phase 5 apiCreateExperience', () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({ ok: true });
  });

  it('apiCreateExperience_posts_to_correct_url', async () => {
    await apiCreateExperience(7, 3, 'We are in Chicago', 'told_by_user');
    expect(fetch).toHaveBeenCalledWith(
      '/api/characters/7/experiences',
      expect.objectContaining({ method: 'POST' })
    );
  });

  it('apiCreateExperience_sends_session_id_in_body', async () => {
    await apiCreateExperience(7, 3, 'We are in Chicago', 'told_by_user');
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body.session_id).toBe(3);
  });

  it('apiCreateExperience_sends_statement_in_body', async () => {
    await apiCreateExperience(7, 3, 'We are in Chicago', 'told_by_user');
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body.statement).toBe('We are in Chicago');
  });

  it('apiCreateExperience_sends_source_in_body', async () => {
    await apiCreateExperience(7, 3, 'We are in Chicago', 'told_by_user');
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body).toHaveProperty('source');
  });

  it('apiCreateExperience_sends_told_by_user_source', async () => {
    await apiCreateExperience(7, 3, 'some text', 'told_by_user');
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body.source).toBe('told_by_user');
  });

  it('apiCreateExperience_sends_observed_source', async () => {
    await apiCreateExperience(7, 3, 'some text', 'observed');
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body.source).toBe('observed');
  });

  it('apiCreateExperience_sends_content_type_json_header', async () => {
    await apiCreateExperience(7, 3, 'some text', 'told_by_user');
    const [, opts] = fetch.mock.calls[0];
    expect(opts.headers['Content-Type']).toBe('application/json');
  });
});

// ---------------------------------------------------------------------------
// Phase 5 API helpers — apiListExperiences (tests 196–197)
// ---------------------------------------------------------------------------

describe('Phase 5 apiListExperiences', () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({ ok: true });
  });

  it('apiListExperiences_gets_correct_url', async () => {
    await apiListExperiences(7);
    expect(fetch).toHaveBeenCalledWith('/api/characters/7/experiences');
  });

  it('apiListExperiences_uses_get_method', async () => {
    await apiListExperiences(7);
    // No options object passed → browser defaults to GET.
    // Assert on the absence of the options arg, not on call arity, so that
    // adding an AbortSignal later does not produce a misleading failure.
    expect(fetch.mock.calls[0][1]).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Phase 5 API helpers — apiDeleteExperience (tests 198–199)
// ---------------------------------------------------------------------------

describe('Phase 5 apiDeleteExperience', () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({ ok: true });
  });

  it('apiDeleteExperience_sends_delete_to_correct_url', async () => {
    await apiDeleteExperience(7, 42);
    expect(fetch).toHaveBeenCalledWith(
      '/api/characters/7/experiences/42',
      expect.objectContaining({ method: 'DELETE' })
    );
  });

});

// ---------------------------------------------------------------------------
// Phase 5 — parseSSEBlock with new message event fields (tests 200–202)
// ---------------------------------------------------------------------------

describe('Phase 5 parseSSEBlock — message event with experience fields', () => {
  it('parseSSEBlock_message_event_exposes_active_experience_ids', () => {
    const block =
      'event: message\n' +
      'data: {"role":"assistant","content":"hi","turn_id":1,"active_experience_ids":[3,7],"experience_scores":[]}';
    const result = parseSSEBlock(block);
    const data = JSON.parse(result.data);
    expect(data.active_experience_ids).toEqual([3, 7]);
  });

  it('parseSSEBlock_message_event_active_experience_ids_defaults_absent', () => {
    const block = 'event: message\ndata: {"role":"assistant","content":"hi","turn_id":1}';
    const result = parseSSEBlock(block);
    const data = JSON.parse(result.data);
    expect(data).not.toHaveProperty('active_experience_ids');
  });

  it('parseSSEBlock_message_event_exposes_experience_scores', () => {
    const block =
      'event: message\n' +
      'data: {"role":"assistant","content":"hi","turn_id":1,"active_experience_ids":[],"experience_scores":[{"id":3,"score":0.8}]}';
    const result = parseSSEBlock(block);
    const data = JSON.parse(result.data);
    expect(data.experience_scores).toEqual([{ id: 3, score: 0.8 }]);
  });
});

// ---------------------------------------------------------------------------
// Phase 5 — sortExperiences (tests 203–208)
// ---------------------------------------------------------------------------

describe('sortExperiences', () => {
  function makeExp(id) {
    return { id, statement: `Experience ${id}` };
  }

  it('sortExperiences_active_before_inactive', () => {
    const experiences = [makeExp(1), makeExp(2)];
    const activeIds = new Set([2]);
    const scoreMap = new Map([[1, 0.5], [2, 0.5]]);
    const sorted = sortExperiences(experiences, activeIds, scoreMap);
    expect(sorted[0].id).toBe(2);
    expect(sorted[1].id).toBe(1);
  });

  it('sortExperiences_active_sorted_by_score_descending', () => {
    const experiences = [makeExp(1), makeExp(2)];
    const activeIds = new Set([1, 2]);
    const scoreMap = new Map([[1, 0.4], [2, 0.9]]);
    const sorted = sortExperiences(experiences, activeIds, scoreMap);
    expect(sorted[0].id).toBe(2);
    expect(sorted[1].id).toBe(1);
  });

  it('sortExperiences_inactive_sorted_by_score_descending', () => {
    const experiences = [makeExp(1), makeExp(2)];
    const activeIds = new Set([]);
    const scoreMap = new Map([[1, 0.2], [2, 0.7]]);
    const sorted = sortExperiences(experiences, activeIds, scoreMap);
    expect(sorted[0].id).toBe(2);
    expect(sorted[1].id).toBe(1);
  });

  it('sortExperiences_active_group_always_above_inactive_group', () => {
    const experiences = [makeExp(1), makeExp(2)];
    const activeIds = new Set([1]);
    const scoreMap = new Map([[1, 0.1], [2, 0.9]]);
    const sorted = sortExperiences(experiences, activeIds, scoreMap);
    expect(sorted[0].id).toBe(1); // active despite lower score
    expect(sorted[1].id).toBe(2);
  });

  it('sortExperiences_no_score_experience_falls_to_bottom', () => {
    const experiences = [makeExp(1), makeExp(2), makeExp(3)];
    const activeIds = new Set([]);
    // exp 1 has no score; exp 2 = 0.5; exp 3 = 0.3
    const scoreMap = new Map([[2, 0.5], [3, 0.3]]);
    const sorted = sortExperiences(experiences, activeIds, scoreMap);
    expect(sorted[0].id).toBe(2);
    expect(sorted[1].id).toBe(3);
    expect(sorted[2].id).toBe(1); // no score → last
  });

  it('sortExperiences_stable_when_scores_and_active_status_equal', () => {
    const experiences = [makeExp(10), makeExp(20), makeExp(30)];
    const activeIds = new Set([]);
    const scoreMap = new Map([[10, 0.5], [20, 0.5], [30, 0.5]]);
    const sorted = sortExperiences(experiences, activeIds, scoreMap);
    // All identical priority — original order must be preserved (stable sort)
    expect(sorted.map(e => e.id)).toEqual([10, 20, 30]);
  });

  it('sortExperiences_two_unscored_experiences_preserve_order', () => {
    // Both absent from scoreMap → aScore = bScore = -Infinity.
    // Comparator must return 0, not NaN (-Infinity - (-Infinity) = NaN).
    const experiences = [makeExp(1), makeExp(2)];
    const activeIds = new Set([]);
    const scoreMap = new Map();
    const sorted = sortExperiences(experiences, activeIds, scoreMap);
    expect(sorted.map(e => e.id)).toEqual([1, 2]);
  });

  it('sortExperiences_does_not_mutate_input_array', () => {
    const experiences = [makeExp(3), makeExp(1), makeExp(2)];
    const activeIds = new Set([1]);
    const scoreMap = new Map([[1, 0.9], [2, 0.5], [3, 0.1]]);
    const inputOrder = experiences.map(e => e.id);
    sortExperiences(experiences, activeIds, scoreMap);
    expect(experiences.map(e => e.id)).toEqual(inputOrder);
  });
});

// ---------------------------------------------------------------------------
// Phase 5 — buildScoreMap (converts SSE experience_scores array → Map)
// ---------------------------------------------------------------------------

describe('buildScoreMap', () => {
  it('buildScoreMap_returns_empty_map_for_empty_input', () => {
    expect(buildScoreMap([])).toEqual(new Map());
  });

  it('buildScoreMap_returns_a_Map_instance', () => {
    expect(buildScoreMap([{ id: 1, score: 0.5 }])).toBeInstanceOf(Map);
  });

  it('buildScoreMap_maps_id_to_score', () => {
    const map = buildScoreMap([{ id: 3, score: 0.8 }, { id: 7, score: 0.2 }]);
    expect(map.get(3)).toBe(0.8);
    expect(map.get(7)).toBe(0.2);
  });

  it('buildScoreMap_missing_key_returns_undefined', () => {
    const map = buildScoreMap([{ id: 1, score: 0.5 }]);
    expect(map.get(99)).toBeUndefined();
  });

  it('buildScoreMap_result_is_directly_usable_by_sortExperiences', () => {
    const experiences = [{ id: 1, statement: 'a' }, { id: 2, statement: 'b' }];
    const scoreMap = buildScoreMap([{ id: 1, score: 0.2 }, { id: 2, score: 0.9 }]);
    const sorted = sortExperiences(experiences, new Set(), scoreMap);
    expect(sorted[0].id).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// buildProposalList
// ---------------------------------------------------------------------------

describe('buildProposalList', () => {
  it('buildProposalList_returns_empty_array_for_empty_input', () => {
    expect(buildProposalList([])).toEqual([]);
  });

  it('buildProposalList_returns_empty_array_for_null_input', () => {
    expect(buildProposalList(null)).toEqual([]);
  });

  it('buildProposalList_returns_empty_array_for_undefined_input', () => {
    expect(buildProposalList(undefined)).toEqual([]);
  });

  it('buildProposalList_wraps_each_proposal_with_ui_state', () => {
    const raw = [{ statement: 'We are in Chicago', source: 'told_by_user', turn_reference: 2 }];
    const result = buildProposalList(raw);
    expect(result[0]._editing).toBe(false);
    expect(result[0]._editStatement).toBe('');
    expect(result[0]._loading).toBe(false);
  });

  it('buildProposalList_preserves_statement', () => {
    const raw = [{ statement: 'We are in Chicago', source: 'told_by_user' }];
    expect(buildProposalList(raw)[0].statement).toBe('We are in Chicago');
  });

  it('buildProposalList_preserves_source', () => {
    const raw = [{ statement: 'text', source: 'observed' }];
    expect(buildProposalList(raw)[0].source).toBe('observed');
  });

  it('buildProposalList_preserves_turn_reference', () => {
    const raw = [{ statement: 'text', source: 'told_by_user', turn_reference: 7 }];
    expect(buildProposalList(raw)[0].turn_reference).toBe(7);
  });

  it('buildProposalList_handles_multiple_proposals', () => {
    const raw = [
      { statement: 'A', source: 'told_by_user' },
      { statement: 'B', source: 'observed' },
    ];
    const result = buildProposalList(raw);
    expect(result).toHaveLength(2);
    expect(result[1].statement).toBe('B');
  });

  it('buildProposalList_does_not_mutate_input', () => {
    const raw = [{ statement: 'A', source: 'told_by_user' }];
    const original = { ...raw[0] };
    buildProposalList(raw);
    expect(raw[0]).toEqual(original);
  });

  it('buildProposalList_returns_new_objects_not_same_references', () => {
    const raw = [{ statement: 'A', source: 'told_by_user' }];
    const result = buildProposalList(raw);
    expect(result[0]).not.toBe(raw[0]);
  });
});

// ---------------------------------------------------------------------------
// removeContradictedExperiences
// ---------------------------------------------------------------------------

describe('removeContradictedExperiences', () => {
  const experiences = [
    { id: 1, statement: 'We are in Chicago', source: 'told_by_user' },
    { id: 2, statement: 'User dislikes mornings', source: 'observed' },
    { id: 3, statement: 'User has a cat', source: 'told_by_user' },
  ];

  it('removeContradictedExperiences_removes_the_contradicted_experience', () => {
    const notif = {
      scType: 'experience_update',
      experience_updates: [{ contradicted_experience_id: 1, description: 'now in New York' }],
    };
    const result = removeContradictedExperiences(experiences, notif);
    expect(result.map(e => e.id)).not.toContain(1);
  });

  it('removeContradictedExperiences_keeps_non_contradicted_experiences', () => {
    const notif = {
      scType: 'experience_update',
      experience_updates: [{ contradicted_experience_id: 1, description: 'desc' }],
    };
    const result = removeContradictedExperiences(experiences, notif);
    expect(result.map(e => e.id)).toContain(2);
    expect(result.map(e => e.id)).toContain(3);
  });

  it('removeContradictedExperiences_removes_multiple_ids', () => {
    const notif = {
      scType: 'experience_update',
      experience_updates: [
        { contradicted_experience_id: 1, description: 'a' },
        { contradicted_experience_id: 3, description: 'b' },
      ],
    };
    const result = removeContradictedExperiences(experiences, notif);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe(2);
  });

  it('removeContradictedExperiences_returns_all_when_no_updates', () => {
    const notif = { scType: 'experience_update', experience_updates: [] };
    const result = removeContradictedExperiences(experiences, notif);
    expect(result).toHaveLength(3);
  });

  it('removeContradictedExperiences_handles_missing_experience_updates_field', () => {
    const notif = { scType: 'experience_update' };
    const result = removeContradictedExperiences(experiences, notif);
    expect(result).toHaveLength(3);
  });

  it('removeContradictedExperiences_returns_empty_when_all_removed', () => {
    const notif = {
      scType: 'experience_update',
      experience_updates: [
        { contradicted_experience_id: 1, description: 'a' },
        { contradicted_experience_id: 2, description: 'b' },
        { contradicted_experience_id: 3, description: 'c' },
      ],
    };
    expect(removeContradictedExperiences(experiences, notif)).toHaveLength(0);
  });

  it('removeContradictedExperiences_ignores_unknown_id_gracefully', () => {
    const notif = {
      scType: 'experience_update',
      experience_updates: [{ contradicted_experience_id: 99, description: 'ghost' }],
    };
    const result = removeContradictedExperiences(experiences, notif);
    expect(result).toHaveLength(3);
  });

  it('removeContradictedExperiences_does_not_mutate_input_array', () => {
    const notif = {
      scType: 'experience_update',
      experience_updates: [{ contradicted_experience_id: 1, description: 'd' }],
    };
    removeContradictedExperiences(experiences, notif);
    expect(experiences).toHaveLength(3);
  });

  it('removeContradictedExperiences_returns_new_array_not_same_reference', () => {
    const notif = { scType: 'experience_update', experience_updates: [] };
    const result = removeContradictedExperiences(experiences, notif);
    expect(result).not.toBe(experiences);
  });
});
