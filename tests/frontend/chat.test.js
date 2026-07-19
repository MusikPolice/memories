import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  parseSSEBlock,
  parseSSEBlocks,
  sseStateToLabel,
  buildNotificationFromSidechannel,
  apiGenerateInferences,
  apiRevalidateInferences,
  apiDeleteInference,
  apiPatchInferenceStatus,
  apiEndSession,
  apiCreateExperience,
  apiListExperiences,
  apiDeleteExperience,
  sortExperiences,
  buildScoreMap,
  buildProposalList,
  flattenSchema,
  lookupBlobValue,
  buildVisibleFactRows,
  schemaLeaf,
  apiGetSchema,
  apiGetFactBlob,
  apiSetFactValue,
  apiRespondRequireFact,
  apiRespondSetFact,
} from '../../src/memories/frontend/chat.js';

// ---------------------------------------------------------------------------
// Shared schema/blob fixtures for the schema-tree helpers
// ---------------------------------------------------------------------------

const SAMPLE_SCHEMA = {
  Character: {
    Identity: {
      Name: { Type: 'String', Mutability: 'Immutable', Description: 'the name' },
      Age: { Type: 'Integer', Mutability: 'Immutable', Description: 'the age' },
    },
    Mood: {
      Type: 'Enum',
      Constraint: ['Calm', 'Anxious'],
      Mutability: 'Fluid',
      Description: 'the mood',
    },
  },
};

const SAMPLE_BLOB = { Character: { Identity: { Name: { Value: 'Sarah' } } } };

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
// buildNotificationFromSidechannel — Facts v2 cards
// ---------------------------------------------------------------------------

describe('buildNotificationFromSidechannel', () => {
  it('builds a contradiction notification with iteration and description', () => {
    const payload = { type: 'contradiction', iteration: 2, description: 'character said London' };
    const notif = buildNotificationFromSidechannel(payload);
    expect(notif.scType).toBe('contradiction');
    expect(notif.iteration).toBe(2);
    expect(notif.description).toBe('character said London');
  });

  it('test_buildNotification_fact_update_fluid_shape', () => {
    const payload = {
      type: 'fact_update_fluid', turn_id: 4,
      path: 'Character.State-Of-Mind.Mood', value: 'Anxious',
    };
    const notif = buildNotificationFromSidechannel(payload);
    expect(notif.scType).toBe('fact_update_fluid');
    expect(notif.path).toBe('Character.State-Of-Mind.Mood');
    expect(notif.value).toBe('Anxious');
  });

  it('test_buildNotification_fact_update_mutable_seeds_editValue_from_proposed', () => {
    const payload = {
      type: 'fact_update_mutable', turn_id: 4,
      path: 'Character.Identity.Occupation', proposed: 'Detective',
    };
    const notif = buildNotificationFromSidechannel(payload);
    expect(notif.scType).toBe('fact_update_mutable');
    expect(notif._editValue).toBe('Detective');
    expect(notif._loading).toBe(false);
  });

  it('test_buildNotification_fact_update_immutable_unset_seeds_editValue_from_proposed', () => {
    const payload = {
      type: 'fact_update_immutable_unset', turn_id: 4,
      path: 'Character.Identity.Name', proposed: 'Sarah',
    };
    const notif = buildNotificationFromSidechannel(payload);
    expect(notif.scType).toBe('fact_update_immutable_unset');
    expect(notif._editValue).toBe('Sarah');
    expect(notif._loading).toBe(false);
  });

  it('test_buildNotification_require_fact_seeds_editValue_from_suggested_value', () => {
    const payload = {
      type: 'require_fact', turn_id: 4,
      path: 'Character.Identity.Name', reason: 'needed', suggested_value: 'Sarah',
    };
    const notif = buildNotificationFromSidechannel(payload);
    expect(notif.scType).toBe('require_fact');
    expect(notif._editValue).toBe('Sarah');
    expect(notif._loading).toBe(false);
  });

  it('test_buildNotification_require_fact_editValue_empty_when_suggested_value_absent', () => {
    const payload = {
      type: 'require_fact', turn_id: 4,
      path: 'Character.Identity.Name', reason: 'needed',
    };
    const notif = buildNotificationFromSidechannel(payload);
    expect(notif._editValue).toBe('');
  });

  it('test_buildNotification_inference_proposed_carries_inference_object', () => {
    const inference = { statement: 'works long hours', derivation: 'occupation=surgeon' };
    const payload = { type: 'inference_proposed', turn_id: 4, inference };
    const notif = buildNotificationFromSidechannel(payload);
    expect(notif.scType).toBe('inference_proposed');
    expect(notif.inference).toEqual(inference);
  });

  it('test_buildNotification_returns_null_for_removed_implication_type', () => {
    expect(buildNotificationFromSidechannel({ type: 'implication' })).toBeNull();
  });

  it('returns null for an unrecognised sidechannel type', () => {
    expect(buildNotificationFromSidechannel({ type: 'unknown' })).toBeNull();
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
// Phase 5 API helpers — apiEndSession
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
// Phase 5 API helpers — apiCreateExperience
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
// Phase 5 API helpers — apiListExperiences
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
    expect(fetch.mock.calls[0][1]).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Phase 5 API helpers — apiDeleteExperience
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
// Phase 5 — parseSSEBlock with new message event fields
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
// Phase 5 — sortExperiences
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
// Phase 6 — sseStateToLabel for extracting state
// ---------------------------------------------------------------------------

describe('Phase 6 sseStateToLabel — extracting state', () => {
  it('sseStateToLabel_extracting_returns_non_empty_string', () => {
    const label = sseStateToLabel('extracting');
    expect(label).not.toBe('');
  });

  it('sseStateToLabel_extracting_differs_from_generating', () => {
    const label = sseStateToLabel('extracting');
    expect(label).not.toBe(sseStateToLabel('generating'));
    expect(label).not.toBe('');
  });
});

// ---------------------------------------------------------------------------
// Step 8 — flattenSchema
// ---------------------------------------------------------------------------

describe('flattenSchema', () => {
  it('test_flattenSchema_marks_leaves_and_groups', () => {
    const rows = flattenSchema(SAMPLE_SCHEMA);
    const group = rows.find(r => r.path === 'Character');
    const leaf = rows.find(r => r.path === 'Character.Identity.Name');
    expect(group.isLeaf).toBe(false);
    expect(leaf.isLeaf).toBe(true);
    expect(leaf.type).toBe('String');
    expect(leaf.mutability).toBe('Immutable');
  });

  it('test_flattenSchema_depth_increases_per_level', () => {
    const rows = flattenSchema(SAMPLE_SCHEMA);
    expect(rows.find(r => r.path === 'Character').depth).toBe(0);
    expect(rows.find(r => r.path === 'Character.Identity').depth).toBe(1);
    expect(rows.find(r => r.path === 'Character.Identity.Name').depth).toBe(2);
  });

  it('test_flattenSchema_leaf_carries_constraint_for_enum', () => {
    const rows = flattenSchema(SAMPLE_SCHEMA);
    const mood = rows.find(r => r.path === 'Character.Mood');
    expect(mood.constraint).toEqual(['Calm', 'Anxious']);
  });

  it('test_flattenSchema_preserves_key_order', () => {
    const rows = flattenSchema(SAMPLE_SCHEMA);
    expect(rows.map(r => r.path)).toEqual([
      'Character',
      'Character.Identity',
      'Character.Identity.Name',
      'Character.Identity.Age',
      'Character.Mood',
    ]);
  });
});

// ---------------------------------------------------------------------------
// Step 8 — lookupBlobValue
// ---------------------------------------------------------------------------

describe('lookupBlobValue', () => {
  it('test_lookupBlobValue_returns_value_for_set_leaf', () => {
    expect(lookupBlobValue(SAMPLE_BLOB, 'Character.Identity.Name')).toBe('Sarah');
  });

  it('test_lookupBlobValue_returns_undefined_for_unset_path', () => {
    expect(lookupBlobValue(SAMPLE_BLOB, 'Character.Identity.Age')).toBeUndefined();
  });

  it('test_lookupBlobValue_returns_undefined_for_partial_path', () => {
    expect(lookupBlobValue(SAMPLE_BLOB, 'Character.Identity')).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Step 8 — buildVisibleFactRows
// ---------------------------------------------------------------------------

describe('buildVisibleFactRows', () => {
  it('test_buildVisibleFactRows_attaches_value_to_leaves', () => {
    const rows = buildVisibleFactRows(SAMPLE_SCHEMA, SAMPLE_BLOB, new Set());
    const name = rows.find(r => r.path === 'Character.Identity.Name');
    expect(name.value).toBe('Sarah');
  });

  it('test_buildVisibleFactRows_hides_descendants_of_collapsed_group', () => {
    const rows = buildVisibleFactRows(SAMPLE_SCHEMA, SAMPLE_BLOB, new Set(['Character.Identity']));
    expect(rows.find(r => r.path === 'Character.Identity.Name')).toBeUndefined();
    expect(rows.find(r => r.path === 'Character.Identity.Age')).toBeUndefined();
  });

  it('test_buildVisibleFactRows_shows_collapsed_group_row_itself', () => {
    const rows = buildVisibleFactRows(SAMPLE_SCHEMA, SAMPLE_BLOB, new Set(['Character.Identity']));
    expect(rows.find(r => r.path === 'Character.Identity')).toBeDefined();
  });

  it('test_buildVisibleFactRows_leaf_value_undefined_when_unset', () => {
    const rows = buildVisibleFactRows(SAMPLE_SCHEMA, SAMPLE_BLOB, new Set());
    const age = rows.find(r => r.path === 'Character.Identity.Age');
    expect(age.value).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Step 8 — schemaLeaf
// ---------------------------------------------------------------------------

describe('schemaLeaf', () => {
  it('test_schemaLeaf_returns_leaf_for_valid_path', () => {
    const leaf = schemaLeaf(SAMPLE_SCHEMA, 'Character.Identity.Name');
    expect(leaf.type).toBe('String');
  });

  it('test_schemaLeaf_returns_null_for_unknown_path', () => {
    expect(schemaLeaf(SAMPLE_SCHEMA, 'Nope.Not.Here')).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Step 8 — API helpers
// ---------------------------------------------------------------------------

describe('Step 8 API helpers', () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({ ok: true });
  });

  it('test_apiGetSchema_gets_slash_api_schema', async () => {
    await apiGetSchema();
    expect(fetch).toHaveBeenCalledWith('/api/schema');
  });

  it('test_apiGetFactBlob_gets_character_facts_url', async () => {
    await apiGetFactBlob(7);
    expect(fetch).toHaveBeenCalledWith('/api/characters/7/facts');
  });

  it('test_apiSetFactValue_puts_path_and_value', async () => {
    await apiSetFactValue(7, 'Character.Identity.Name', 'Sarah');
    expect(fetch).toHaveBeenCalledWith(
      '/api/characters/7/facts',
      expect.objectContaining({ method: 'PUT' })
    );
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body.path).toBe('Character.Identity.Name');
    expect(body.value).toBe('Sarah');
  });

  it('test_apiRespondRequireFact_posts_value', async () => {
    await apiRespondRequireFact(5, 3, 'Sarah');
    expect(fetch).toHaveBeenCalledWith(
      '/api/sessions/5/turns/3/require-fact/respond',
      expect.objectContaining({ method: 'POST' })
    );
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body.value).toBe('Sarah');
  });

  it('test_apiRespondSetFact_posts_action_and_value', async () => {
    await apiRespondSetFact(5, 3, 'edit', 'Sarah');
    expect(fetch).toHaveBeenCalledWith(
      '/api/sessions/5/turns/3/set-fact/respond',
      expect.objectContaining({ method: 'POST' })
    );
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body.action).toBe('edit');
    expect(body.value).toBe('Sarah');
  });

  it('test_apiRespondSetFact_defaults_value_null', async () => {
    await apiRespondSetFact(5, 3, 'accept');
    const [, opts] = fetch.mock.calls[0];
    const body = JSON.parse(opts.body);
    expect(body.value).toBeNull();
  });
});
