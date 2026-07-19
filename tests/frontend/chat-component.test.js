import { describe, it, expect, vi, beforeAll, afterAll, beforeEach, afterEach } from 'vitest';
import { ChatComponent } from '../../src/memories/frontend/chat-component.js';

// Vue emits onMounted/onUnmounted warnings when setup() is called outside a component tree.
// This is intentional — we test the setup() function directly without mounting.
beforeAll(() => { vi.spyOn(console, 'warn').mockImplementation(() => {}); });
afterAll(() => { vi.restoreAllMocks(); });

// ---------------------------------------------------------------------------
// Shared schema/blob fixtures
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
// Helpers
// ---------------------------------------------------------------------------

/**
 * Deliver a list of SSE blocks as a single streaming response chunk.
 * Each entry in sseBlocks should be a complete `event: ...\ndata: ...` string.
 */
function makeStreamResponse(sseBlocks) {
  const text = sseBlocks.join('\n\n') + '\n\n';
  const bytes = new TextEncoder().encode(text);
  let consumed = false;
  return {
    ok: true,
    body: {
      getReader: () => ({
        read: async () => {
          if (!consumed) { consumed = true; return { done: false, value: bytes }; }
          return { done: true, value: undefined };
        },
      }),
    },
  };
}

/**
 * Stub fetch with a URL-routing mock, then call ChatComponent.setup().
 *
 * routes maps URL substrings to plain data objects/arrays.
 * Unmatched URLs return { ok: true, json: () => [] }.
 * The stub is set up before setup() is called so the loadCharacters() fire-and-forget
 * that runs at the end of setup() hits the correct mock.
 */
function setupComponent(routes = {}) {
  vi.stubGlobal('fetch', vi.fn().mockImplementation(async (url) => {
    for (const [pattern, data] of Object.entries(routes)) {
      if (url && url.includes(pattern)) return { ok: true, json: async () => data };
    }
    return { ok: true, json: async () => [] };
  }));
  return ChatComponent.setup();
}

// ---------------------------------------------------------------------------
// endSession state management
// ---------------------------------------------------------------------------

describe('endSession', () => {
  let vm;

  beforeEach(() => { vm = setupComponent(); });
  afterEach(() => { vi.unstubAllGlobals(); });

  function mockEnd(proposedExperiences = []) {
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async (url) => {
      if (url && url.includes('/end')) return { ok: true, json: async () => ({ proposed_experiences: proposedExperiences }) };
      return { ok: true, json: async () => [] };
    }));
  }

  it('endSession_sets_sessionEnded_true', async () => {
    mockEnd();
    await vm.endSession();
    expect(vm.sessionEnded.value).toBe(true);
  });

  it('endSession_resets_activeExperienceIds_to_empty_set', async () => {
    vm.activeExperienceIds.value = new Set([1, 2, 3]);
    mockEnd();
    await vm.endSession();
    expect(vm.activeExperienceIds.value.size).toBe(0);
  });

  it('endSession_resets_experienceScoreMap_to_empty_map', async () => {
    vm.experienceScoreMap.value = new Map([[1, 0.9], [2, 0.5]]);
    mockEnd();
    await vm.endSession();
    expect(vm.experienceScoreMap.value.size).toBe(0);
  });

  it('endSession_sets_sessionProposals_from_response', async () => {
    mockEnd([{ statement: 'We went to the park', source: 'told_by_user' }]);
    await vm.endSession();
    expect(vm.sessionProposals.value).toHaveLength(1);
    expect(vm.sessionProposals.value[0].statement).toBe('We went to the park');
  });

  it('endSession_calls_loadExperiences_when_no_proposals', async () => {
    vm.currentCharacter.value = { id: 7, name: 'Alice' };
    let experiencesFetched = false;
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async (url) => {
      if (url && url.includes('/end')) return { ok: true, json: async () => ({ proposed_experiences: [] }) };
      if (url && url.includes('/experiences')) { experiencesFetched = true; return { ok: true, json: async () => [] }; }
      return { ok: true, json: async () => [] };
    }));
    await vm.endSession();
    expect(experiencesFetched).toBe(true);
  });

  it('endSession_sets_reviewingSession_false_after_response', async () => {
    mockEnd();
    await vm.endSession();
    expect(vm.reviewingSession.value).toBe(false);
  });

  it('endSession_sets_reviewingSession_false_on_api_failure', async () => {
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async (url) => {
      if (url && url.includes('/end')) return { ok: false, json: async () => ({}) };
      return { ok: true, json: async () => [] };
    }));
    await vm.endSession();
    expect(vm.reviewingSession.value).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// SSE message event — active experience tracking
// ---------------------------------------------------------------------------

describe('sendMessage SSE message event — active experience tracking', () => {
  let vm;

  beforeEach(() => { vm = setupComponent(); });
  afterEach(() => { vi.unstubAllGlobals(); });

  function mockMessages(messageData) {
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async (url) => {
      if (url && url.includes('/messages')) {
        return makeStreamResponse([
          'event: message\ndata: ' + JSON.stringify(messageData),
          'event: done\ndata: {}',
        ]);
      }
      return { ok: true, json: async () => [] };
    }));
  }

  it('sendMessage_message_event_updates_activeExperienceIds', async () => {
    vm.inputText.value = 'hello';
    mockMessages({ role: 'assistant', content: 'hi', turn_id: 1, active_experience_ids: [3, 7], experience_scores: [] });
    await vm.sendMessage();
    expect(vm.activeExperienceIds.value.has(3)).toBe(true);
    expect(vm.activeExperienceIds.value.has(7)).toBe(true);
    expect(vm.activeExperienceIds.value.size).toBe(2);
  });

  it('sendMessage_message_event_updates_experienceScoreMap', async () => {
    vm.inputText.value = 'hello';
    mockMessages({
      role: 'assistant', content: 'hi', turn_id: 1,
      active_experience_ids: [3],
      experience_scores: [{ id: 3, score: 0.85 }, { id: 7, score: 0.42 }],
    });
    await vm.sendMessage();
    expect(vm.experienceScoreMap.value.get(3)).toBe(0.85);
    expect(vm.experienceScoreMap.value.get(7)).toBe(0.42);
  });

  it('sendMessage_message_event_calls_buildScoreMap', async () => {
    // Verifies the Map is fully replaced — not merged with previous scores.
    vm.experienceScoreMap.value = new Map([[99, 0.1], [100, 0.2]]);
    vm.inputText.value = 'hello';
    mockMessages({
      role: 'assistant', content: 'hi', turn_id: 1,
      active_experience_ids: [],
      experience_scores: [{ id: 3, score: 0.7 }],
    });
    await vm.sendMessage();
    expect(vm.experienceScoreMap.value.has(99)).toBe(false);
    expect(vm.experienceScoreMap.value.get(3)).toBe(0.7);
  });
});

// ---------------------------------------------------------------------------
// newSession reset
// ---------------------------------------------------------------------------

describe('newSession', () => {
  let vm;

  beforeEach(() => { vm = setupComponent(); });
  afterEach(() => { vi.unstubAllGlobals(); });

  it('newSession_resets_experiences_to_empty', async () => {
    vm.experiences.value = [{ id: 1, statement: 'test', source: 'observed' }];
    await vm.newSession();
    expect(vm.experiences.value).toEqual([]);
  });

  it('newSession_resets_sessionProposals_to_empty', async () => {
    vm.sessionProposals.value = [{ statement: 'test', source: 'observed', _editing: false, _editStatement: '', _loading: false }];
    await vm.newSession();
    expect(vm.sessionProposals.value).toEqual([]);
  });

  it('newSession_resets_activeExperienceIds_to_empty_set', async () => {
    vm.activeExperienceIds.value = new Set([1, 2]);
    await vm.newSession();
    expect(vm.activeExperienceIds.value.size).toBe(0);
  });

  it('newSession_resets_experienceScoreMap_to_empty_map', async () => {
    vm.experienceScoreMap.value = new Map([[1, 0.5], [2, 0.8]]);
    await vm.newSession();
    expect(vm.experienceScoreMap.value.size).toBe(0);
  });

  it('newSession_resets_factsBlob_to_empty', async () => {
    vm.factsBlob.value = SAMPLE_BLOB;
    await vm.newSession();
    expect(vm.factsBlob.value).toEqual({});
  });

  it('newSession_resets_collapsedGroups_to_empty', async () => {
    vm.collapsedGroups.value = new Set(['Character']);
    await vm.newSession();
    expect(vm.collapsedGroups.value.size).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Proposal lifecycle
// ---------------------------------------------------------------------------

describe('proposal lifecycle', () => {
  let vm;

  beforeEach(() => {
    vm = setupComponent();
    vm.currentCharacter.value = { id: 7, name: 'Alice' };
    vm.sessionId.value = 3;
  });
  afterEach(() => { vi.unstubAllGlobals(); });

  function makeProposal(statement = 'We went to the park') {
    return { statement, source: 'told_by_user', _editing: false, _editStatement: '', _loading: false };
  }

  it('acceptProposal_calls_apiCreateExperience', async () => {
    const p = makeProposal();
    vm.sessionProposals.value = [p];
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => [] }));
    await vm.acceptProposal(p);
    const urls = fetch.mock.calls.map(c => c[0]);
    expect(urls.some(u => u && u.includes('/characters/7/experiences'))).toBe(true);
  });

  it('acceptProposal_removes_proposal_from_sessionProposals', async () => {
    const p = makeProposal();
    vm.sessionProposals.value = [p];
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => [] }));
    await vm.acceptProposal(p);
    expect(vm.sessionProposals.value).toHaveLength(0);
  });

  it('acceptProposal_calls_loadExperiences_when_last_proposal_accepted', async () => {
    const p = makeProposal();
    vm.sessionProposals.value = [p];
    let experiencesFetched = false;
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async (url) => {
      if (url && url.includes('/experiences')) { experiencesFetched = true; }
      return { ok: true, json: async () => [] };
    }));
    await vm.acceptProposal(p);
    expect(experiencesFetched).toBe(true);
  });

  it('confirmEditProposal_calls_apiCreateExperience_with_edited_text', async () => {
    const p = makeProposal('Original statement');
    p._editStatement = 'Edited statement';
    vm.sessionProposals.value = [p];
    let capturedBody;
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async (url, opts) => {
      if (url && url.includes('/experiences') && opts?.method === 'POST') {
        capturedBody = JSON.parse(opts.body);
      }
      return { ok: true, json: async () => [] };
    }));
    await vm.confirmEditProposal(p);
    expect(capturedBody.statement).toBe('Edited statement');
  });

  it('discardProposal_removes_proposal_without_api_call', () => {
    const p1 = makeProposal('First');
    const p2 = makeProposal('Second');
    vm.sessionProposals.value = [p1, p2];
    const callsBefore = fetch.mock.calls.length;
    vm.discardProposal(0);
    expect(vm.sessionProposals.value).toHaveLength(1);
    expect(vm.sessionProposals.value[0].statement).toBe('Second');
    expect(fetch.mock.calls.length).toBe(callsBefore);
  });

  it('deleteExperience_calls_apiDeleteExperience', async () => {
    const exp = { id: 42, statement: 'test', source: 'observed' };
    vm.experiences.value = [exp];
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => [] }));
    await vm.deleteExperience(exp);
    const urls = fetch.mock.calls.map(c => c[0]);
    expect(urls.some(u => u && u.includes('/experiences/42'))).toBe(true);
  });

  it('deleteExperience_reloads_experiences_after_delete', async () => {
    const exp = { id: 42, statement: 'test', source: 'observed' };
    vm.experiences.value = [exp];
    const refreshed = [{ id: 99, statement: 'fresh', source: 'observed' }];
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async (url) => {
      // GET /experiences returns the refreshed list; DELETE to /experiences/42 returns ok
      if (url && url.includes('/experiences') && !url.includes('/42')) {
        return { ok: true, json: async () => refreshed };
      }
      return { ok: true, json: async () => [] };
    }));
    await vm.deleteExperience(exp);
    expect(vm.experiences.value).toEqual(refreshed);
  });
});

// ---------------------------------------------------------------------------
// dismissNotification
// ---------------------------------------------------------------------------

describe('dismissNotification', () => {
  let vm;
  beforeEach(() => { vm = setupComponent(); });
  afterEach(() => { vi.unstubAllGlobals(); });

  it('dismissNotification_removes_notification_from_messages', () => {
    const notif = { role: 'notification', scType: 'contradiction' };
    vm.messages.value = [{ role: 'user', content: 'hi' }, notif];
    vm.dismissNotification(notif);
    expect(vm.messages.value).toHaveLength(1);
    expect(vm.messages.value[0].role).toBe('user');
  });

  it('dismissNotification_is_noop_when_notification_not_in_messages', () => {
    vm.messages.value = [{ role: 'user', content: 'hi' }];
    vm.dismissNotification({ role: 'notification', scType: 'other' });
    expect(vm.messages.value).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// Step 8 — schema fetch, blob load, and tree state
// ---------------------------------------------------------------------------

describe('schema / blob / tree state', () => {
  let vm;
  afterEach(() => { vi.unstubAllGlobals(); });

  it('test_loadSchema_populates_schema_ref', async () => {
    vm = setupComponent({ '/api/schema': SAMPLE_SCHEMA });
    await vm.loadSchema();
    expect(vm.schema.value).toHaveProperty('Character');
  });

  it('test_loadFacts_populates_factsBlob', async () => {
    vm = setupComponent({ '/facts': SAMPLE_BLOB });
    vm.currentCharacter.value = { id: 7, name: 'Alice' };
    await vm.loadFacts();
    expect(vm.factsBlob.value).toEqual(SAMPLE_BLOB);
  });

  it('test_visibleFactRows_reflects_schema_and_blob', () => {
    vm = setupComponent();
    vm.schema.value = SAMPLE_SCHEMA;
    vm.factsBlob.value = SAMPLE_BLOB;
    const name = vm.visibleFactRows.value.find(r => r.path === 'Character.Identity.Name');
    expect(name.value).toBe('Sarah');
  });

  it('test_toggleGroup_collapses_and_expands', () => {
    vm = setupComponent();
    vm.toggleGroup('Character');
    expect(vm.collapsedGroups.value.has('Character')).toBe(true);
    vm.toggleGroup('Character');
    expect(vm.collapsedGroups.value.has('Character')).toBe(false);
  });

  it('test_saveLeaf_puts_value_and_reloads', async () => {
    vm = setupComponent();
    vm.currentCharacter.value = { id: 7, name: 'Alice' };
    vm.leafEdits.value = { 'Character.Identity.Name': 'Sarah' };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) }));
    await vm.saveLeaf({ path: 'Character.Identity.Name' });
    const calls = fetch.mock.calls.map(([url, opts]) => ({ url, method: opts?.method }));
    expect(calls.some(c => c.url.includes('/facts') && c.method === 'PUT')).toBe(true);
    expect(calls.some(c => c.url.includes('/facts') && c.method === undefined)).toBe(true);
  });

  it('test_leafType_and_leafConstraint_read_from_schema', () => {
    vm = setupComponent();
    vm.schema.value = SAMPLE_SCHEMA;
    expect(vm.leafType('Character.Mood')).toBe('Enum');
    expect(vm.leafConstraint('Character.Mood')).toEqual(['Calm', 'Anxious']);
  });
});

// ---------------------------------------------------------------------------
// Step 8 — deleteInference
// ---------------------------------------------------------------------------

describe('deleteInference', () => {
  let vm;
  beforeEach(() => {
    vm = setupComponent();
    vm.currentCharacter.value = { id: 7, name: 'Alice' };
  });
  afterEach(() => { vi.unstubAllGlobals(); });

  it('test_deleteInference_deletes_and_reloads', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => [] }));
    await vm.deleteInference({ id: 99 });
    const calls = fetch.mock.calls.map(([url, opts]) => ({ url, method: opts?.method }));
    expect(calls.some(c => c.url.includes('/inferences/99') && c.method === 'DELETE')).toBe(true);
    expect(calls.some(c => c.url.includes('/inferences') && c.method === undefined)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Step 8 — sendMessage sidechannel cards (Facts v2)
// ---------------------------------------------------------------------------

describe('sendMessage Facts v2 sidechannel cards', () => {
  let vm;

  beforeEach(() => { vm = setupComponent(); });
  afterEach(() => { vi.unstubAllGlobals(); });

  function mockMessages(sseBlocks) {
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async (url) => {
      if (url && url.includes('/messages')) return makeStreamResponse(sseBlocks);
      return { ok: true, json: async () => [] };
    }));
  }

  it('test_sendMessage_fact_update_fluid_pushes_quiet_notification', async () => {
    vm.inputText.value = 'hello';
    mockMessages([
      'event: sidechannel\ndata: ' + JSON.stringify({
        type: 'fact_update_fluid', turn_id: 1,
        path: 'Character.State-Of-Mind.Mood', value: 'Anxious',
      }),
      'event: message\ndata: {"role":"assistant","content":"hi","turn_id":1}',
      'event: done\ndata: {}',
    ]);
    await vm.sendMessage();
    const notif = vm.messages.value.find(m => m.scType === 'fact_update_fluid');
    expect(notif).toBeDefined();
  });

  it('test_sendMessage_fact_update_mutable_pushes_blocking_card_and_stops_spinner', async () => {
    vm.inputText.value = 'hello';
    mockMessages([
      'event: sidechannel\ndata: ' + JSON.stringify({
        type: 'fact_update_mutable', turn_id: 1,
        path: 'Character.Identity.Occupation', proposed: 'Detective',
      }),
    ]);
    await vm.sendMessage();
    const notif = vm.messages.value.find(m => m.scType === 'fact_update_mutable');
    expect(notif).toBeDefined();
    expect(vm.generating.value).toBe(false);
  });

  it('test_sendMessage_require_fact_pushes_blocking_card', async () => {
    vm.inputText.value = 'hello';
    mockMessages([
      'event: sidechannel\ndata: ' + JSON.stringify({
        type: 'require_fact', turn_id: 1,
        path: 'Character.Identity.Name', reason: 'needed', suggested_value: 'Sarah',
      }),
    ]);
    await vm.sendMessage();
    const notif = vm.messages.value.find(m => m.scType === 'require_fact');
    expect(notif).toBeDefined();
  });

  it('test_sendMessage_inference_proposed_pushes_notification', async () => {
    vm.inputText.value = 'hello';
    mockMessages([
      'event: sidechannel\ndata: ' + JSON.stringify({
        type: 'inference_proposed', turn_id: 1,
        inference: { statement: 'works long hours', derivation: 'occupation=surgeon' },
      }),
      'event: message\ndata: {"role":"assistant","content":"hi","turn_id":1}',
      'event: done\ndata: {}',
    ]);
    await vm.sendMessage();
    const notif = vm.messages.value.find(m => m.scType === 'inference_proposed');
    expect(notif).toBeDefined();
  });

  it('test_sendMessage_done_reloads_facts_and_inferences', async () => {
    vm.currentCharacter.value = { id: 7, name: 'Alice' };
    vm.sessionId.value = 3;
    let factsFetched = false;
    let inferencesFetched = false;
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async (url) => {
      if (url && url.includes('/messages')) {
        return makeStreamResponse([
          'event: message\ndata: {"role":"assistant","content":"Hi","turn_id":1}',
          'event: done\ndata: {}',
        ]);
      }
      if (url && url.includes('/facts')) factsFetched = true;
      if (url && url.includes('/inferences')) inferencesFetched = true;
      return { ok: true, json: async () => [] };
    }));
    vm.inputText.value = 'hello';
    await vm.sendMessage();
    expect(factsFetched).toBe(true);
    expect(inferencesFetched).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Step 8 — blocking-card resolve handlers
// ---------------------------------------------------------------------------

describe('blocking-card resolve handlers', () => {
  let vm;

  beforeEach(() => {
    vm = setupComponent();
    vm.sessionId.value = 5;
  });
  afterEach(() => { vi.unstubAllGlobals(); });

  function makeMutableCard() {
    return {
      role: 'notification', scType: 'fact_update_mutable', turn_id: 3,
      path: 'Character.Identity.Occupation', proposed: 'Detective',
      _editValue: 'Detective', _loading: false,
    };
  }

  function makeImmutableCard() {
    return {
      role: 'notification', scType: 'fact_update_immutable_unset', turn_id: 3,
      path: 'Character.Identity.Name', proposed: 'Sarah',
      _editValue: 'Sarah', _loading: false,
    };
  }

  function makeRequireCard() {
    return {
      role: 'notification', scType: 'require_fact', turn_id: 3,
      path: 'Character.Identity.Name', reason: 'needed',
      _editValue: 'Sarah', _loading: false,
    };
  }

  it('test_resolveMutable_accept_posts_action_and_dismisses', async () => {
    const notif = makeMutableCard();
    vm.messages.value = [notif];
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) }));
    await vm.resolveMutable(notif, 'accept');
    const call = fetch.mock.calls.find(([url]) => url.includes('/set-fact/respond'));
    expect(call).toBeDefined();
    expect(JSON.parse(call[1].body).action).toBe('accept');
    expect(vm.messages.value.find(m => m === notif)).toBeUndefined();
  });

  it('test_resolveMutable_edit_posts_editValue', async () => {
    const notif = makeMutableCard();
    notif._editValue = 'Nurse';
    vm.messages.value = [notif];
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) }));
    await vm.resolveMutable(notif, 'edit');
    const call = fetch.mock.calls.find(([url]) => url.includes('/set-fact/respond'));
    const body = JSON.parse(call[1].body);
    expect(body.action).toBe('edit');
    expect(body.value).toBe('Nurse');
  });

  it('test_resolveImmutable_dismiss_posts_dismiss', async () => {
    const notif = makeImmutableCard();
    vm.messages.value = [notif];
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) }));
    await vm.resolveImmutable(notif, 'dismiss');
    const call = fetch.mock.calls.find(([url]) => url.includes('/set-fact/respond'));
    const body = JSON.parse(call[1].body);
    expect(body.action).toBe('dismiss');
    expect(body.value).toBeNull();
  });

  it('test_resolveRequireFact_confirm_posts_editValue', async () => {
    const notif = makeRequireCard();
    vm.messages.value = [notif];
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) }));
    await vm.resolveRequireFact(notif, true);
    const call = fetch.mock.calls.find(([url]) => url.includes('/require-fact/respond'));
    expect(call).toBeDefined();
    expect(JSON.parse(call[1].body).value).toBe('Sarah');
  });

  it('test_resolveRequireFact_notnow_posts_null', async () => {
    const notif = makeRequireCard();
    vm.messages.value = [notif];
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) }));
    await vm.resolveRequireFact(notif, false);
    const call = fetch.mock.calls.find(([url]) => url.includes('/require-fact/respond'));
    expect(JSON.parse(call[1].body).value).toBeNull();
  });
});
