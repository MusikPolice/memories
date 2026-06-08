import { describe, it, expect, vi, beforeAll, afterAll, beforeEach, afterEach } from 'vitest';
import { ChatComponent } from '../../src/memories/frontend/chat-component.js';

// Vue emits onMounted/onUnmounted warnings when setup() is called outside a component tree.
// This is intentional — we test the setup() function directly without mounting.
beforeAll(() => { vi.spyOn(console, 'warn').mockImplementation(() => {}); });
afterAll(() => { vi.restoreAllMocks(); });

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
// SSE sidechannel — experience_update
// ---------------------------------------------------------------------------

describe('sendMessage SSE sidechannel — experience_update', () => {
  let vm;

  beforeEach(() => { vm = setupComponent(); });
  afterEach(() => { vi.unstubAllGlobals(); });

  function mockMessages(sseBlocks) {
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async (url) => {
      if (url && url.includes('/messages')) return makeStreamResponse(sseBlocks);
      return { ok: true, json: async () => [] };
    }));
  }

  it('sendMessage_experience_update_sidechannel_removes_experience_from_list', async () => {
    vm.experiences.value = [{ id: 5, statement: 'We are in Chicago', source: 'told_by_user' }];
    vm.inputText.value = 'hello';
    mockMessages([
      'event: sidechannel\ndata: ' + JSON.stringify({
        type: 'experience_update', turn_id: 1,
        experience_updates: [{ contradicted_experience_id: 5, description: 'Now in New York' }],
      }),
      'event: message\ndata: {"role":"assistant","content":"hi","turn_id":1}',
      'event: done\ndata: {}',
    ]);
    await vm.sendMessage();
    expect(vm.experiences.value).toHaveLength(0);
  });

  it('sendMessage_experience_update_sidechannel_pushes_notification_to_messages', async () => {
    vm.inputText.value = 'hello';
    mockMessages([
      'event: sidechannel\ndata: ' + JSON.stringify({
        type: 'experience_update', turn_id: 1,
        experience_updates: [{ contradicted_experience_id: 5, description: 'Now in New York' }],
      }),
      'event: message\ndata: {"role":"assistant","content":"hi","turn_id":1}',
      'event: done\ndata: {}',
    ]);
    await vm.sendMessage();
    const notif = vm.messages.value.find(m => m.scType === 'experience_update');
    expect(notif).toBeDefined();
    expect(notif.experience_updates).toHaveLength(1);
  });

  it('sendMessage_experience_update_sidechannel_calls_removeContradictedExperiences', async () => {
    // Three experiences; only id:5 is contradicted → ids 10 and 20 must survive.
    vm.experiences.value = [
      { id: 5, statement: 'We are in Chicago', source: 'told_by_user' },
      { id: 10, statement: 'User likes coffee', source: 'observed' },
      { id: 20, statement: 'User works mornings', source: 'observed' },
    ];
    vm.inputText.value = 'hello';
    mockMessages([
      'event: sidechannel\ndata: ' + JSON.stringify({
        type: 'experience_update', turn_id: 1,
        experience_updates: [{ contradicted_experience_id: 5, description: 'desc' }],
      }),
      'event: message\ndata: {"role":"assistant","content":"hi","turn_id":1}',
      'event: done\ndata: {}',
    ]);
    await vm.sendMessage();
    expect(vm.experiences.value.map(e => e.id)).not.toContain(5);
    expect(vm.experiences.value.map(e => e.id)).toContain(10);
    expect(vm.experiences.value.map(e => e.id)).toContain(20);
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
// mutabilityIcon helper
// ---------------------------------------------------------------------------

describe('mutabilityIcon', () => {
  let vm;
  beforeEach(() => { vm = setupComponent(); });
  afterEach(() => { vi.unstubAllGlobals(); });

  it('mutabilityIcon_returns_lock_for_immutable', () => {
    expect(vm.mutabilityIcon('immutable')).toBe('🔒');
  });

  it('mutabilityIcon_returns_pin_for_low', () => {
    expect(vm.mutabilityIcon('low')).toBe('📌');
  });

  it('mutabilityIcon_returns_droplet_for_high', () => {
    expect(vm.mutabilityIcon('high')).toBe('💧');
  });
});

// ---------------------------------------------------------------------------
// factsByCategory computed
// ---------------------------------------------------------------------------

describe('factsByCategory', () => {
  let vm;
  beforeEach(() => { vm = setupComponent(); });
  afterEach(() => { vi.unstubAllGlobals(); });

  it('factsByCategory_groups_facts_by_category', () => {
    vm.facts.value = [
      { id: 1, category: 'user', key: 'name', value: 'Alice' },
      { id: 2, category: 'character', key: 'mood', value: 'happy' },
      { id: 3, category: 'setting', key: 'city', value: 'Chicago' },
      { id: 4, category: 'character', key: 'age', value: '30' },
    ];
    const groups = vm.factsByCategory.value;
    expect(groups.user).toHaveLength(1);
    expect(groups.character).toHaveLength(2);
    expect(groups.setting).toHaveLength(1);
  });

  it('factsByCategory_returns_empty_arrays_when_no_facts', () => {
    vm.facts.value = [];
    const groups = vm.factsByCategory.value;
    expect(groups.user).toEqual([]);
    expect(groups.character).toEqual([]);
    expect(groups.setting).toEqual([]);
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
// saveFact — also exercises loadFacts and loadInferences
// ---------------------------------------------------------------------------

describe('saveFact', () => {
  let vm;
  beforeEach(() => {
    vm = setupComponent();
    vm.currentCharacter.value = { id: 7, name: 'Alice' };
  });
  afterEach(() => { vi.unstubAllGlobals(); });

  it('saveFact_sends_put_request_with_updated_value', async () => {
    const fact = { id: 42, key: 'city', value: 'Chicago', _editValue: 'New York' };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => [] }));
    await vm.saveFact(fact);
    const calls = fetch.mock.calls.map(([url, opts]) => ({ url, method: opts?.method }));
    expect(calls.some(c => c.url.includes('/facts/42') && c.method === 'PUT')).toBe(true);
  });

  it('saveFact_reloads_facts_and_inferences_after_save', async () => {
    const fact = { id: 42, key: 'city', value: 'Chicago', _editValue: 'New York' };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => [] }));
    await vm.saveFact(fact);
    const urls = fetch.mock.calls.map(c => c[0]);
    expect(urls.some(u => u && u.includes('/facts') && !u.includes('/42'))).toBe(true);
    expect(urls.some(u => u && u.includes('/inferences'))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// acceptInference / ignoreInference
// ---------------------------------------------------------------------------

describe('acceptInference', () => {
  let vm;
  beforeEach(() => {
    vm = setupComponent();
    vm.sessionId.value = 5;
    vm.currentCharacter.value = { id: 7, name: 'Alice' };
  });
  afterEach(() => { vi.unstubAllGlobals(); });

  it('acceptInference_removes_inference_from_notification_on_success', async () => {
    const inf = { statement: 'works long hours', derivation: 'surgeon', _loading: false };
    const notif = { role: 'notification', scType: 'new_inference_probabilistic', turn_id: 1, new_inferences: [inf] };
    vm.messages.value = [notif];
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => [] }));
    await vm.acceptInference(notif, inf);
    expect(notif.new_inferences).toHaveLength(0);
  });

  it('acceptInference_dismisses_notification_when_last_inference_accepted', async () => {
    const inf = { statement: 'works long hours', derivation: 'surgeon', _loading: false };
    const notif = { role: 'notification', scType: 'new_inference_probabilistic', turn_id: 1, new_inferences: [inf] };
    vm.messages.value = [notif];
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => [] }));
    await vm.acceptInference(notif, inf);
    expect(vm.messages.value.find(m => m.scType === 'new_inference_probabilistic')).toBeUndefined();
  });
});

describe('ignoreInference', () => {
  let vm;
  beforeEach(() => {
    vm = setupComponent();
    vm.sessionId.value = 5;
  });
  afterEach(() => { vi.unstubAllGlobals(); });

  it('ignoreInference_removes_inference_from_notification', async () => {
    const inf = { statement: 'works long hours', _loading: false };
    const inf2 = { statement: 'second inference', _loading: false };
    const notif = { role: 'notification', turn_id: 1, new_inferences: [inf, inf2] };
    vm.messages.value = [notif];
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true }));
    await vm.ignoreInference(notif, inf);
    // One inference removed; one remains → notification stays, no API call yet
    expect(notif.new_inferences).toHaveLength(1);
    expect(notif.new_inferences[0].statement).toBe('second inference');
  });

  it('ignoreInference_calls_api_and_dismisses_when_last_inference_ignored', async () => {
    const inf = { statement: 'works long hours', _loading: false };
    const notif = { role: 'notification', turn_id: 1, new_inferences: [inf] };
    vm.messages.value = [notif];
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true }));
    await vm.ignoreInference(notif, inf);
    const urls = fetch.mock.calls.map(c => c[0]);
    expect(urls.some(u => u && u.includes('ignore-inference'))).toBe(true);
    expect(vm.messages.value.find(m => m === notif)).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// acceptImplication / ignoreImplication
// ---------------------------------------------------------------------------

describe('acceptImplication', () => {
  let vm;
  beforeEach(() => {
    vm = setupComponent();
    vm.sessionId.value = 5;
    vm.currentCharacter.value = { id: 7, name: 'Alice' };
  });
  afterEach(() => { vi.unstubAllGlobals(); });

  it('acceptImplication_does_nothing_when_violation_has_no_suggested_fact', async () => {
    const violation = { _loading: false, suggested_fact: null, _editValue: '' };
    const notif = { violations: [violation], turn_id: 1 };
    const callsBefore = fetch.mock.calls.length;
    await vm.acceptImplication(notif, violation);
    expect(fetch.mock.calls.length).toBe(callsBefore);
  });

  it('acceptImplication_updates_assistant_message_content_on_success', async () => {
    const violation = {
      suggested_fact: { key: 'city', value: 'Chicago', category: 'setting' },
      _editValue: 'Chicago',
      _loading: false,
    };
    const notif = { violations: [violation], turn_id: 2 };
    const assistantMsg = { role: 'assistant', turn_id: 2, content: 'Original' };
    vm.messages.value = [notif, assistantMsg];
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async (url) => {
      if (url && url.includes('accept-implication')) return { ok: true, json: async () => ({ content: 'Updated' }) };
      return { ok: true, json: async () => [] };
    }));
    await vm.acceptImplication(notif, violation);
    expect(assistantMsg.content).toBe('Updated');
  });
});

describe('ignoreImplication', () => {
  let vm;
  beforeEach(() => {
    vm = setupComponent();
    vm.sessionId.value = 5;
  });
  afterEach(() => { vi.unstubAllGlobals(); });

  it('ignoreImplication_calls_ignore_endpoint_when_last_violation_removed', async () => {
    const violation = { suggested_fact: { key: 'city', value: 'Chicago' }, _editValue: '', _loading: false };
    const notif = { violations: [violation], turn_id: 1 };
    vm.messages.value = [notif];
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true }));
    await vm.ignoreImplication(notif, violation);
    const urls = fetch.mock.calls.map(c => c[0]);
    expect(urls.some(u => u && u.includes('ignore-implication'))).toBe(true);
  });

  it('ignoreImplication_dismisses_notification_when_last_violation_removed', async () => {
    const violation = { suggested_fact: { key: 'city', value: 'Chicago' }, _editValue: '', _loading: false };
    const notif = { violations: [violation], turn_id: 1 };
    vm.messages.value = [notif];
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true }));
    await vm.ignoreImplication(notif, violation);
    expect(vm.messages.value.find(m => m === notif)).toBeUndefined();
  });
});
