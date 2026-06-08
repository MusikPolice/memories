/**
 * Pure/testable frontend logic for the chat UI.
 *
 * These functions are imported by index.html (ES module) and tested by
 * tests/frontend/chat.test.js (Vitest + jsdom).  Keep this file free of
 * Vue-specific imports so it can be exercised without a Vue runtime.
 */

// ---------------------------------------------------------------------------
// SSE parsing
// ---------------------------------------------------------------------------

/**
 * Parse one SSE block (the text between two double-newlines) into an object.
 * Returns null if the block has no `event:` line.
 *
 * @param {string} block
 * @returns {{ event: string, data: string } | null}
 */
export function parseSSEBlock(block) {
  let event = '';
  let data = '';
  for (const line of block.split('\n')) {
    if (line.startsWith('event: ')) event = line.slice(7);
    if (line.startsWith('data: ')) data = line.slice(6);
  }
  return event ? { event, data } : null;
}

/**
 * Parse a complete SSE response body into an array of `{ event, data }` objects.
 * Blocks without an `event:` line are dropped.
 *
 * @param {string} text
 * @returns {Array<{ event: string, data: string }>}
 */
export function parseSSEBlocks(text) {
  return text
    .split('\n\n')
    .filter(b => b.trim())
    .map(parseSSEBlock)
    .filter(Boolean);
}

// ---------------------------------------------------------------------------
// Status label
// ---------------------------------------------------------------------------

const _STATE_LABELS = {
  generating: 'Generating response…',
  reviewing: 'Reviewing against facts…',
  regenerating: 'Regenerating (contradiction found)…',
};

/**
 * Map an SSE status state string to a human-readable label.
 *
 * @param {string} state
 * @returns {string}
 */
export function sseStateToLabel(state) {
  return _STATE_LABELS[state] ?? '';
}

// ---------------------------------------------------------------------------
// Notification builder
// ---------------------------------------------------------------------------

/**
 * Convert a `sidechannel` SSE payload into a notification object for the
 * messages array.  Returns null for unrecognised types.
 *
 * @param {object} payload  Parsed JSON from a `sidechannel` SSE event.
 * @returns {object | null}
 */
export function buildNotificationFromSidechannel(payload) {
  if (payload.type === 'implication') {
    return {
      role: 'notification',
      scType: 'implication',
      turn_id: payload.turn_id,
      violations: (payload.violations || []).map(v => ({
        ...v,
        _editValue: v.suggested_fact?.value ?? '',
        _loading: false,
      })),
      _loading: false,
    };
  }

  if (payload.type === 'new_inference_probabilistic') {
    return {
      role: 'notification',
      scType: 'new_inference_probabilistic',
      turn_id: payload.turn_id,
      new_inferences: (payload.new_inferences || []).map(inf => ({ ...inf, _loading: false })),
      _loading: false,
    };
  }

  if (payload.type === 'contradiction') {
    return {
      role: 'notification',
      scType: 'contradiction',
      iteration: payload.iteration,
      description: payload.description,
    };
  }

  if (payload.type === 'experience_update') {
    return {
      role: 'notification',
      scType: 'experience_update',
      turn_id: payload.turn_id,
      experience_updates: (payload.experience_updates || []).map(u => ({ ...u })),
      _loading: false,
    };
  }

  return null;
}

// ---------------------------------------------------------------------------
// Violation helpers
// ---------------------------------------------------------------------------

/**
 * Remove one violation from a notification's violations array in-place.
 * Returns true when the list is now empty (caller should dismiss the card).
 *
 * @param {{ violations: object[] }} notif
 * @param {object} violation
 * @returns {boolean}
 */
export function removeViolation(notif, violation) {
  const idx = notif.violations.indexOf(violation);
  if (idx !== -1) notif.violations.splice(idx, 1);
  return notif.violations.length === 0;
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

/**
 * @param {number} sessionId
 * @param {number} turnId
 * @param {string} key
 * @param {string} value
 * @param {boolean} [regenerate=true]  Pass false when the value is unchanged from the suggestion;
 *                                     the backend will save the fact without re-running the LLM.
 * @param {string} [category='character']  Fact category ('user', 'character', or 'setting').
 *                                         Should come from the evaluator's suggested_fact.category.
 * @returns {Promise<Response>}
 */
export function apiAcceptImplication(sessionId, turnId, key, value, regenerate = true, category = 'character') {
  return fetch(`/api/sessions/${sessionId}/turns/${turnId}/accept-implication`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key, value, regenerate, category }),
  });
}

/**
 * @param {number} sessionId
 * @param {number} turnId
 * @returns {Promise<Response>}
 */
export function apiIgnoreImplication(sessionId, turnId) {
  return fetch(`/api/sessions/${sessionId}/turns/${turnId}/ignore-implication`, {
    method: 'POST',
  });
}

/**
 * @param {number} sessionId
 * @param {number} turnId
 * @param {{ statement: string, derivation: string, source_fact_ids?: number[], inference_type?: string }} inference
 * @returns {Promise<Response>}
 */
export function apiAcceptInference(sessionId, turnId, inference) {
  return fetch(`/api/sessions/${sessionId}/turns/${turnId}/accept-inference`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      statement: inference.statement,
      derivation: inference.derivation,
      source_fact_ids: inference.source_fact_ids || [],
      inference_type: inference.inference_type || 'probabilistic',
    }),
  });
}

/**
 * @param {number} sessionId
 * @param {number} turnId
 * @returns {Promise<Response>}
 */
export function apiIgnoreInference(sessionId, turnId) {
  return fetch(`/api/sessions/${sessionId}/turns/${turnId}/ignore-inference`, {
    method: 'POST',
  });
}

/**
 * @param {number} characterId
 * @returns {Promise<Response>}
 */
export function apiGenerateInferences(characterId) {
  return fetch(`/api/characters/${characterId}/inferences/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
}

/**
 * @param {number} characterId
 * @param {number} changedFactId
 * @returns {Promise<Response>}
 */
export function apiRevalidateInferences(characterId, changedFactId) {
  return fetch(`/api/characters/${characterId}/inferences/revalidate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ changed_fact_id: changedFactId }),
  });
}

/**
 * @param {number} characterId
 * @param {number} inferenceId
 * @returns {Promise<Response>}
 */
export function apiDeleteInference(characterId, inferenceId) {
  return fetch(`/api/characters/${characterId}/inferences/${inferenceId}`, {
    method: 'DELETE',
  });
}

/**
 * @param {number} characterId
 * @param {number} inferenceId
 * @param {string} status
 * @returns {Promise<Response>}
 */
export function apiPatchInferenceStatus(characterId, inferenceId, status) {
  return fetch(`/api/characters/${characterId}/inferences/${inferenceId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status }),
  });
}

// ---------------------------------------------------------------------------
// Phase 4 API helpers — fact categories, mutability, and inference promotion
// ---------------------------------------------------------------------------

/**
 * @param {number} characterId
 * @param {string} key
 * @param {string} value
 * @param {string} [category='character']
 * @param {string} [mutability='immutable']
 * @returns {Promise<Response>}
 */
export function apiCreateFact(characterId, key, value, category = 'character', mutability = 'immutable') {
  return fetch(`/api/characters/${characterId}/facts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key, value, category, mutability }),
  });
}

/**
 * @param {number} characterId
 * @param {number} factId
 * @param {string} mutability
 * @returns {Promise<Response>}
 */
export function apiPatchFactMutability(characterId, factId, mutability) {
  return fetch(`/api/characters/${characterId}/facts/${factId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mutability }),
  });
}

/**
 * @param {number} characterId
 * @param {number} factId
 * @param {string} category
 * @returns {Promise<Response>}
 */
export function apiPatchFactCategory(characterId, factId, category) {
  return fetch(`/api/characters/${characterId}/facts/${factId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ category }),
  });
}

/**
 * @param {number} characterId
 * @param {number} inferenceId
 * @param {string} key
 * @param {string} value
 * @param {string} [category='character']
 * @param {string} [mutability='immutable']
 * @returns {Promise<Response>}
 */
export function apiPromoteInference(characterId, inferenceId, key, value, category = 'character', mutability = 'immutable') {
  return fetch(`/api/characters/${characterId}/inferences/${inferenceId}/promote`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key, value, category, mutability }),
  });
}

// ---------------------------------------------------------------------------
// Phase 5 API helpers — session end and experiences
// ---------------------------------------------------------------------------

/**
 * @param {number} sessionId
 * @returns {Promise<Response>}
 */
export function apiEndSession(sessionId) {
  return fetch(`/api/sessions/${sessionId}/end`, { method: 'POST' });
}

/**
 * @param {number} characterId
 * @param {number} sessionId
 * @param {string} statement
 * @param {string} source  'told_by_user' | 'observed'
 * @returns {Promise<Response>}
 */
export function apiCreateExperience(characterId, sessionId, statement, source) {
  return fetch(`/api/characters/${characterId}/experiences`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, statement, source }),
  });
}

/**
 * @param {number} characterId
 * @returns {Promise<Response>}
 */
export function apiListExperiences(characterId) {
  return fetch(`/api/characters/${characterId}/experiences`);
}

/**
 * @param {number} characterId
 * @param {number} experienceId
 * @returns {Promise<Response>}
 */
export function apiDeleteExperience(characterId, experienceId) {
  return fetch(`/api/characters/${characterId}/experiences/${experienceId}`, {
    method: 'DELETE',
  });
}

// ---------------------------------------------------------------------------
// Phase 5 — experience sorting
// ---------------------------------------------------------------------------

/**
 * Sort experiences for display: active first (by score desc), then inactive (by score desc),
 * then any without a score at the bottom.
 *
 * @param {object[]} experiences
 * @param {Set<number>} activeIds
 * @param {Map<number, number>} scoreMap
 * @returns {object[]}
 */
export function sortExperiences(experiences, activeIds, scoreMap) {
  return experiences.slice().sort((a, b) => {
    const aActive = activeIds.has(a.id) ? 1 : 0;
    const bActive = activeIds.has(b.id) ? 1 : 0;
    if (aActive !== bActive) return bActive - aActive;
    const aScore = scoreMap.get(a.id) ?? -Infinity;
    const bScore = scoreMap.get(b.id) ?? -Infinity;
    // Guard against -Infinity - (-Infinity) = NaN when both lack a score.
    if (aScore === bScore) return 0;
    return bScore - aScore;
  });
}

// ---------------------------------------------------------------------------
// Phase 5 — score map construction from SSE payload
// ---------------------------------------------------------------------------

/**
 * Convert the `experience_scores` array from the SSE `message` event into the
 * Map<number, number> required by sortExperiences.
 *
 * @param {Array<{id: number, score: number}>} experienceScores
 * @returns {Map<number, number>}
 */
export function buildScoreMap(experienceScores) {
  return new Map(experienceScores.map(s => [s.id, s.score]));
}

// ---------------------------------------------------------------------------
// Phase 5 — proposal list and experience-update helpers
// ---------------------------------------------------------------------------

/**
 * Wrap raw proposed-experience objects from the session-end API response with
 * the UI state fields the review panel needs.
 *
 * @param {Array<{statement: string, source: string, turn_reference?: number}>} proposedExperiences
 * @returns {object[]}
 */
export function buildProposalList(proposedExperiences) {
  return (proposedExperiences || []).map(p => ({
    ...p,
    _editing: false,
    _editStatement: '',
    _loading: false,
  }));
}

/**
 * Remove experiences that were contradicted by an `experience_update` verdict.
 * Returns a new array; does not mutate the input.
 *
 * @param {object[]} experiences  current approved experiences list
 * @param {object}   notification experience_update notification from buildNotificationFromSidechannel
 * @returns {object[]}
 */
export function removeContradictedExperiences(experiences, notification) {
  const deletedIds = new Set(
    (notification.experience_updates || []).map(u => u.contradicted_experience_id),
  );
  return experiences.filter(e => !deletedIds.has(e.id));
}
