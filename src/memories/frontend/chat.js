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
 * @returns {Promise<Response>}
 */
export function apiAcceptImplication(sessionId, turnId, key, value) {
  return fetch(`/api/sessions/${sessionId}/turns/${turnId}/accept-implication`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key, value }),
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
