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
  extracting: 'Extracting facts from conversation…',
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
  if (payload.type === 'contradiction') {
    return {
      role: 'notification',
      scType: 'contradiction',
      iteration: payload.iteration,
      description: payload.description,
    };
  }

  if (payload.type === 'fact_update_fluid') {
    return {
      role: 'notification', scType: 'fact_update_fluid',
      turn_id: payload.turn_id, path: payload.path, value: payload.value,
    };
  }

  if (payload.type === 'fact_update_mutable') {
    return {
      role: 'notification', scType: 'fact_update_mutable',
      turn_id: payload.turn_id, path: payload.path, proposed: payload.proposed,
      _editValue: payload.proposed ?? '', _loading: false,
    };
  }

  if (payload.type === 'fact_update_immutable_unset') {
    return {
      role: 'notification', scType: 'fact_update_immutable_unset',
      turn_id: payload.turn_id, path: payload.path, proposed: payload.proposed,
      _editValue: payload.proposed ?? '', _loading: false,
    };
  }

  if (payload.type === 'require_fact') {
    return {
      role: 'notification', scType: 'require_fact',
      turn_id: payload.turn_id, path: payload.path, reason: payload.reason,
      _editValue: payload.suggested_value ?? '', _loading: false,
    };
  }

  if (payload.type === 'inference_proposed') {
    return {
      role: 'notification', scType: 'inference_proposed',
      turn_id: payload.turn_id, inference: payload.inference,
    };
  }

  return null;
}

// ---------------------------------------------------------------------------
// Schema-tree pure helpers
// ---------------------------------------------------------------------------

/**
 * Flatten a fact schema tree into an ordered, depth-first row list.
 * A node with a `Type` key is a leaf; any other node is a grouping.
 * @param {object} schema
 * @param {string} [prefix='']
 * @param {number} [depth=0]
 * @returns {Array<{path,name,depth,isLeaf,type?,mutability?,constraint?,description?}>}
 */
export function flattenSchema(schema, prefix = '', depth = 0) {
  const rows = [];
  for (const [name, node] of Object.entries(schema || {})) {
    const path = prefix ? `${prefix}.${name}` : name;
    if (node && typeof node === 'object' && 'Type' in node) {
      rows.push({
        path, name, depth, isLeaf: true,
        type: node.Type, mutability: node.Mutability,
        constraint: node.Constraint ?? null, description: node.Description ?? '',
      });
    } else {
      rows.push({ path, name, depth, isLeaf: false });
      rows.push(...flattenSchema(node, path, depth + 1));
    }
  }
  return rows;
}

/**
 * Read a leaf value out of a fact blob by dot-notation path.
 * @returns the stored Value, or undefined if the path is unset.
 */
export function lookupBlobValue(blob, path) {
  let node = blob;
  for (const part of path.split('.')) {
    if (!node || typeof node !== 'object' || !(part in node)) return undefined;
    node = node[part];
  }
  return (node && typeof node === 'object') ? node.Value : undefined;
}

/**
 * Produce the visible fact-tree rows: flatten the schema, drop rows whose ancestor
 * grouping is collapsed, and attach the current value to each leaf.
 * @param {object} schema
 * @param {object} blob
 * @param {Set<string>} collapsedPaths  grouping paths whose children are hidden
 * @returns {Array<object>}  rows with `value` set on leaves (undefined if unset)
 */
export function buildVisibleFactRows(schema, blob, collapsedPaths) {
  return flattenSchema(schema)
    .filter(row => {
      for (const c of collapsedPaths) {
        if (row.path !== c && row.path.startsWith(c + '.')) return false;
      }
      return true;
    })
    .map(row => (row.isLeaf ? { ...row, value: lookupBlobValue(blob, row.path) } : row));
}

/** Look up a single leaf's schema definition by path (for type-aware card inputs). */
export function schemaLeaf(schema, path) {
  return flattenSchema(schema).find(r => r.isLeaf && r.path === path) ?? null;
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

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
// Facts v2 API helpers — schema, blob read/write, blocking-card responses
// ---------------------------------------------------------------------------

export function apiGetSchema() {
  return fetch('/api/schema');
}

export function apiGetFactBlob(characterId) {
  return fetch(`/api/characters/${characterId}/facts`);
}

export function apiSetFactValue(characterId, path, value) {
  return fetch(`/api/characters/${characterId}/facts`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path, value }),
  });
}

export function apiRespondRequireFact(sessionId, turnId, value) {
  return fetch(`/api/sessions/${sessionId}/turns/${turnId}/require-fact/respond`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ value }),
  });
}

export function apiRespondSetFact(sessionId, turnId, action, value = null) {
  return fetch(`/api/sessions/${sessionId}/turns/${turnId}/set-fact/respond`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, value }),
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

