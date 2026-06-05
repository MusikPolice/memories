import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  parseSSEBlock,
  parseSSEBlocks,
  sseStateToLabel,
  buildNotificationFromSidechannel,
  apiAcceptImplication,
  apiIgnoreImplication,
  apiAcceptInference,
  apiIgnoreInference,
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
// API helpers
// ---------------------------------------------------------------------------

describe('API helpers', () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({ ok: true });
  });

  it('apiAcceptImplication POSTs to the correct URL with key and value', async () => {
    await apiAcceptImplication(5, 3, 'siblings', 'one sister');
    expect(fetch).toHaveBeenCalledWith(
      '/api/sessions/5/turns/3/accept-implication',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ key: 'siblings', value: 'one sister' }),
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
