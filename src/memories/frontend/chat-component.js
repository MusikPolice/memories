import { ref, computed, nextTick } from 'vue';
import {
  parseSSEBlock,
  sseStateToLabel,
  buildNotificationFromSidechannel,
  apiEndSession,
  apiCreateExperience,
  apiDeleteExperience,
  buildScoreMap,
  buildProposalList,
  sortExperiences,
  apiDeleteInference,
  buildVisibleFactRows,
  schemaLeaf,
  apiGetSchema,
  apiGetFactBlob,
  apiSetFactValue,
  apiRespondRequireFact,
  apiRespondSetFact,
} from './chat.js';

export const ChatComponent = {
  setup() {
    const showPicker = ref(true);
    const characters = ref([]);
    const currentCharacter = ref(null);
    const sessionId = ref(null);
    const messages = ref([]);
    const inferences = ref([]);
    const inputText = ref('');
    const sending = ref(false);
    const generating = ref(false);
    const statusText = ref('');
    const sessionEnded = ref(false);
    const reviewingSession = ref(false);
    const experiences = ref([]);
    const activeExperienceIds = ref(new Set());
    const experienceScoreMap = ref(new Map());
    const sessionProposals = ref([]);
    const thinkEnabled = ref(false);
    const messagesEl = ref(null);
    const inputEl = ref(null);

    // ── Fact schema / blob / tree ──

    const schema = ref({});
    const factsBlob = ref({});
    const collapsedGroups = ref(new Set());
    const leafEdits = ref({});

    async function loadSchema() {
      const r = await apiGetSchema();
      if (r.ok) schema.value = await r.json();
    }

    async function loadFacts() {
      if (!currentCharacter.value) return;
      const r = await apiGetFactBlob(currentCharacter.value.id);
      factsBlob.value = r.ok ? await r.json() : {};
      syncLeafEdits();
    }

    function syncLeafEdits() {
      const edits = {};
      for (const row of buildVisibleFactRows(schema.value, factsBlob.value, new Set())) {
        if (row.isLeaf) edits[row.path] = row.value ?? '';
      }
      leafEdits.value = edits;
    }

    const visibleFactRows = computed(() =>
      buildVisibleFactRows(schema.value, factsBlob.value, collapsedGroups.value)
    );

    function toggleGroup(path) {
      const next = new Set(collapsedGroups.value);
      next.has(path) ? next.delete(path) : next.add(path);
      collapsedGroups.value = next;
    }

    function leafType(path) { return schemaLeaf(schema.value, path)?.type ?? 'String'; }
    function leafConstraint(path) { return schemaLeaf(schema.value, path)?.constraint ?? []; }

    async function saveLeaf(row) {
      const value = leafEdits.value[row.path] ?? '';
      await apiSetFactValue(currentCharacter.value.id, row.path, value);
      await loadFacts();
    }

    // ── Characters / sessions ──

    async function loadCharacters() {
      const r = await fetch('/api/characters/');
      characters.value = await r.json();
      if (characters.value.length === 1) {
        pickCharacter(characters.value[0]);
      }
    }

    async function loadExperiences() {
      if (!currentCharacter.value) return;
      const r = await fetch(`/api/characters/${currentCharacter.value.id}/experiences`);
      experiences.value = await r.json();
    }

    async function pickCharacter(ch) {
      currentCharacter.value = ch;
      const r = await fetch('/api/sessions/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ character_id: ch.id }),
      });
      const data = await r.json();
      sessionId.value = data.session.id;
      showPicker.value = false;
      if (data.previous_journal) {
        messages.value.push({ role: 'journal', content: data.previous_journal, name: ch.name });
      }
      await Promise.all([loadFacts(), loadInferences(), loadExperiences()]);
      nextTick(() => inputEl.value?.focus());
    }

    async function loadInferences() {
      if (!currentCharacter.value) return;
      const r = await fetch(`/api/characters/${currentCharacter.value.id}/inferences`);
      const raw = await r.json();
      inferences.value = raw.map(inf => ({ ...inf, _expanded: false }));
    }

    async function deleteInference(inf) {
      await apiDeleteInference(currentCharacter.value.id, inf.id);
      await loadInferences();
    }

    async function endSession() {
      reviewingSession.value = true;
      sessionEnded.value = true;
      activeExperienceIds.value = new Set();
      experienceScoreMap.value = new Map();
      try {
        const r = await apiEndSession(sessionId.value);
        if (r.ok) {
          const data = await r.json();
          const proposals = buildProposalList(data.proposed_experiences);
          sessionProposals.value = proposals;
          if (proposals.length === 0) await loadExperiences();
        }
      } finally {
        reviewingSession.value = false;
      }
    }

    async function acceptProposal(p) {
      p._loading = true;
      try {
        const r = await apiCreateExperience(
          currentCharacter.value.id, sessionId.value, p.statement, p.source,
        );
        if (r.ok) {
          const idx = sessionProposals.value.indexOf(p);
          if (idx !== -1) sessionProposals.value.splice(idx, 1);
          if (sessionProposals.value.length === 0) await loadExperiences();
        }
      } finally {
        p._loading = false;
      }
    }

    async function confirmEditProposal(p) {
      const edited = p._editStatement.trim();
      if (!edited) return;
      p._loading = true;
      try {
        const r = await apiCreateExperience(
          currentCharacter.value.id, sessionId.value, edited, p.source,
        );
        if (r.ok) {
          const idx = sessionProposals.value.indexOf(p);
          if (idx !== -1) sessionProposals.value.splice(idx, 1);
          if (sessionProposals.value.length === 0) await loadExperiences();
        }
      } finally {
        p._loading = false;
      }
    }

    function discardProposal(i) {
      sessionProposals.value.splice(i, 1);
      if (sessionProposals.value.length === 0) loadExperiences();
    }

    async function deleteExperience(exp) {
      await apiDeleteExperience(currentCharacter.value.id, exp.id);
      await loadExperiences();
    }

    async function newSession() {
      messages.value = [];
      factsBlob.value = {};
      collapsedGroups.value = new Set();
      leafEdits.value = {};
      inferences.value = [];
      experiences.value = [];
      sessionProposals.value = [];
      activeExperienceIds.value = new Set();
      experienceScoreMap.value = new Map();
      reviewingSession.value = false;
      inputText.value = '';
      sending.value = false;
      generating.value = false;
      statusText.value = '';
      sessionEnded.value = false;
      sessionId.value = null;
      currentCharacter.value = null;
      if (characters.value.length === 1) {
        await pickCharacter(characters.value[0]);
      } else {
        showPicker.value = true;
      }
    }

    // ── Notification actions ──

    function dismissNotification(notif) {
      const idx = messages.value.indexOf(notif);
      if (idx !== -1) messages.value.splice(idx, 1);
    }

    async function resolveRequireFact(notif, confirmed) {
      notif._loading = true;
      try {
        const value = confirmed ? (notif._editValue ?? '') : null;
        await apiRespondRequireFact(sessionId.value, notif.turn_id, value);
        dismissNotification(notif);
      } finally { notif._loading = false; }
    }

    async function resolveMutable(notif, action) {
      notif._loading = true;
      try {
        const value = action === 'edit' ? (notif._editValue ?? '') : null;
        await apiRespondSetFact(sessionId.value, notif.turn_id, action, value);
        dismissNotification(notif);
      } finally { notif._loading = false; }
    }

    async function resolveImmutable(notif, action) {
      notif._loading = true;
      try {
        const value = action === 'edit' ? (notif._editValue ?? '') : null;
        await apiRespondSetFact(sessionId.value, notif.turn_id, action, value);
        dismissNotification(notif);
      } finally { notif._loading = false; }
    }

    // ── Chat ──

    async function sendMessage() {
      const text = inputText.value.trim();
      if (!text || sending.value || sessionEnded.value) return;
      inputText.value = '';
      sending.value = true;
      generating.value = true;
      statusText.value = 'Generating response…';
      messages.value.push({ role: 'user', content: text });
      await scrollToBottom();

      try {
        const response = await fetch(`/api/sessions/${sessionId.value}/messages`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: text, think: thinkEnabled.value }),
        });

        if (!response.ok) {
          messages.value.push({ role: 'assistant', content: `[Error ${response.status}]` });
          return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          const completedBlocks = buffer.split('\n\n');
          buffer = completedBlocks.pop() ?? '';

          for (const block of completedBlocks) {
            const parsed = parseSSEBlock(block);
            if (!parsed) continue;
            const { event: eventName, data: dataStr } = parsed;

            if (eventName === 'status' && dataStr) {
              statusText.value = sseStateToLabel(JSON.parse(dataStr).state);

            } else if (eventName === 'thinking' && dataStr) {
              generating.value = false;
              messages.value.push({
                role: 'thinking', content: JSON.parse(dataStr).content, _open: false,
              });
              await scrollToBottom();

            } else if (eventName === 'message' && dataStr) {
              const payload = JSON.parse(dataStr);
              generating.value = false;
              statusText.value = '';
              messages.value.push({
                role: 'assistant',
                content: payload.content,
                turn_id: payload.turn_id,
                contradictionExhausted: payload.contradiction_exhausted || false,
              });
              if (payload.active_experience_ids) {
                activeExperienceIds.value = new Set(payload.active_experience_ids);
              }
              if (payload.experience_scores) {
                experienceScoreMap.value = buildScoreMap(payload.experience_scores);
              }
              await scrollToBottom();

            } else if (eventName === 'sidechannel' && dataStr) {
              const notif = buildNotificationFromSidechannel(JSON.parse(dataStr));
              if (notif) {
                const blocking = [
                  'fact_update_mutable', 'fact_update_immutable_unset', 'require_fact',
                ];
                if (blocking.includes(notif.scType)) {
                  generating.value = false;
                  statusText.value = '';
                }
                messages.value.push(notif);
                await scrollToBottom();
              }

            } else if (eventName === 'done') {
              if (currentCharacter.value) {
                await loadFacts();
                await loadInferences();
              }
            }
          }
        }
      } finally {
        generating.value = false;
        statusText.value = '';
        sending.value = false;
        await nextTick();
        inputEl.value?.focus();
      }
    }

    async function scrollToBottom() {
      await nextTick();
      if (messagesEl.value) {
        messagesEl.value.scrollTop = messagesEl.value.scrollHeight;
      }
    }

    const sortedExperiences = computed(() =>
      sortExperiences(experiences.value, activeExperienceIds.value, experienceScoreMap.value)
    );

    loadSchema();
    loadCharacters();

    return {
      showPicker, characters, currentCharacter, sessionId,
      messages, inferences,
      schema, factsBlob, visibleFactRows, collapsedGroups, leafEdits,
      experiences, sortedExperiences, activeExperienceIds, experienceScoreMap, sessionProposals,
      reviewingSession,
      inputText, sending, generating, statusText, sessionEnded,
      thinkEnabled, messagesEl, inputEl,
      loadSchema, loadFacts, loadInferences,
      toggleGroup, leafType, leafConstraint, saveLeaf,
      deleteInference,
      pickCharacter,
      endSession, newSession, sendMessage,
      dismissNotification,
      resolveRequireFact, resolveMutable, resolveImmutable,
      acceptProposal, confirmEditProposal, discardProposal, deleteExperience,
    };
  },
};
