import { ref, computed, nextTick, onMounted, onUnmounted } from 'vue';
import {
  parseSSEBlock,
  sseStateToLabel,
  buildNotificationFromSidechannel,
  removeViolation,
  apiAcceptImplication,
  apiIgnoreImplication,
  apiAcceptInference,
  apiIgnoreInference,
  apiCreateFact,
  apiPatchFactMutability,
  apiPatchFactCategory,
  apiPromoteInference,
  apiEndSession,
  apiCreateExperience,
  apiDeleteExperience,
  buildScoreMap,
  buildProposalList,
  removeContradictedExperiences,
  sortExperiences,
} from './chat.js';

export const ChatComponent = {
  setup() {
    const showPicker = ref(true);
    const characters = ref([]);
    const currentCharacter = ref(null);
    const sessionId = ref(null);
    const messages = ref([]);
    const facts = ref([]);
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
    const newKey = ref('');
    const newValue = ref('');
    const newCategory = ref('character');
    const newMutability = ref('immutable');
    const factError = ref('');
    const messagesEl = ref(null);
    const inputEl = ref(null);

    // ── Fact helpers ──

    function mutabilityIcon(m) {
      return m === 'low' ? '📌' : m === 'high' ? '💧' : '🔒';
    }

    const factsByCategory = computed(() => ({
      user: facts.value.filter(f => f.category === 'user'),
      character: facts.value.filter(f => f.category === 'character'),
      setting: facts.value.filter(f => f.category === 'setting'),
    }));

    function closeAllMutDropdowns() {
      facts.value.forEach(f => { if (f._mutOpen) f._mutOpen = false; });
    }

    function closeAllCatDropdowns() {
      facts.value.forEach(f => { if (f._catOpen) f._catOpen = false; });
    }

    onMounted(() => {
      document.addEventListener('click', closeAllMutDropdowns);
      document.addEventListener('click', closeAllCatDropdowns);
    });
    onUnmounted(() => {
      document.removeEventListener('click', closeAllMutDropdowns);
      document.removeEventListener('click', closeAllCatDropdowns);
    });

    function toggleMutability(fact) {
      const wasOpen = fact._mutOpen;
      closeAllMutDropdowns();
      fact._mutOpen = !wasOpen;
    }

    function toggleCategory(fact) {
      const wasOpen = fact._catOpen;
      closeAllCatDropdowns();
      fact._catOpen = !wasOpen;
    }

    async function patchCategory(fact, newCat) {
      fact._catOpen = false;
      fact._catError = '';
      if (newCat === fact.category) return;
      const r = await apiPatchFactCategory(currentCharacter.value.id, fact.id, newCat);
      if (r.ok) {
        fact.category = newCat;
      } else if (r.status === 409) {
        fact._catError = `A ${newCat} fact '${fact.key}' already exists.`;
      }
    }

    async function patchMutability(fact, newMut) {
      fact._mutOpen = false;
      if (newMut === fact.mutability) return;
      const r = await apiPatchFactMutability(currentCharacter.value.id, fact.id, newMut);
      if (r.ok) {
        fact.mutability = newMut;
      }
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

    async function loadFacts() {
      if (!currentCharacter.value) return;
      const r = await fetch(`/api/characters/${currentCharacter.value.id}/facts`);
      const raw = await r.json();
      facts.value = raw.map(f => ({ ...f, _editValue: f.value, _mutOpen: false, _catOpen: false, _catError: '' }));
    }

    async function loadInferences() {
      if (!currentCharacter.value) return;
      const r = await fetch(`/api/characters/${currentCharacter.value.id}/inferences`);
      const raw = await r.json();
      inferences.value = raw.map(inf => ({
        ...inf,
        _expanded: false,
        _promoteOpen: false,
        _promoteKey: '',
        _promoteValue: inf.statement,
        _promoteCategory: 'character',
        _promoteMutability: 'immutable',
        _promoteError: '',
        _promoteLoading: false,
      }));
    }

    async function addFact() {
      factError.value = '';
      if (!newKey.value.trim() || !newValue.value.trim()) return;
      const r = await apiCreateFact(
        currentCharacter.value.id,
        newKey.value.trim(),
        newValue.value.trim(),
        newCategory.value,
        newMutability.value,
      );
      if (r.status === 409) {
        factError.value = `A ${newCategory.value} fact '${newKey.value}' already exists.`;
        return;
      }
      newKey.value = '';
      newValue.value = '';
      newCategory.value = 'character';
      newMutability.value = 'immutable';
      await loadFacts();
    }

    async function saveFact(fact) {
      await fetch(`/api/characters/${currentCharacter.value.id}/facts/${fact.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ value: fact._editValue }),
      });
      await loadFacts();
      await loadInferences();
    }

    async function deleteFact(fact) {
      await fetch(`/api/characters/${currentCharacter.value.id}/facts/${fact.id}`, {
        method: 'DELETE',
      });
      await loadFacts();
      await loadInferences();
    }

    // ── Inference promote ──

    function togglePromote(inf) {
      if (!inf._promoteOpen) {
        inf._promoteKey = '';
        inf._promoteValue = inf.statement;
        inf._promoteCategory = 'character';
        inf._promoteMutability = 'immutable';
        inf._promoteError = '';
        inf._promoteLoading = false;
      }
      inf._promoteOpen = !inf._promoteOpen;
    }

    async function promoteInference(inf) {
      if (!inf._promoteKey?.trim()) return;
      inf._promoteLoading = true;
      inf._promoteError = '';
      try {
        const r = await apiPromoteInference(
          currentCharacter.value.id,
          inf.id,
          inf._promoteKey.trim(),
          inf._promoteValue,
          inf._promoteCategory,
          inf._promoteMutability,
        );
        if (r.status === 201) {
          const data = await r.json();
          const idx = inferences.value.indexOf(inf);
          if (idx !== -1) inferences.value.splice(idx, 1);
          const newFact = { ...data.fact, _editValue: data.fact.value, _mutOpen: false };
          facts.value.push(newFact);
        } else if (r.status === 409) {
          inf._promoteError = 'A fact with this key already exists in that category.';
        }
      } finally {
        inf._promoteLoading = false;
      }
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
      facts.value = [];
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

    async function acceptImplication(notif, violation) {
      if (!violation.suggested_fact) return;
      violation._loading = true;
      const value = violation._editValue ?? violation.suggested_fact.value;
      const regenerate = value !== violation.suggested_fact.value;
      try {
        const r = await apiAcceptImplication(
          sessionId.value, notif.turn_id, violation.suggested_fact.key, value, regenerate,
          violation.suggested_fact?.category ?? 'character'
        );
        if (r.ok) {
          const data = await r.json();
          const assistantMsg = messages.value.find(
            m => m.role === 'assistant' && m.turn_id === notif.turn_id
          );
          if (assistantMsg) assistantMsg.content = data.content;
          if (removeViolation(notif, violation)) dismissNotification(notif);
          await loadFacts();
        }
      } finally {
        violation._loading = false;
      }
    }

    async function ignoreImplication(notif, violation) {
      if (removeViolation(notif, violation)) {
        await apiIgnoreImplication(sessionId.value, notif.turn_id);
        dismissNotification(notif);
      }
    }

    async function acceptInference(notif, inference) {
      inference._loading = true;
      try {
        const r = await apiAcceptInference(sessionId.value, notif.turn_id, inference);
        if (r.ok) {
          const idx = notif.new_inferences.indexOf(inference);
          if (idx !== -1) notif.new_inferences.splice(idx, 1);
          if (notif.new_inferences.length === 0) dismissNotification(notif);
          await loadInferences();
        }
      } finally {
        inference._loading = false;
      }
    }

    async function ignoreInference(notif, inference) {
      inference._loading = true;
      try {
        const idx = notif.new_inferences.indexOf(inference);
        if (idx !== -1) notif.new_inferences.splice(idx, 1);
        if (notif.new_inferences.length === 0) {
          await apiIgnoreInference(sessionId.value, notif.turn_id);
          dismissNotification(notif);
        }
      } finally {
        inference._loading = false;
      }
    }

    function dismissNotification(notif) {
      const idx = messages.value.indexOf(notif);
      if (idx !== -1) messages.value.splice(idx, 1);
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
                if (notif.scType === 'experience_update') {
                  experiences.value = removeContradictedExperiences(experiences.value, notif);
                }
                messages.value.push(notif);
                await scrollToBottom();
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

    loadCharacters();

    return {
      showPicker, characters, currentCharacter, sessionId,
      messages, facts, inferences, factsByCategory,
      experiences, sortedExperiences, activeExperienceIds, experienceScoreMap, sessionProposals,
      reviewingSession,
      inputText, sending, generating, statusText, sessionEnded,
      thinkEnabled, newKey, newValue, newCategory, newMutability,
      factError, messagesEl, inputEl,
      mutabilityIcon,
      pickCharacter, addFact, saveFact, deleteFact,
      toggleMutability, patchMutability,
      toggleCategory, patchCategory,
      togglePromote, promoteInference,
      endSession, newSession, sendMessage,
      acceptImplication, ignoreImplication, acceptInference, ignoreInference,
      dismissNotification,
      acceptProposal, confirmEditProposal, discardProposal, deleteExperience,
    };
  },
};
