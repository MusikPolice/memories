# Frontend Testing Strategy

## Problem Statement

The current test suite covers `chat.js` pure functions exhaustively (100% statement
coverage) but cannot test anything in `index.html`. This is a structural gap: `index.html`
is a monolithic file — Vue template, component logic, and styles all in one — with no
exports, so Vitest cannot import it.

Three bugs survived code review and the full test suite because of this gap:

| Bug | Class | Detectable by current tests? |
|-----|-------|------------------------------|
| Missing `experience_update` notification card template | Template gap | No — requires rendering |
| `activeExperienceIds`/`experienceScoreMap` not reset in `endSession()` | State logic gap | No — requires component test |
| Deleted experience not removed from `experiences[]` after `experience_update` | State logic gap | No — requires component test |

The fix for bugs 2 and 3 was to extract helper functions (`buildProposalList`,
`removeContradictedExperiences`) into `chat.js` so the logic became testable. Bug 1
(missing template block) required reading the code manually; no test could have caught it
under the current setup.

---

## Current Test Infrastructure

```
package.json            # type: "module"; devDeps: vitest, jsdom, @vitest/coverage-v8, eslint
vitest.config.js        # environment: jsdom; include: tests/frontend/**/*.test.js
                        # coverage.include: src/memories/frontend/chat.js only
                        # coverage threshold: 80% lines/functions/branches/statements
tests/frontend/
  chat.test.js          # ~116 tests; imports named exports from chat.js
src/memories/frontend/
  chat.js               # pure functions: SSE parsing, notification building, API helpers,
                        # sort helpers — all exported, all testable
  index.html            # Vue 3 CDN app; imports chat.js via ES module; not testable
```

`vue` is **not** currently in `package.json`. The app uses Vue 3 via CDN
(`<script src="https://unpkg.com/vue@3/dist/vue.global.prod.js">`), which exposes a
`Vue` global. The `index.html` destructures from that global:
```javascript
const { createApp, ref, computed, nextTick, onMounted, onUnmounted } = Vue;
```

---

## Option A — Extract Component Logic (Recommended)

### What it achieves

- Tests all reactive state management: `ref()` values, `computed()` values, and every
  method that modifies them
- Catches state logic bugs (classes 2 and 3 above) without a running server
- Same test runner (Vitest), same test file convention, no new tooling
- Does **not** catch missing template blocks (class 1 above) — that requires rendering

### Approach: importmap + extracted component module

The production app switches from the CDN UMD bundle (`vue.global.prod.js`) to the CDN
ESM bundle (`vue.esm-browser.prod.js`) via an HTML importmap. This makes `import { ref }
from 'vue'` work in both the browser (resolved to CDN) and in Vitest (resolved to
`node_modules/vue`).

**Step 1 — Add `vue` to devDependencies**

```bash
npm install --save-dev vue
```

No other package changes are needed.

**Step 2 — Add an importmap to `index.html`**

Replace the existing `<script src="https://unpkg.com/vue@3/dist/vue.global.prod.js">` tag:

```html
<!-- BEFORE -->
<script src="https://unpkg.com/vue@3/dist/vue.global.prod.js"></script>
```

```html
<!-- AFTER -->
<script type="importmap">
{
  "imports": {
    "vue": "https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js"
  }
}
</script>
```

Then change the `<script type="module">` tag at the bottom of `index.html` — the
`const { createApp, ref, ... } = Vue;` destructure becomes:

```javascript
import { createApp, ref, computed, nextTick, onMounted, onUnmounted } from 'vue';
```

**Step 3 — Create `src/memories/frontend/chat-component.js`**

Extract the entire `setup()` function body and its helper functions from `index.html` into
a new file. The component is exported as a plain Vue options object:

```javascript
// chat-component.js
import {
  ref, computed, nextTick, onMounted, onUnmounted,
} from 'vue';
import {
  parseSSEBlock, sseStateToLabel, buildNotificationFromSidechannel, removeViolation,
  apiAcceptImplication, apiIgnoreImplication, apiAcceptInference, apiIgnoreInference,
  apiCreateFact, apiPatchFactMutability, apiPatchFactCategory, apiPromoteInference,
  apiEndSession, apiCreateExperience, apiDeleteExperience,
  buildScoreMap, buildProposalList, removeContradictedExperiences, sortExperiences,
} from './chat.js';

export const ChatComponent = {
  setup() {
    // ... entire setup() body from index.html, unchanged ...
    return { /* all the existing return values */ };
  },
};
```

**Step 4 — Slim down `index.html`**

`index.html` keeps only: the `<style>` block, the HTML template (`<div id="app">...`),
and a thin bootstrap script:

```javascript
import { createApp } from 'vue';
import { ChatComponent } from './chat-component.js';
createApp(ChatComponent).mount('#app');
```

**Step 5 — Add `chat-component.js` to Vitest coverage**

Update `vitest.config.js`:

```javascript
coverage: {
  include: [
    'src/memories/frontend/chat.js',
    'src/memories/frontend/chat-component.js',
  ],
  // thresholds unchanged
}
```

**Step 6 — Write component tests**

Create `tests/frontend/chat-component.test.js`. Use `@vue/test-utils` to mount the
component (add `npm install --save-dev @vue/test-utils`), OR test the `setup()` function
directly against Vue's reactivity system without rendering:

```javascript
// Option: test setup() directly (no @vue/test-utils needed)
import { ChatComponent } from '../../src/memories/frontend/chat-component.js';

// Call setup() directly; Vue ref/computed work because 'vue' is in node_modules.
// Do NOT call onMounted/onUnmounted — they are no-ops outside a component tree.
// Wrap calls that trigger onMounted in a component mount if needed.
```

Using `@vue/test-utils` (required to test lifecycle hooks and template-dependent refs
like `messagesEl`):

```javascript
import { mount } from '@vue/test-utils';
import { ChatComponent } from '../../src/memories/frontend/chat-component.js';

// mount() runs onMounted, wires up reactive dependencies, and provides a
// wrapper.vm to call methods and inspect state.
```

**Tests to write (priority order)**

These correspond directly to the three bug classes described above:

1. **`endSession()` state management**
   - `endSession_sets_sessionEnded_true`
   - `endSession_resets_activeExperienceIds_to_empty_set`
   - `endSession_resets_experienceScoreMap_to_empty_map`
   - `endSession_sets_sessionProposals_from_response`
   - `endSession_calls_loadExperiences_when_no_proposals`
   - `endSession_sets_reviewingSession_false_after_response`
   - `endSession_sets_reviewingSession_false_on_api_failure` (finally block)

2. **SSE `sidechannel` handler — `experience_update`**
   - `sendMessage_experience_update_sidechannel_removes_experience_from_list`
   - `sendMessage_experience_update_sidechannel_pushes_notification_to_messages`
   - `sendMessage_experience_update_sidechannel_calls_removeContradictedExperiences`

3. **SSE `message` handler — active experience tracking**
   - `sendMessage_message_event_updates_activeExperienceIds`
   - `sendMessage_message_event_calls_buildScoreMap`
   - `sendMessage_message_event_updates_experienceScoreMap`

4. **`newSession()` reset**
   - `newSession_resets_experiences_to_empty`
   - `newSession_resets_sessionProposals_to_empty`
   - `newSession_resets_activeExperienceIds_to_empty_set`
   - `newSession_resets_experienceScoreMap_to_empty_map`

5. **Proposal lifecycle**
   - `acceptProposal_calls_apiCreateExperience`
   - `acceptProposal_removes_proposal_from_sessionProposals`
   - `acceptProposal_calls_loadExperiences_when_last_proposal_accepted`
   - `confirmEditProposal_calls_apiCreateExperience_with_edited_text`
   - `discardProposal_removes_proposal_without_api_call`
   - `deleteExperience_calls_apiDeleteExperience`
   - `deleteExperience_reloads_experiences_after_delete`

**Mocking strategy**

The component methods call `fetch` (for API calls) and navigate via `router` (none here).
Mock `fetch` with `vi.stubGlobal('fetch', vi.fn())` in `beforeEach`, same as the existing
`chat.test.js` pattern:

```javascript
beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ /* fixture data */ }),
  }));
});
```

**What importmap support requires**

Importmaps are supported in all modern browsers (Chrome 89+, Firefox 108+, Safari 16.4+).
The app already targets localhost only, so browser compatibility is not a concern.
Vitest resolves `'vue'` from `node_modules` and ignores the importmap entirely — no
Vitest configuration change is needed for the import resolution.

---

## Option B — Playwright End-to-End Tests

### What it achieves

- Tests the real rendered output: catches missing `v-else-if` blocks, CSS display bugs,
  and interaction flows that span template + logic
- Exercises the actual SSE streaming pipeline
- Catches regressions that require a running server (DB, Ollama mock) to reproduce

### What it requires

- `@playwright/test` package
- A test-mode FastAPI server (same process or subprocess)
- An Ollama stub that speaks the NDJSON streaming protocol

### Approach

**Step 1 — Add Playwright**

```bash
npm install --save-dev @playwright/test
npx playwright install chromium  # headless Chromium only; others not needed
```

Add to `package.json` scripts:
```json
"test:e2e": "playwright test",
"test:e2e:ui": "playwright test --ui"
```

**Step 2 — Playwright config**

`playwright.config.js` at repo root:

```javascript
import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: 'tests/e2e',
  use: {
    baseURL: 'http://localhost:8001',  // separate port from dev server
  },
  webServer: {
    command: 'uv run uvicorn memories.main:app --port 8001',
    port: 8001,
    reuseExistingServer: false,
    env: {
      MEMORIES_DB_PATH: ':memory:',       // in-memory DB, isolated per run
      OLLAMA_BASE_URL: 'http://localhost:11435',  // stub server port
    },
  },
});
```

**Step 3 — Ollama stub**

The app makes two kinds of Ollama calls: `POST /api/chat` (NDJSON streaming) and
`POST /api/embed`. A minimal stub is a FastAPI app in `tests/e2e/ollama_stub.py`:

```python
# Returns a canned NDJSON stream for /api/chat and a fixed 4-dim embedding for /api/embed.
# Run alongside the main server during e2e tests.
```

Key fixture responses needed:
- `/api/chat` → evaluator response: `{"verdict": "pass", "new_inferences": [], ...}`
- `/api/embed` → `{"embeddings": [[0.1, 0.2, 0.3, 0.4]]}`
- `/api/generate` → `{}` (warmup response for character models)

The stub must implement Ollama's NDJSON streaming format:
```
{"message": {"role": "assistant", "content": "Hello"}, "done": false}
{"message": {"role": "assistant", "content": ""}, "done": true, "eval_count": 10}
```

A fixture that provides an evaluator-only response (separate from the character response)
is also needed. See `tests/unit/conftest.py` for `make_ollama_ndjson()` and
`make_evaluator_ndjson()` — port these patterns to the stub.

**Step 4 — Test structure**

```
tests/e2e/
  conftest.py          # start/stop Ollama stub as a subprocess fixture
  test_session_flow.py # session start → chat → end → experience review
  test_notifications.py # experience_update, implication, inference cards render
```

**Key test cases for the bugs this document is motivated by**

```python
# test_notifications.py

async def test_experience_update_notification_renders(page):
    """The experience_update sidechannel card appears in the chat stream."""
    # Arrange: seed one experience in DB; configure stub to return
    # experience_update verdict with contradicted_experience_id matching it.
    # Act: send a message.
    # Assert: the notification card with text "Experience updated" is visible.
    await expect(page.locator('.notification-card.experience-update')).to_be_visible()

async def test_experience_removed_from_pane_after_update(page):
    """The contradicted experience disappears from the Experiences pane."""
    # After the experience_update sidechannel, the pane no longer shows the
    # experience's statement text.
    await expect(page.locator('.experiences-section')).not_to_contain_text('We are in Chicago')

async def test_session_end_shows_proposals(page):
    """Clicking End Session populates the proposal review cards."""
    # Configure stub session-end response with two proposed experiences.
    await page.click('.btn-end')
    await expect(page.locator('.experience-proposal')).to_have_count(2)

async def test_active_ids_reset_on_session_end(page):
    """Active dot indicators (●) are cleared when the session ends."""
    # After End Session, no ● dots should remain in the experiences pane.
    await expect(page.locator('.exp-dot.active')).to_have_count(0)
```

### Complexity and cost

Option B is significantly more work than Option A:
- The Ollama stub is a non-trivial piece of infrastructure (~150–200 lines)
- SSE timing requires careful `wait_for` assertions to avoid flakiness
- The `webServer` lifecycle in `playwright.config.js` does not currently support
  starting a second process (the Ollama stub) — this needs a custom global setup/teardown
- CI will need Chromium installed (`npx playwright install --with-deps chromium`)
- Typical run time: 30–90 seconds vs. under 5 seconds for Option A

---

## Recommendation

**Implement Option A first.** It closes the state-logic gap (the majority of bugs found)
with minimal tooling overhead. The importmap approach is clean and future-proof.

Option B is worth adding later, narrowly scoped to the critical interaction flows listed
above, once the component logic is covered. Don't attempt full coverage with Playwright —
use it only for the paths that genuinely require rendering to verify.

**Process change (regardless of which option is chosen):**
Add to `CLAUDE.md` under the test layout section:
> Any new SSE sidechannel type requires three things in the same commit:
> 1. A case in `buildNotificationFromSidechannel` in `chat.js`
> 2. A `v-else-if="msg.scType === '...' "` notification card in `index.html`
> 3. Tests for both

This rule would have caught bug 1 (missing template block) without any tooling change.

---

## Implementation Notes for a Fresh Session

- Read `src/memories/frontend/index.html` in full before making changes — the setup()
  function is ~400 lines and all of it moves to `chat-component.js`.
- The `messagesEl` and `inputEl` template refs (`ref="messagesEl"`, `ref="inputEl"` in
  HTML) are assigned by Vue's template binding; they remain `null` in tests unless the
  component is fully mounted with a DOM. Tests that exercise `scrollToBottom()` or
  `inputEl.value?.focus()` should either mock these refs or use `@vue/test-utils` mount.
- `onMounted` registers two `document.addEventListener` calls for dropdown close
  behaviour. These will fire in jsdom without issue but should be cleaned up in
  `afterEach` if tests share a document.
- The `fetch` mock pattern used in `chat.test.js` (`vi.stubGlobal('fetch', vi.fn())`)
  works identically in component tests.
- Coverage threshold: after Option A, update `vitest.config.js` to include
  `chat-component.js` in `coverage.include`. The threshold can stay at 80% initially;
  the priority tests listed above should comfortably exceed it.
- For Option B, the existing `respx` mock patterns in `tests/unit/` and
  `tests/integration/` show how the team mocks Ollama HTTP calls in Python tests —
  the Playwright stub is the browser-side equivalent of the same approach.
