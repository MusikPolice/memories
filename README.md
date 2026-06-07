# Memories

A locally-hosted character roleplay chatbot. It solves the problem that LLMs hallucinate biographical details freely and forget them just as freely — ask a raw model what a character's mother's name is and it will invent one confidently, then invent a different one ten turns later. Memories grounds character behaviour in a structured, user-defined fact sheet and uses a second LLM pass to enforce it.

## How it works

Every user message triggers two sequential LLM calls:

1. **Character LLM** — generates an in-character response, buffered server-side
2. **Evaluator LLM** — checks the buffered response against the character's established Facts and Inferences; returns a structured verdict

If the evaluator finds a contradiction, the response is suppressed and the character regenerates automatically. If it finds a new ungrounded detail, the user is prompted to accept or discard it. Only clean responses reach the chat window.

Over multiple sessions, Experiences accumulate — things the character learned through conversation — and are retrieved by semantic similarity at the start of each new session so the character remembers without being explicitly told.

## Prerequisites

### Ollama

This project uses [Ollama](https://ollama.com) to run LLMs locally. Install it from [ollama.com/download](https://ollama.com/download) and confirm it is running before starting the server:

```
ollama --version
```

Two models must be available in Ollama. Pull each one before launching the app.

---

### Model 1 — Embedding model (fixed): `nomic-embed-text`

```
ollama pull nomic-embed-text
```

[nomic-embed-text](https://ollama.com/library/nomic-embed-text) is a dedicated text embedding model. It is used exclusively to embed Experience statements and user messages into vectors for semantic similarity retrieval — it does not generate any text. There is no practical reason to substitute a different model here: nomic-embed-text is small (~274 MB), fast (a single embed call takes under a second on CPU), produces 768-dimensional normalised vectors, and is specifically trained for retrieval tasks.

This model is required for the Experiences feature (Phase 5). If it is not installed, the character will still function but will have no episodic memory across sessions.

---

### Model 2 — Character and evaluator LLM (your choice)

The same model handles both the character roleplay and the evaluator critic pass. Choose one based on your hardware and quality requirements.

#### What to look for

**Instruction following.** The evaluator must return a specific JSON structure with a fixed vocabulary of verdict strings (`pass`, `contradiction`, `implication`, etc.). A model that drifts from the schema — returning prose instead of JSON, or inventing verdict names — will cause evaluator parse errors and degrade reliability. Prioritise models known for strong instruction following.

**JSON output compatibility.** The evaluator call uses Ollama's `format: "json"` constraint to nudge the model toward valid JSON. Most modern instruction-tuned models handle this well. Older or smaller models may still produce malformed output.

**Context window.** The character's Facts, Inferences, and conversation history are all injected into the prompt on every turn. A 4K context window is marginal for anything beyond a short session. 8K is comfortable for typical use; 16K or more gives headroom for long sessions and rich character sheets without degradation.

**Roleplay quality.** The character LLM must stay in character, respond naturally to creative prompts, and produce responses that feel like a specific person rather than a generic assistant. General-purpose instruction models vary significantly in this dimension.

#### Hardware sizing

| Available RAM / VRAM | Recommended parameter count | Typical quantisation |
|---|---|---|
| 8 GB | 3B–7B | Q4_K_M |
| 16 GB | 7B–14B | Q4_K_M or Q5_K_M |
| 24 GB+ | 14B–32B | Q5_K_M or higher |
| CPU only (no GPU) | 3B–7B | Q4_K_M (expect slow responses) |

Running on CPU is possible but character response latency will be several seconds to tens of seconds depending on the model and machine. GPU acceleration (CUDA or Metal) is strongly recommended for a usable experience.

#### Recommended starting point

**[qwen3:7b](https://ollama.com/library/qwen3)** is the model this project was developed and tested against. It has strong instruction following, reliably produces valid JSON for the evaluator, handles the system prompt structure well, and fits comfortably in 8 GB of VRAM at Q4_K_M quantisation.

```
ollama pull qwen3:7b
```

If you have more VRAM available, `qwen3:14b` or `qwen3:30b-a3b` (a Mixture-of-Experts variant that runs efficiently despite its parameter count) will produce noticeably better roleplay quality and more reliable evaluator verdicts.

Other models worth trying:

- **[llama3.2:3b](https://ollama.com/library/llama3.2)** — for very constrained hardware; quality is reduced but functional
- **[gemma3:12b](https://ollama.com/library/gemma3)** — strong instruction following, good creative writing
- **[mistral-nemo](https://ollama.com/library/mistral-nemo)** — 12B, excellent JSON discipline, good context handling
- **[llama3.1:8b](https://ollama.com/library/llama3.1)** — solid all-rounder if you prefer Meta's model family

Browse the full model library at [ollama.com/library](https://ollama.com/library). Filter by `embedding` to see alternatives to nomic-embed-text; filter by size to find what fits your hardware.

> **Note on thinking models.** Both LLM calls run with `think: false`. Thinking mode is deliberately disabled — the evaluator prompt already supplies the structured reasoning context that thinking would otherwise provide, and extended thinking adds latency without improving verdict quality for this task. If you choose a model that defaults to thinking (e.g., `qwen3` variants with `:thinking` tags), make sure you are pulling the standard variant, not the thinking-enabled one.

---

## Setup

```bash
# Install Python dependencies
uv sync

# Install frontend test dependencies (optional, for running JS tests)
npm install
```

## Running the server

```bash
uv run uvicorn memories.main:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in a browser. The app is accessible from other devices on the same LAN at your machine's local IP address on port 8000.

On first launch, create a character by entering a name and the Ollama model name (e.g., `qwen3:7b`) in the UI. The server will warm up the model on startup in subsequent runs.

## Running tests

```bash
# All Python tests (requires 80% coverage)
uv run pytest

# Without coverage enforcement (faster iteration)
uv run pytest --no-cov

# Frontend tests
npm test
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model for Experience retrieval |
| `MEMORIES_DB_PATH` | `memories.db` | SQLite database file path |
| `MAX_CONTRADICTION_RETRIES` | `3` | Evaluator retry limit before delivering unverified |
| `MAX_INFERENCE_DEPTH` | `5` | Maximum inference chain depth from root Facts |
| `MAX_INFERENCE_BREADTH` | `5` | Maximum inferences generated per eager pass |
| `TOP_K_EXPERIENCES` | `5` | Experiences retrieved per turn for context |

## Known limitations

- **Embedding model change invalidates stored Experiences.** If you change `EMBED_MODEL` after Experiences have been stored, the stored vectors and new query vectors are from incompatible spaces and similarity scores will be meaningless. Delete all Experiences via the UI before switching models.
- **Long sessions may approach the model's context window.** Facts, Inferences, active Experiences, and conversation history are all injected on every turn. Very long sessions with rich character sheets can degrade response quality as the prompt approaches the model's limit. Context budget tracking and compression are planned for a later phase.
