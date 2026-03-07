# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenJarvis is a modular AI assistant backend / research framework for on-device AI systems. It is organized around **five composable "pillars"** that are wired together via a config-driven `JarvisSystem` composition layer.

## Build & Development Commands

**Package manager:** [uv](https://github.com/astral-sh/uv) (lock file `uv.lock` is tracked for reproducibility)

```bash
# Install core + dev dependencies
uv sync --extra dev

# Lint (ruff, rules: E/F/I/W, target Python 3.10)
uv run ruff check src/ tests/

# Run full test suite
uv run pytest tests/ -v --tb=short

# Run a single test file / single test
uv run pytest tests/agents/test_native_react.py -v
uv run pytest tests/agents/test_native_react.py::test_function_name -v

# Run tests by marker (live, cloud, nvidia, amd, apple, slow)
uv run pytest -m "not live and not cloud" tests/

# CLI entry point
uv run jarvis --help

# Run evals
uv run python -m openjarvis.evals --config src/openjarvis/evals/configs/<config>.toml

# Docs (MkDocs Material)
uv sync --extra docs
uv run mkdocs serve
```

**Rust workspace** (in `rust/`):
```bash
cd rust
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

**Frontend / Desktop** (Tauri + Vite, in `frontend/` and `desktop/`):
```bash
cd frontend && npm install && npm run dev
cd desktop && npm install && npm run tauri dev
```

## Architecture: The Five Pillars

All pillars are wired together by `JarvisSystem` (`src/openjarvis/system.py`) which is constructed from `configs/openjarvis/config.toml` (or `~/.openjarvis/config.toml`).

### 1. Intelligence (`src/openjarvis/intelligence/`)
Model selection, provider routing. Config section: `[intelligence]`.

### 2. Agent (`src/openjarvis/agents/`)
Multi-turn reasoning and tool use. Agents register via `@AgentRegistry.register()` decorator. Key agents: `simple`, `native_react`, `native_openhands`, `orchestrator`, `monitor_operative`, `claude_code`, `rlm`. Config section: `[agent]`.

### 3. Tools (`src/openjarvis/tools/`)
Built-in tools (code_interpreter, web_search, file_read, shell_exec, calculator, think, browser, etc.) plus MCP adapter. Tool storage backends in `tools/storage/`. Config section: `[tools]`.

### 4. Engine (`src/openjarvis/engine/`)
Inference runtime abstraction. All engines implement `InferenceEngine` (defined in `engine/_stubs.py`) and use OpenAI-compatible chat completions. Supported: vLLM, Ollama, llama.cpp, SGLang, MLX, cloud (OpenAI/Anthropic/Google), LiteLLM, Apple FM, Exo, Nexa. Discovery in `engine/_discovery.py`. Config section: `[engine]`.

### 5. Learning (`src/openjarvis/learning/`)
Improvement methodologies: router policies (heuristic, bandit, trace-based), SFT/GRPO training, ICL updater, agent evolution, skill discovery. Orchestrated by `LearningOrchestrator`. Config section: `[learning]`.

### Supporting Systems
- **Core** (`core/`): `RegistryBase` pattern (decorator-based registration), types (`Message`, `Conversation`, `ToolCall`), config loader with hardware detection, `EventBus`.
- **Evals** (`evals/`): Benchmark framework with TOML configs. Datasets: SuperGPQA, GPQA, MMLU-Pro, MATH-500, GAIA, SWE-bench, FRAMES, SimpleQA, TerminalBench, PaperArena, etc. Run via `python -m openjarvis.evals`.
- **Channels** (`channels/`): Chat platform integrations (Telegram, Discord, Slack, WhatsApp, Signal, IRC, Matrix, etc.).
- **Telemetry** (`telemetry/`): GPU monitoring, energy measurement (NVIDIA/AMD/Apple/RAPL), latency instrumentation, vLLM metrics.
- **Traces** (`traces/`): Execution trace recording for analysis.
- **MCP** (`mcp/`): Model Context Protocol server.
- **Security** (`security/`): PII scanning, capability policies.
- **Server** (`server/`): FastAPI REST API.
- **SDK** (`sdk.py`): High-level `Jarvis` and `JarvisSystem` classes, `MemoryHandle` for memory operations.
- **Rust** (`rust/`): Parallel Rust implementation with PyO3 bindings. Workspace crates mirror Python pillars (core, engine, agents, tools, learning, telemetry, traces, security, mcp, python).

## Key Patterns

- **Registry pattern**: Components (agents, engines, memory backends, tools, channels, etc.) self-register via `@XRegistry.register("key")` decorators. Tests auto-clear all registries via `conftest.py` fixture.
- **Optional dependencies**: Heavy deps are extras in `pyproject.toml` (e.g., `inference-cloud`, `memory-faiss`, `channel-telegram`). Import failures are caught with try/except so the core stays lightweight.
- **OpenAI-compatible**: All engines expose an OpenAI-format chat completions interface. `messages_to_dicts()` in `engine/_base.py` handles conversion.
- **Config-driven**: TOML configs control everything. `load_config()` detects hardware, fills defaults, then overlays user overrides.

## Testing Conventions

- Tests mirror `src/` structure under `tests/`.
- Markers: `live` (needs running engine), `cloud` (needs API keys), `nvidia`/`amd`/`apple` (GPU-specific), `slow`.
- `conftest.py` provides hardware fixtures (`hardware_nvidia`, `hardware_apple`, etc.) and `mock_engine` factory.
- E501 line length is relaxed for `evals/datasets/*.py` and `evals/scorers/*.py`.
