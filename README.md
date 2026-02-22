<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/openjarvis-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/openjarvis-logo-light.svg">
    <img alt="OpenJarvis" src="assets/openjarvis-logo-light.svg" width="400">
  </picture>
</p>

<p align="center"><i>Programming abstractions for on-device AI.</i></p>

---

OpenJarvis is a framework for building AI systems that run *entirely on local hardware*. Rather than treating intelligence as a cloud service, OpenJarvis provides composable abstractions for local model selection, inference, agentic reasoning, tool use, and learning — all aware of the hardware they run on.

You write Python programs that compose five pillars — **Intelligence** (which model), **Engine** (which runtime), **Agents** (which reasoning strategy), **Tools** (which capabilities, via MCP), and **Learning** (which adaptation policy) — and OpenJarvis handles hardware detection, model routing, telemetry, and trace-driven improvement automatically.

```python
from openjarvis import Jarvis

j = Jarvis()                                      # auto-detect hardware + engine
response = j.ask("Explain backpropagation")       # route to best local model

j.ask("Solve x^2 - 5x + 6 = 0",                  # multi-turn agent with tools
      agent="orchestrator",
      tools=["calculator", "think"])

j.memory.index("./papers/")                       # index documents into local storage
results = j.memory.search("attention mechanism")  # semantic retrieval

j.close()
```

```bash
pip install openjarvis
jarvis ask "Hello, what can you do?"
jarvis serve --port 8000                           # OpenAI-compatible API
```

## Installation

```bash
pip install openjarvis            # core framework
pip install openjarvis[server]    # + FastAPI server
pip install openjarvis[openclaw]  # + OpenClaw agent (requires Node.js 22+)
```

You also need a local inference backend: [Ollama](https://ollama.com), [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), or [llama.cpp](https://github.com/ggerganov/llama.cpp).

## The Five Pillars

| Pillar | What it does | Key abstractions |
|--------|-------------|-----------------|
| **Intelligence** | Model management and routing | `RouterPolicy`, `QueryAnalyzer`, `ModelCatalog` |
| **Engine** | Inference runtime abstraction | `InferenceEngine` ABC — Ollama, vLLM, SGLang, llama.cpp, MLX, cloud |
| **Agents** | Pluggable reasoning strategies | `BaseAgent` ABC — Simple, Orchestrator, ReAct, OpenHands, OpenClaw |
| **Tools** | Capabilities via MCP | `BaseTool` ABC — calculator, code interpreter, web search, memory, LLM sub-calls; external MCP servers auto-discovered |
| **Learning** | Trace-driven adaptation | `LearningPolicy` ABC — SFT (model routing), AgentAdvisor (restructuring), ICL (tool usage) |

Every interaction produces a **Trace** — a structured record of the full reasoning chain (routing decisions, tool calls, latencies, outcomes). Learning policies consume traces to improve model selection, agent behavior, and tool usage over time.

## Config-Driven Composition

OpenJarvis is fully configurable via `~/.openjarvis/config.toml` or programmatically via `SystemBuilder`:

```python
from openjarvis.system import SystemBuilder

system = (SystemBuilder()
          .engine("ollama")
          .model("qwen3:8b")
          .agent("orchestrator")
          .tools(["calculator", "think", "memory_retrieve"])
          .telemetry(True)
          .build())

result = system.ask("What is 2+2?")
system.close()
```

Hardware auto-detection selects the best engine: Apple Silicon &rarr; Ollama, NVIDIA datacenter GPUs &rarr; vLLM, AMD &rarr; vLLM, CPU-only &rarr; llama.cpp.

## MCP Interoperability

All tools are managed via the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP). The built-in MCP server exposes every OpenJarvis tool — including memory operations — to any MCP-compatible client (Claude, GPT, Gemini, etc.). External MCP servers are auto-discovered and their tools appear as native `BaseTool` instances inside OpenJarvis agents.

## Documentation

Full docs at the [OpenJarvis documentation site](docs/) or in-repo:

- **[VISION.md](VISION.md)** — Project vision and design principles
- **[CLAUDE.md](CLAUDE.md)** — Developer reference for the codebase
- **[docs/](docs/)** — Architecture guides, API reference, tutorials

## About

OpenJarvis is part of [Intelligence Per Watt](https://www.intelligence-per-watt.ai/), a research initiative studying the efficiency of on-device AI systems. The project is developed at [Hazy Research](https://hazyresearch.stanford.edu/) and the [Scaling Intelligence Lab](https://scalingintelligence.stanford.edu/) at [Stanford SAIL](https://ai.stanford.edu/).

## License

Apache 2.0
