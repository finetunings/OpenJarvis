# Intelligence Pillar

The Intelligence pillar handles **model management and query routing**. It maintains a catalog of known models with detailed metadata and provides a heuristic router that selects the best model for a given query based on its characteristics.

---

## Purpose

When a user sends a query to OpenJarvis, the system needs to decide which model should handle it. A short, simple question like "What time is it?" does not need a 70B parameter model, while a complex multi-step math problem benefits from the largest available model. The Intelligence pillar encapsulates this decision-making logic.

The pillar provides three key capabilities:

1. **Model catalog** -- a registry of well-known models with metadata (parameter count, context length, VRAM requirements, supported engines)
2. **Query routing** -- analyzing query characteristics and selecting the optimal model
3. **Auto-discovery** -- merging models discovered from running engines into the catalog

---

## ModelSpec

Every model in the system is described by a `ModelSpec` dataclass, defined in `core/types.py`:

```python
@dataclass(slots=True)
class ModelSpec:
    model_id: str                              # Unique identifier (e.g., "qwen3:8b")
    name: str                                  # Human-readable name
    parameter_count_b: float                   # Total parameters in billions
    context_length: int                        # Maximum context window (tokens)
    active_parameter_count_b: Optional[float]  # MoE active params (None for dense)
    quantization: Quantization                 # Quantization format (none, fp8, int4, etc.)
    min_vram_gb: float                         # Minimum VRAM required
    supported_engines: Sequence[str]           # Which engines can run this model
    provider: str                              # Model provider (e.g., "alibaba", "meta")
    requires_api_key: bool                     # Whether cloud API key is needed
    metadata: Dict[str, Any]                   # Additional metadata (pricing, architecture)
```

Models are registered in the `ModelRegistry`:

```python
from openjarvis.core.registry import ModelRegistry

# Register a model
ModelRegistry.register_value("qwen3:8b", ModelSpec(
    model_id="qwen3:8b",
    name="Qwen3 8B",
    parameter_count_b=8.2,
    context_length=32768,
    supported_engines=("vllm", "ollama", "llamacpp", "sglang"),
    provider="alibaba",
))
```

---

## Model Catalog

The built-in model catalog is defined in `intelligence/model_catalog.py` as the `BUILTIN_MODELS` list. It includes models across three categories:

### Local Models -- Dense

| Model ID | Name | Parameters | Context | Supported Engines |
|----------|------|-----------|---------|-------------------|
| `qwen3:8b` | Qwen3 8B | 8.2B | 32K | vLLM, Ollama, llama.cpp, SGLang |
| `qwen3:32b` | Qwen3 32B | 32B | 32K | Ollama, vLLM |
| `llama3.3:70b` | Llama 3.3 70B | 70B | 128K | Ollama, vLLM |
| `llama3.2:3b` | Llama 3.2 3B | 3B | 128K | Ollama, vLLM, llama.cpp |
| `deepseek-coder-v2:16b` | DeepSeek Coder V2 16B | 16B | 128K | Ollama, vLLM |
| `mistral:7b` | Mistral 7B | 7B | 32K | Ollama, vLLM, llama.cpp |

### Local Models -- Mixture of Experts (MoE)

| Model ID | Name | Total / Active Params | Context | Min VRAM |
|----------|------|----------------------|---------|----------|
| `gpt-oss:120b` | GPT-OSS 120B | 117B / 5.1B | 128K | 12 GB |
| `glm-4.7-flash` | GLM 4.7 Flash | 30B / 3B | 128K | 8 GB |
| `trinity-mini` | Trinity Mini | 26B / 3B | 128K | 8 GB |

### Cloud Models

| Model ID | Provider | Context | Pricing (input/output per 1M tokens) |
|----------|----------|---------|--------------------------------------|
| `gpt-4o` | OpenAI | 128K | $2.50 / $10.00 |
| `gpt-4o-mini` | OpenAI | 128K | $0.15 / $0.60 |
| `gpt-5-mini` | OpenAI | 400K | $0.25 / $2.00 |
| `claude-sonnet-4-20250514` | Anthropic | 200K | $3.00 / $15.00 |
| `claude-opus-4-20250514` | Anthropic | 200K | $15.00 / $75.00 |
| `claude-opus-4-6` | Anthropic | 200K | $5.00 / $25.00 |
| `gemini-2.5-pro` | Google | 1M | $1.25 / $10.00 |
| `gemini-2.5-flash` | Google | 1M | $0.30 / $2.50 |

### Registering Built-in Models

The `register_builtin_models()` function populates the `ModelRegistry` with all built-in models. It skips models that are already registered, making it safe to call multiple times:

```python
from openjarvis.intelligence import register_builtin_models

register_builtin_models()
# All BUILTIN_MODELS are now in ModelRegistry
```

---

## Auto-Discovery: Merging Runtime Models

When engines are discovered at runtime, they report models that may not be in the built-in catalog. The `merge_discovered_models()` function creates minimal `ModelSpec` entries for these:

```python
from openjarvis.intelligence import merge_discovered_models

# Models reported by Ollama that aren't in the catalog
merge_discovered_models("ollama", ["phi3:3.8b", "codellama:7b"])
```

For each model ID not already in the registry, a `ModelSpec` is created with the model ID as both the `model_id` and `name`, with zero-value defaults for unknown fields. This ensures the routing system can still select from all available models, even ones it has no metadata for.

---

## HeuristicRouter

The `HeuristicRouter` is a rule-based model router that selects the best model based on query characteristics. It applies six priority rules in order:

### Routing Rules

| Priority | Rule | Condition | Action |
|----------|------|-----------|--------|
| 1 | Code detection | Query contains code patterns (backticks, `def`, `class`, `import`, `function`, `=>`, etc.) | Prefer model with "code" or "coder" in name; fall back to largest model |
| 2 | Math detection | Query contains math keywords (`solve`, `integral`, `equation`, `calculate`, `compute`, etc.) | Select the largest available model |
| 3 | Short query | Query length < 50 characters, no code/math | Select the smallest available model (faster response) |
| 4 | Long/complex query | Query length > 500 characters OR contains reasoning keywords (`explain`, `analyze`, `compare`, `step-by-step`, etc.) | Select the largest available model |
| 5 | High urgency | `urgency > 0.8` | Override to smallest model (fastest response) |
| 6 | Default fallback | None of the above match | Use `default_model`, then `fallback_model`, then first available |

!!! note "Priority 5 overrides all others"
    The urgency check (rule 5) is actually evaluated **first** in the code -- if urgency exceeds 0.8, the router immediately returns the smallest model regardless of query content.

### Usage

```python
from openjarvis.intelligence import HeuristicRouter, build_routing_context

router = HeuristicRouter(
    available_models=["qwen3:8b", "llama3.2:3b", "deepseek-coder-v2:16b"],
    default_model="qwen3:8b",
    fallback_model="llama3.2:3b",
)

ctx = build_routing_context("Write a Python function to sort a list")
model = router.select_model(ctx)  # Returns "deepseek-coder-v2:16b" (has "coder")
```

---

## RouterPolicy and QueryAnalyzer ABCs

The Intelligence pillar now defines the `RouterPolicy` and `QueryAnalyzer` ABCs in `intelligence/_stubs.py`:

```python
# intelligence/_stubs.py
class RouterPolicy(ABC):
    @abstractmethod
    def select_model(self, context: RoutingContext) -> str:
        """Return the model registry key best suited for *context*."""

class QueryAnalyzer(ABC):
    @abstractmethod
    def analyze(self, query: str) -> RoutingContext:
        """Analyze a raw query string and return a RoutingContext."""
```

The `HeuristicRouter` implements both of these ABCs.

---

## build_routing_context()

The `build_routing_context()` function analyzes a raw query string and produces a `RoutingContext` dataclass (now defined in `core/types.py`):

```python
# core/types.py
@dataclass(slots=True)
class RoutingContext:
    query: str = ""
    query_length: int = 0
    has_code: bool = False
    has_math: bool = False
    language: str = "en"
    urgency: float = 0.5  # 0 = low priority, 1 = real-time
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Code detection** uses regex patterns matching:

- Backtick code blocks (`` ``` `` or `` `inline` ``)
- Language keywords (`def`, `class`, `import`, `function`, `const`, `var`, `let`)
- Syntax patterns (`if (`, `->`, `=>`, `{ }`, `for x in`, `#include`, `System.out`)

**Math detection** uses regex patterns matching:

- Mathematical terms (`solve`, `integral`, `equation`, `proof`, `derivative`, `matrix`)
- Computational keywords (`calculate`, `compute`, `sigma`, `sum`, `limit`, `probability`)

```python
from openjarvis.intelligence import build_routing_context

ctx = build_routing_context("Solve the integral of x^2 dx")
# ctx.has_math = True, ctx.has_code = False, ctx.query_length = 32

ctx = build_routing_context("```python\ndef hello():\n    pass\n```")
# ctx.has_code = True, ctx.has_math = False
```

---

## Integration with Learning

The `HeuristicRouter` implements the `RouterPolicy` ABC (now defined in `intelligence/_stubs.py`), which means it can be swapped out for a `TraceDrivenPolicy`, `SFTPolicy`, or any other policy via the `RouterPolicyRegistry`. See the [Learning & Traces](learning.md) documentation for details on how trace-driven routing and the broader `LearningPolicy` taxonomy work.

The router is registered as `"heuristic"` in the `RouterPolicyRegistry` and is the default routing policy. Users can switch policies via the `--router` CLI flag or the `learning.default_policy` config setting.
