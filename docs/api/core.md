# Core Module

The core module contains the foundational abstractions shared across all
OpenJarvis pillars: the decorator-based registry system for runtime discovery,
canonical data types, configuration loading with hardware detection, and the
pub/sub event bus for inter-pillar communication.

## Registry System

The registry pattern is the primary extension mechanism. Each pillar has its
own typed registry subclass. New implementations are added by decorating a
class with `@XRegistry.register("name")`.

### RegistryBase

::: openjarvis.core.registry.RegistryBase
    options:
      show_source: true
      members_order: source

### ModelRegistry

::: openjarvis.core.registry.ModelRegistry
    options:
      show_source: true
      members_order: source

### EngineRegistry

::: openjarvis.core.registry.EngineRegistry
    options:
      show_source: true
      members_order: source

### MemoryRegistry

::: openjarvis.core.registry.MemoryRegistry
    options:
      show_source: true
      members_order: source

### AgentRegistry

::: openjarvis.core.registry.AgentRegistry
    options:
      show_source: true
      members_order: source

### ToolRegistry

::: openjarvis.core.registry.ToolRegistry
    options:
      show_source: true
      members_order: source

### RouterPolicyRegistry

::: openjarvis.core.registry.RouterPolicyRegistry
    options:
      show_source: true
      members_order: source

### BenchmarkRegistry

::: openjarvis.core.registry.BenchmarkRegistry
    options:
      show_source: true
      members_order: source

### ChannelRegistry

::: openjarvis.core.registry.ChannelRegistry
    options:
      show_source: true
      members_order: source

### LearningRegistry

::: openjarvis.core.registry.LearningRegistry
    options:
      show_source: true
      members_order: source

---

## Types

Canonical data types shared across all OpenJarvis pillars, including message
structures, model specifications, telemetry records, and trace objects.

### Role

::: openjarvis.core.types.Role
    options:
      show_source: true
      members_order: source

### Quantization

::: openjarvis.core.types.Quantization
    options:
      show_source: true
      members_order: source

### StepType

::: openjarvis.core.types.StepType
    options:
      show_source: true
      members_order: source

### ToolCall

::: openjarvis.core.types.ToolCall
    options:
      show_source: true
      members_order: source

### Message

::: openjarvis.core.types.Message
    options:
      show_source: true
      members_order: source

### Conversation

::: openjarvis.core.types.Conversation
    options:
      show_source: true
      members_order: source

### ModelSpec

::: openjarvis.core.types.ModelSpec
    options:
      show_source: true
      members_order: source

### ToolResult

::: openjarvis.core.types.ToolResult
    options:
      show_source: true
      members_order: source

### TelemetryRecord

::: openjarvis.core.types.TelemetryRecord
    options:
      show_source: true
      members_order: source

### TraceStep

::: openjarvis.core.types.TraceStep
    options:
      show_source: true
      members_order: source

### Trace

::: openjarvis.core.types.Trace
    options:
      show_source: true
      members_order: source

### RoutingContext

::: openjarvis.core.types.RoutingContext
    options:
      show_source: true
      members_order: source

---

## Configuration

Configuration loading, hardware detection, and engine recommendation. User
configuration lives at `~/.openjarvis/config.toml`. The `load_config()`
function detects hardware, fills sensible defaults, then overlays any user
overrides from the TOML file.

### JarvisConfig

::: openjarvis.core.config.JarvisConfig
    options:
      show_source: true
      members_order: source

### EngineConfig

::: openjarvis.core.config.EngineConfig
    options:
      show_source: true
      members_order: source

### IntelligenceConfig

::: openjarvis.core.config.IntelligenceConfig
    options:
      show_source: true
      members_order: source

### LearningConfig

::: openjarvis.core.config.LearningConfig
    options:
      show_source: true
      members_order: source

### MemoryConfig

::: openjarvis.core.config.MemoryConfig
    options:
      show_source: true
      members_order: source

### AgentConfig

::: openjarvis.core.config.AgentConfig
    options:
      show_source: true
      members_order: source

### ServerConfig

::: openjarvis.core.config.ServerConfig
    options:
      show_source: true
      members_order: source

### TelemetryConfig

::: openjarvis.core.config.TelemetryConfig
    options:
      show_source: true
      members_order: source

### TracesConfig

::: openjarvis.core.config.TracesConfig
    options:
      show_source: true
      members_order: source

### StorageConfig

::: openjarvis.core.config.StorageConfig
    options:
      show_source: true
      members_order: source

### MCPConfig

::: openjarvis.core.config.MCPConfig
    options:
      show_source: true
      members_order: source

### ToolsConfig

::: openjarvis.core.config.ToolsConfig
    options:
      show_source: true
      members_order: source

### HardwareInfo

::: openjarvis.core.config.HardwareInfo
    options:
      show_source: true
      members_order: source

### GpuInfo

::: openjarvis.core.config.GpuInfo
    options:
      show_source: true
      members_order: source

### detect_hardware

::: openjarvis.core.config.detect_hardware
    options:
      show_source: true

### load_config

::: openjarvis.core.config.load_config
    options:
      show_source: true

### recommend_engine

::: openjarvis.core.config.recommend_engine
    options:
      show_source: true

---

## Event Bus

Thread-safe pub/sub event bus for inter-pillar telemetry. Any pillar can emit
events (e.g. `INFERENCE_END`) and any other pillar can react without direct
coupling.

### EventType

::: openjarvis.core.events.EventType
    options:
      show_source: true
      members_order: source

### Event

::: openjarvis.core.events.Event
    options:
      show_source: true
      members_order: source

### EventBus

::: openjarvis.core.events.EventBus
    options:
      show_source: true
      members_order: source

### get_event_bus

::: openjarvis.core.events.get_event_bus
    options:
      show_source: true

### reset_event_bus

::: openjarvis.core.events.reset_event_bus
    options:
      show_source: true
