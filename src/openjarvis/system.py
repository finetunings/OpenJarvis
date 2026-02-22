"""Composition layer -- config-driven construction of a fully wired JarvisSystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openjarvis.core.config import JarvisConfig, load_config
from openjarvis.core.events import EventBus, get_event_bus
from openjarvis.core.types import Message, Role
from openjarvis.engine._stubs import InferenceEngine
from openjarvis.tools._stubs import BaseTool, ToolExecutor


@dataclass
class JarvisSystem:
    """Fully wired system -- the single source of truth for pillar composition."""

    config: JarvisConfig
    bus: EventBus
    engine: InferenceEngine
    engine_key: str
    model: str
    agent: Optional[Any] = None  # BaseAgent
    agent_name: str = ""
    tools: List[BaseTool] = field(default_factory=list)
    tool_executor: Optional[ToolExecutor] = None
    memory_backend: Optional[Any] = None  # MemoryBackend
    router: Optional[Any] = None  # RouterPolicy
    mcp_server: Optional[Any] = None  # MCPServer
    telemetry_store: Optional[Any] = None
    trace_store: Optional[Any] = None
    trace_collector: Optional[Any] = None

    def ask(
        self,
        query: str,
        *,
        context: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        agent: Optional[str] = None,
        tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute a query through the system and return a result dict."""
        messages = [Message(role=Role.USER, content=query)]

        # Context injection from memory
        if context and self.memory_backend and self.config.memory.context_injection:
            try:
                from openjarvis.tools.storage.context import (
                    ContextConfig,
                    inject_context,
                )

                ctx_cfg = ContextConfig(
                    top_k=self.config.memory.context_top_k,
                    min_score=self.config.memory.context_min_score,
                    max_context_tokens=self.config.memory.context_max_tokens,
                )
                messages = inject_context(
                    query, messages, self.memory_backend, config=ctx_cfg,
                )
            except Exception:
                pass

        # Agent mode
        use_agent = agent or self.agent_name
        if use_agent and use_agent != "none":
            return self._run_agent(
                query, messages, use_agent, tools, temperature, max_tokens,
            )

        # Direct engine mode
        result = self.engine.generate(
            messages, model=self.model,
            temperature=temperature, max_tokens=max_tokens,
        )
        return {
            "content": result.get("content", ""),
            "usage": result.get("usage", {}),
            "model": self.model,
            "engine": self.engine_key,
        }

    def _run_agent(
        self, query, messages, agent_name, tool_names, temperature, max_tokens,
    ) -> Dict[str, Any]:
        """Run through an agent."""
        from openjarvis.agents._stubs import AgentContext
        from openjarvis.core.registry import AgentRegistry

        # Resolve agent
        try:
            agent_cls = AgentRegistry.get(agent_name)
        except KeyError:
            return {"content": f"Unknown agent: {agent_name}", "error": True}

        # Build tools for agent
        agent_tools = self.tools
        if tool_names:
            agent_tools = self._build_tools(tool_names)

        # Build context
        ctx = AgentContext()

        # Inject memory context messages into the agent conversation
        if messages and len(messages) > 1:
            # Context messages were prepended by inject_context
            for msg in messages[:-1]:
                ctx.conversation.add(msg)

        # Instantiate agent with the same pattern as CLI
        agent_kwargs: Dict[str, Any] = {
            "bus": self.bus,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if agent_name == "orchestrator":
            agent_kwargs["tools"] = agent_tools
            agent_kwargs["max_turns"] = self.config.agent.max_turns

        try:
            ag = agent_cls(self.engine, self.model, **agent_kwargs)
        except TypeError:
            try:
                ag = agent_cls(self.engine, self.model)
            except TypeError:
                ag = agent_cls()

        # Run
        result = ag.run(query, context=ctx)
        return {
            "content": result.content,
            "usage": getattr(result, "usage", {}),
            "tool_results": [
                {
                    "tool_name": tr.tool_name,
                    "content": tr.content,
                    "success": tr.success,
                }
                for tr in getattr(result, "tool_results", [])
            ],
            "turns": getattr(result, "turns", 1),
            "model": self.model,
            "engine": self.engine_key,
        }

    def _build_tools(self, tool_names: List[str]) -> List[BaseTool]:
        """Build tool instances from tool names."""
        from openjarvis.core.registry import ToolRegistry

        tools: List[BaseTool] = []
        for name in tool_names:
            try:
                if name == "retrieval" and self.memory_backend:
                    from openjarvis.tools.retrieval import RetrievalTool

                    tools.append(RetrievalTool(self.memory_backend))
                elif name == "llm":
                    from openjarvis.tools.llm_tool import LLMTool

                    tools.append(LLMTool(self.engine, model=self.model))
                elif ToolRegistry.contains(name):
                    tools.append(ToolRegistry.create(name))
            except Exception:
                pass
        return tools

    def close(self) -> None:
        """Release resources."""
        if self.telemetry_store and hasattr(self.telemetry_store, "close"):
            self.telemetry_store.close()
        if self.trace_store and hasattr(self.trace_store, "close"):
            self.trace_store.close()


class SystemBuilder:
    """Config-driven fluent builder for JarvisSystem."""

    def __init__(
        self,
        config: Optional[JarvisConfig] = None,
        *,
        config_path: Optional[Any] = None,
    ) -> None:
        if config is not None:
            self._config = config
        elif config_path is not None:
            from pathlib import Path

            self._config = load_config(Path(config_path))
        else:
            self._config = load_config()

        self._engine_key: Optional[str] = None
        self._model: Optional[str] = None
        self._agent_name: Optional[str] = None
        self._tool_names: Optional[List[str]] = None
        self._telemetry: Optional[bool] = None
        self._traces: Optional[bool] = None
        self._bus: Optional[EventBus] = None

    def engine(self, key: str) -> SystemBuilder:
        self._engine_key = key
        return self

    def model(self, name: str) -> SystemBuilder:
        self._model = name
        return self

    def agent(self, name: str) -> SystemBuilder:
        self._agent_name = name
        return self

    def tools(self, names: List[str]) -> SystemBuilder:
        self._tool_names = names
        return self

    def telemetry(self, enabled: bool) -> SystemBuilder:
        self._telemetry = enabled
        return self

    def traces(self, enabled: bool) -> SystemBuilder:
        self._traces = enabled
        return self

    def event_bus(self, bus: EventBus) -> SystemBuilder:
        self._bus = bus
        return self

    def build(self) -> JarvisSystem:
        """Construct a fully wired JarvisSystem."""
        config = self._config
        bus = self._bus or get_event_bus()

        # Resolve engine
        engine, engine_key = self._resolve_engine(config)

        # Resolve model
        model = self._resolve_model(config, engine)

        # Wrap with InstrumentedEngine if telemetry enabled
        telemetry_enabled = (
            self._telemetry if self._telemetry is not None
            else config.telemetry.enabled
        )
        if telemetry_enabled:
            from openjarvis.telemetry.instrumented_engine import InstrumentedEngine
            engine = InstrumentedEngine(engine, bus)

        # Apply security guardrails to engine
        engine = self._apply_security(config, engine, bus)

        # Set up telemetry
        telemetry_store = None
        telemetry_enabled = (
            self._telemetry if self._telemetry is not None else config.telemetry.enabled
        )
        if telemetry_enabled:
            telemetry_store = self._setup_telemetry(config, bus)

        # Resolve memory backend
        memory_backend = self._resolve_memory(config)

        # Resolve tools
        tool_list = self._resolve_tools(config, engine, model, memory_backend)

        # Build tool executor
        tool_executor = ToolExecutor(tool_list, bus) if tool_list else None

        # Resolve agent name
        agent_name = self._agent_name or config.agent.default_agent

        return JarvisSystem(
            config=config,
            bus=bus,
            engine=engine,
            engine_key=engine_key,
            model=model,
            agent_name=agent_name,
            tools=tool_list,
            tool_executor=tool_executor,
            memory_backend=memory_backend,
            telemetry_store=telemetry_store,
        )

    def _resolve_engine(self, config: JarvisConfig):
        """Resolve the inference engine."""
        from openjarvis.engine._discovery import get_engine

        key = self._engine_key or config.engine.default
        resolved = get_engine(config, key)
        if resolved is None:
            raise RuntimeError(
                "No inference engine available. "
                "Make sure an engine is running (e.g. ollama serve)."
            )
        return resolved[1], resolved[0]

    def _resolve_model(self, config: JarvisConfig, engine: InferenceEngine) -> str:
        """Resolve which model to use."""
        if self._model:
            return self._model
        if config.intelligence.default_model:
            return config.intelligence.default_model

        # Try to discover from engine
        try:
            models = engine.list_models()
            if models:
                return models[0]
        except Exception:
            pass

        return config.intelligence.fallback_model or ""

    def _apply_security(self, config, engine, bus):
        """Wrap engine with security guardrails if enabled."""
        if config.security.enabled:
            try:
                from openjarvis.security.guardrails import GuardrailsEngine
                from openjarvis.security.scanner import PIIScanner, SecretScanner
                from openjarvis.security.types import RedactionMode

                scanners = []
                if config.security.secret_scanner:
                    scanners.append(SecretScanner())
                if config.security.pii_scanner:
                    scanners.append(PIIScanner())

                if scanners:
                    mode_map = {
                        "warn": RedactionMode.WARN,
                        "redact": RedactionMode.REDACT,
                        "block": RedactionMode.BLOCK,
                    }
                    mode = mode_map.get(config.security.mode, RedactionMode.WARN)
                    engine = GuardrailsEngine(
                        engine, scanners, mode=mode, bus=bus,
                        scan_input=config.security.scan_input,
                        scan_output=config.security.scan_output,
                    )
            except Exception:
                pass
        return engine

    def _setup_telemetry(self, config, bus):
        """Set up telemetry store."""
        try:
            from openjarvis.telemetry.store import TelemetryStore

            store = TelemetryStore(db_path=config.telemetry.db_path)
            store.subscribe_to_bus(bus)
            return store
        except Exception:
            return None

    def _resolve_memory(self, config):
        """Resolve memory backend."""
        try:
            import openjarvis.memory  # noqa: F401 -- trigger registration
            from openjarvis.core.registry import MemoryRegistry

            key = config.memory.default_backend
            if MemoryRegistry.contains(key):
                return MemoryRegistry.create(key, db_path=config.memory.db_path)
        except Exception:
            pass
        return None

    def _resolve_tools(self, config, engine, model, memory_backend):
        """Resolve tool instances."""
        tool_names_str = self._tool_names
        if tool_names_str is None:
            # Use config default or agent defaults
            raw = config.tools.enabled or config.agent.default_tools
            if raw:
                tool_names_str = [n.strip() for n in raw.split(",") if n.strip()]
            else:
                tool_names_str = []

        tools: List[BaseTool] = []
        for name in tool_names_str:
            try:
                if name == "retrieval" and memory_backend:
                    from openjarvis.tools.retrieval import RetrievalTool

                    tools.append(RetrievalTool(memory_backend))
                elif name == "llm":
                    from openjarvis.tools.llm_tool import LLMTool

                    tools.append(LLMTool(engine, model=model))
                elif name in (
                    "memory_store",
                    "memory_retrieve",
                    "memory_search",
                    "memory_index",
                ):
                    from openjarvis.core.registry import ToolRegistry
                    from openjarvis.tools import storage_tools  # noqa: F401

                    if ToolRegistry.contains(name):
                        tool = ToolRegistry.create(name, backend=memory_backend)
                        tools.append(tool)
                else:
                    from openjarvis.core.registry import ToolRegistry

                    if ToolRegistry.contains(name):
                        tools.append(ToolRegistry.create(name))
                    else:
                        # Try direct import for tools that need ensure_registered
                        import openjarvis.tools  # noqa: F401

                        if ToolRegistry.contains(name):
                            tools.append(ToolRegistry.create(name))
            except Exception:
                pass

        return tools


__all__ = ["JarvisSystem", "SystemBuilder"]
