"""Tests for core data types."""

from __future__ import annotations

from evals.core.types import (
    BenchmarkConfig,
    DefaultsConfig,
    EvalRecord,
    EvalResult,
    EvalSuiteConfig,
    ExecutionConfig,
    JudgeConfig,
    MetaConfig,
    ModelConfig,
    RunConfig,
    RunSummary,
)


class TestEvalRecord:
    def test_creation(self):
        r = EvalRecord(
            record_id="r1", problem="What?", reference="42",
            category="reasoning",
        )
        assert r.record_id == "r1"
        assert r.problem == "What?"
        assert r.reference == "42"
        assert r.category == "reasoning"
        assert r.subject == ""
        assert r.metadata == {}

    def test_with_subject_and_metadata(self):
        r = EvalRecord(
            record_id="r2", problem="Q", reference="A",
            category="chat", subject="greet",
            metadata={"key": "val"},
        )
        assert r.subject == "greet"
        assert r.metadata == {"key": "val"}


class TestEvalResult:
    def test_defaults(self):
        r = EvalResult(record_id="r1", model_answer="42")
        assert r.is_correct is None
        assert r.score is None
        assert r.latency_seconds == 0.0
        assert r.prompt_tokens == 0
        assert r.completion_tokens == 0
        assert r.cost_usd == 0.0
        assert r.error is None
        assert r.scoring_metadata == {}

    def test_full(self):
        r = EvalResult(
            record_id="r1", model_answer="42", is_correct=True,
            score=1.0, latency_seconds=1.5, prompt_tokens=100,
            completion_tokens=50, cost_usd=0.01,
            scoring_metadata={"match": "exact"},
        )
        assert r.is_correct is True
        assert r.score == 1.0
        assert r.cost_usd == 0.01


class TestRunConfig:
    def test_defaults(self):
        c = RunConfig(benchmark="supergpqa", backend="jarvis-direct", model="qwen3:8b")
        assert c.max_samples is None
        assert c.max_workers == 4
        assert c.temperature == 0.0
        assert c.max_tokens == 2048
        assert c.judge_model == "gpt-5-mini-2025-08-07"
        assert c.seed == 42
        assert c.tools == []

    def test_with_agent(self):
        c = RunConfig(
            benchmark="gaia", backend="jarvis-agent", model="gpt-4o",
            engine_key="cloud", agent_name="orchestrator",
            tools=["calculator", "think"],
        )
        assert c.agent_name == "orchestrator"
        assert c.tools == ["calculator", "think"]


class TestRunSummary:
    def test_creation(self):
        s = RunSummary(
            benchmark="supergpqa", category="reasoning",
            backend="jarvis-direct", model="qwen3:8b",
            total_samples=100, scored_samples=95, correct=47,
            accuracy=0.495, errors=5, mean_latency_seconds=2.1,
            total_cost_usd=0.0,
            per_subject={"math": {"accuracy": 0.5}},
        )
        assert s.accuracy == 0.495
        assert s.per_subject["math"]["accuracy"] == 0.5
        assert s.started_at == 0.0


# ---------------------------------------------------------------------------
# Eval suite config dataclasses
# ---------------------------------------------------------------------------


class TestMetaConfig:
    def test_defaults(self):
        m = MetaConfig()
        assert m.name == ""
        assert m.description == ""

    def test_with_values(self):
        m = MetaConfig(name="suite-1", description="First suite")
        assert m.name == "suite-1"
        assert m.description == "First suite"


class TestDefaultsConfig:
    def test_defaults(self):
        d = DefaultsConfig()
        assert d.temperature == 0.0
        assert d.max_tokens == 2048

    def test_with_values(self):
        d = DefaultsConfig(temperature=0.7, max_tokens=4096)
        assert d.temperature == 0.7
        assert d.max_tokens == 4096


class TestJudgeConfig:
    def test_defaults(self):
        j = JudgeConfig()
        assert j.model == "gpt-5-mini-2025-08-07"
        assert j.provider is None
        assert j.temperature == 0.0
        assert j.max_tokens == 1024

    def test_with_values(self):
        j = JudgeConfig(model="claude", provider="anthropic", temperature=0.1)
        assert j.model == "claude"
        assert j.provider == "anthropic"
        assert j.temperature == 0.1


class TestExecutionConfig:
    def test_defaults(self):
        e = ExecutionConfig()
        assert e.max_workers == 4
        assert e.output_dir == "results/"
        assert e.seed == 42

    def test_with_values(self):
        e = ExecutionConfig(max_workers=16, output_dir="out/", seed=99)
        assert e.max_workers == 16
        assert e.output_dir == "out/"
        assert e.seed == 99


class TestModelConfig:
    def test_required_name(self):
        m = ModelConfig(name="qwen3:8b")
        assert m.name == "qwen3:8b"
        assert m.engine is None
        assert m.provider is None
        assert m.temperature is None
        assert m.max_tokens is None

    def test_with_overrides(self):
        m = ModelConfig(
            name="gpt-4o", engine="cloud", provider="openai",
            temperature=0.5, max_tokens=4096,
        )
        assert m.engine == "cloud"
        assert m.provider == "openai"
        assert m.temperature == 0.5
        assert m.max_tokens == 4096


class TestBenchmarkConfig:
    def test_defaults(self):
        b = BenchmarkConfig(name="supergpqa")
        assert b.name == "supergpqa"
        assert b.backend == "jarvis-direct"
        assert b.max_samples is None
        assert b.split is None
        assert b.agent is None
        assert b.tools == []
        assert b.judge_model is None
        assert b.temperature is None
        assert b.max_tokens is None

    def test_with_overrides(self):
        b = BenchmarkConfig(
            name="gaia", backend="jarvis-agent", max_samples=50,
            split="test", agent="orchestrator",
            tools=["calc", "think"], judge_model="custom-judge",
            temperature=0.3, max_tokens=1024,
        )
        assert b.backend == "jarvis-agent"
        assert b.max_samples == 50
        assert b.split == "test"
        assert b.agent == "orchestrator"
        assert b.tools == ["calc", "think"]
        assert b.judge_model == "custom-judge"
        assert b.temperature == 0.3

    def test_tools_list_independent(self):
        """Each BenchmarkConfig instance should have its own tools list."""
        b1 = BenchmarkConfig(name="a")
        b2 = BenchmarkConfig(name="b")
        b1.tools.append("calc")
        assert b2.tools == []


class TestEvalSuiteConfig:
    def test_defaults(self):
        s = EvalSuiteConfig()
        assert isinstance(s.meta, MetaConfig)
        assert isinstance(s.defaults, DefaultsConfig)
        assert isinstance(s.judge, JudgeConfig)
        assert isinstance(s.run, ExecutionConfig)
        assert s.models == []
        assert s.benchmarks == []

    def test_with_entries(self):
        s = EvalSuiteConfig(
            meta=MetaConfig(name="test"),
            models=[ModelConfig(name="m1"), ModelConfig(name="m2")],
            benchmarks=[BenchmarkConfig(name="supergpqa")],
        )
        assert s.meta.name == "test"
        assert len(s.models) == 2
        assert len(s.benchmarks) == 1
