"""TOML config loader and matrix expansion for eval suites."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

from evals.core.types import (
    BenchmarkConfig,
    DefaultsConfig,
    EvalSuiteConfig,
    ExecutionConfig,
    JudgeConfig,
    MetaConfig,
    ModelConfig,
    RunConfig,
)

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError as exc:
        raise ImportError(
            "Python 3.10 requires the 'tomli' package. "
            "Install it with: pip install tomli"
        ) from exc

logger = logging.getLogger(__name__)

VALID_BACKENDS = {"jarvis-direct", "jarvis-agent"}

# Known benchmark names (used for warnings, not hard validation)
KNOWN_BENCHMARKS = {"supergpqa", "gaia", "frames", "wildchat"}


class EvalConfigError(Exception):
    """Raised for structural issues in eval config files."""


def load_eval_config(path: str | Path) -> EvalSuiteConfig:
    """Load and validate an eval suite config from a TOML file.

    Args:
        path: Path to the TOML config file.

    Returns:
        Validated EvalSuiteConfig.

    Raises:
        EvalConfigError: On structural validation failures.
        FileNotFoundError: If the config file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    # Parse [meta]
    meta_raw = raw.get("meta", {})
    meta = MetaConfig(
        name=meta_raw.get("name", ""),
        description=meta_raw.get("description", ""),
    )

    # Parse [defaults]
    defaults_raw = raw.get("defaults", {})
    defaults = DefaultsConfig(
        temperature=float(defaults_raw.get("temperature", 0.0)),
        max_tokens=int(defaults_raw.get("max_tokens", 2048)),
    )

    # Parse [judge]
    judge_raw = raw.get("judge", {})
    judge = JudgeConfig(
        model=judge_raw.get("model", "gpt-5-mini-2025-08-07"),
        provider=judge_raw.get("provider"),
        temperature=float(judge_raw.get("temperature", 0.0)),
        max_tokens=int(judge_raw.get("max_tokens", 1024)),
    )

    # Parse [run]
    run_raw = raw.get("run", {})
    execution = ExecutionConfig(
        max_workers=int(run_raw.get("max_workers", 4)),
        output_dir=run_raw.get("output_dir", "results/"),
        seed=int(run_raw.get("seed", 42)),
        telemetry=bool(run_raw.get("telemetry", False)),
        gpu_metrics=bool(run_raw.get("gpu_metrics", False)),
    )

    # Parse [[models]]
    models_raw = raw.get("models", [])
    if not models_raw:
        raise EvalConfigError("Config must define at least one [[models]] entry")

    models: List[ModelConfig] = []
    for m in models_raw:
        if not m.get("name"):
            raise EvalConfigError("Each [[models]] entry must have a 'name' field")
        models.append(ModelConfig(
            name=m["name"],
            engine=m.get("engine"),
            provider=m.get("provider"),
            temperature=float(m["temperature"]) if "temperature" in m else None,
            max_tokens=int(m["max_tokens"]) if "max_tokens" in m else None,
        ))

    # Parse [[benchmarks]]
    benchmarks_raw = raw.get("benchmarks", [])
    if not benchmarks_raw:
        raise EvalConfigError(
            "Config must define at least one [[benchmarks]] entry"
        )

    benchmarks: List[BenchmarkConfig] = []
    for b in benchmarks_raw:
        if not b.get("name"):
            raise EvalConfigError(
                "Each [[benchmarks]] entry must have a 'name' field"
            )

        backend = b.get("backend", "jarvis-direct")
        if backend not in VALID_BACKENDS:
            raise EvalConfigError(
                f"Invalid backend '{backend}' for benchmark '{b['name']}'. "
                f"Must be one of: {', '.join(sorted(VALID_BACKENDS))}"
            )

        bench_name = b["name"]
        if bench_name not in KNOWN_BENCHMARKS:
            logger.warning("Unknown benchmark name: '%s'", bench_name)

        tools_raw = b.get("tools", [])
        benchmarks.append(BenchmarkConfig(
            name=bench_name,
            backend=backend,
            max_samples=int(b["max_samples"]) if "max_samples" in b else None,
            split=b.get("split"),
            agent=b.get("agent"),
            tools=list(tools_raw),
            judge_model=b.get("judge_model"),
            temperature=float(b["temperature"]) if "temperature" in b else None,
            max_tokens=int(b["max_tokens"]) if "max_tokens" in b else None,
        ))

    return EvalSuiteConfig(
        meta=meta,
        defaults=defaults,
        judge=judge,
        run=execution,
        models=models,
        benchmarks=benchmarks,
    )


def expand_suite(suite: EvalSuiteConfig) -> List[RunConfig]:
    """Expand an EvalSuiteConfig into a list of RunConfigs (models x benchmarks).

    Merge precedence (highest wins):
        benchmark-level > model-level > [defaults] > built-in defaults

    Args:
        suite: The parsed eval suite config.

    Returns:
        List of RunConfig, one per model-benchmark pair.
    """
    configs: List[RunConfig] = []
    output_dir = suite.run.output_dir.rstrip("/")

    for model in suite.models:
        for bench in suite.benchmarks:
            # Temperature: benchmark > model > defaults
            temperature = suite.defaults.temperature
            if model.temperature is not None:
                temperature = model.temperature
            if bench.temperature is not None:
                temperature = bench.temperature

            # Max tokens: benchmark > model > defaults
            max_tokens = suite.defaults.max_tokens
            if model.max_tokens is not None:
                max_tokens = model.max_tokens
            if bench.max_tokens is not None:
                max_tokens = bench.max_tokens

            # Judge model: benchmark > [judge]
            judge_model = suite.judge.model
            if bench.judge_model is not None:
                judge_model = bench.judge_model

            # Auto-generate output path
            model_slug = model.name.replace("/", "-").replace(":", "-")
            output_path = f"{output_dir}/{bench.name}_{model_slug}.jsonl"

            configs.append(RunConfig(
                benchmark=bench.name,
                backend=bench.backend,
                model=model.name,
                max_samples=bench.max_samples,
                max_workers=suite.run.max_workers,
                temperature=temperature,
                max_tokens=max_tokens,
                judge_model=judge_model,
                engine_key=model.engine,
                agent_name=bench.agent,
                tools=list(bench.tools),
                output_path=output_path,
                seed=suite.run.seed,
                dataset_split=bench.split,
                telemetry=suite.run.telemetry,
                gpu_metrics=suite.run.gpu_metrics,
            ))

    return configs


__all__ = ["EvalConfigError", "load_eval_config", "expand_suite"]
