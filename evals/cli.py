"""CLI for the OpenJarvis evaluation framework."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import click

# Registry of available benchmarks and their metadata
BENCHMARKS = {
    "supergpqa": {"category": "reasoning", "description": "SuperGPQA multiple-choice"},
    "gaia": {"category": "agentic", "description": "GAIA agentic benchmark"},
    "frames": {"category": "rag", "description": "FRAMES multi-hop RAG"},
    "wildchat": {"category": "chat", "description": "WildChat conversation quality"},
}

BACKENDS = {
    "jarvis-direct": "Engine-level inference (local or cloud)",
    "jarvis-agent": "Agent-level inference with tool calling",
}


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_backend(backend_name: str, engine_key: Optional[str],
                    agent_name: str, tools: list[str],
                    telemetry: bool = False, gpu_metrics: bool = False):
    """Construct the appropriate backend."""
    if backend_name == "jarvis-agent":
        from evals.backends.jarvis_agent import JarvisAgentBackend
        return JarvisAgentBackend(
            engine_key=engine_key,
            agent_name=agent_name,
            tools=tools,
            telemetry=telemetry,
            gpu_metrics=gpu_metrics,
        )
    else:
        from evals.backends.jarvis_direct import JarvisDirectBackend
        return JarvisDirectBackend(
            engine_key=engine_key,
            telemetry=telemetry,
            gpu_metrics=gpu_metrics,
        )


def _build_dataset(benchmark: str):
    """Construct the dataset provider for a benchmark."""
    if benchmark == "supergpqa":
        from evals.datasets.supergpqa import SuperGPQADataset
        return SuperGPQADataset()
    elif benchmark == "gaia":
        from evals.datasets.gaia import GAIADataset
        return GAIADataset()
    elif benchmark == "frames":
        from evals.datasets.frames import FRAMESDataset
        return FRAMESDataset()
    elif benchmark == "wildchat":
        from evals.datasets.wildchat import WildChatDataset
        return WildChatDataset()
    else:
        raise click.ClickException(f"Unknown benchmark: {benchmark}")


def _build_scorer(benchmark: str, judge_backend, judge_model: str):
    """Construct the scorer for a benchmark."""
    if benchmark == "supergpqa":
        from evals.scorers.supergpqa_mcq import SuperGPQAScorer
        return SuperGPQAScorer(judge_backend, judge_model)
    elif benchmark == "gaia":
        from evals.scorers.gaia_exact import GAIAScorer
        return GAIAScorer(judge_backend, judge_model)
    elif benchmark == "frames":
        from evals.scorers.frames_judge import FRAMESScorer
        return FRAMESScorer(judge_backend, judge_model)
    elif benchmark == "wildchat":
        from evals.scorers.wildchat_judge import WildChatScorer
        return WildChatScorer(judge_backend, judge_model)
    else:
        raise click.ClickException(f"Unknown benchmark: {benchmark}")


def _build_judge_backend(judge_model: str):
    """Build the judge backend (always cloud for LLM-as-judge)."""
    from evals.backends.jarvis_direct import JarvisDirectBackend
    return JarvisDirectBackend(engine_key="cloud")


def _print_summary(summary) -> None:
    """Print a single run summary."""
    click.echo(f"\n{'=' * 60}")
    click.echo(f"Benchmark: {summary.benchmark}")
    click.echo(f"Model:     {summary.model}")
    click.echo(f"Backend:   {summary.backend}")
    click.echo(f"Samples:   {summary.total_samples}")
    click.echo(f"Scored:    {summary.scored_samples}")
    click.echo(f"Correct:   {summary.correct}")
    click.echo(f"Accuracy:  {summary.accuracy:.4f}")
    click.echo(f"Errors:    {summary.errors}")
    click.echo(f"Latency:   {summary.mean_latency_seconds:.2f}s (mean)")
    click.echo(f"Cost:      ${summary.total_cost_usd:.4f}")
    if summary.per_subject:
        click.echo("\nPer-subject breakdown:")
        for subj, stats in sorted(summary.per_subject.items()):
            click.echo(f"  {subj}: {stats['accuracy']:.4f} "
                       f"({int(stats['correct'])}/{int(stats['scored'])})")
    # GPU telemetry stats
    _stats_rows = []
    for label, stats_field in [
        ("Accuracy", "accuracy_stats"),
        ("Latency (s)", "latency_stats"),
        ("TTFT (s)", "ttft_stats"),
        ("Energy (J)", "energy_stats"),
        ("Power (W)", "power_stats"),
        ("GPU Util (%)", "gpu_utilization_stats"),
        ("Throughput (tok/s)", "throughput_stats"),
        ("MFU (%)", "mfu_stats"),
        ("MBU (%)", "mbu_stats"),
        ("IPW", "ipw_stats"),
        ("IPJ", "ipj_stats"),
    ]:
        ms = getattr(summary, stats_field, None)
        if ms is not None:
            _stats_rows.append((label, ms))
    if _stats_rows:
        click.echo(f"\n{'Metric':20s} {'Mean':>10s} {'Median':>10s} "
                   f"{'Min':>10s} {'Max':>10s} {'Std':>10s}")
        click.echo(f"{'-' * 20} {'-' * 10} {'-' * 10} "
                   f"{'-' * 10} {'-' * 10} {'-' * 10}")
        for label, ms in _stats_rows:
            click.echo(f"{label:20s} {ms.mean:10.4f} {ms.median:10.4f} "
                       f"{ms.min:10.4f} {ms.max:10.4f} {ms.std:10.4f}")
    if getattr(summary, "total_energy_joules", 0.0) > 0:
        click.echo(f"\nTotal Energy: {summary.total_energy_joules:.4f} J")
    click.echo(f"{'=' * 60}")


def _run_single(config) -> object:
    """Run a single eval from a RunConfig and return the summary."""
    from evals.core.runner import EvalRunner

    eval_backend = _build_backend(
        config.backend,
        config.engine_key,
        config.agent_name or "orchestrator",
        config.tools,
        telemetry=getattr(config, "telemetry", False),
        gpu_metrics=getattr(config, "gpu_metrics", False),
    )
    dataset = _build_dataset(config.benchmark)
    judge_backend = _build_judge_backend(config.judge_model)
    scorer = _build_scorer(config.benchmark, judge_backend, config.judge_model)

    runner = EvalRunner(config, dataset, eval_backend, scorer)
    try:
        return runner.run()
    finally:
        eval_backend.close()
        judge_backend.close()


def _run_from_config(config_path: str, verbose: bool) -> None:
    """Load a TOML config and run the full models x benchmarks matrix."""
    from evals.core.config import expand_suite, load_eval_config

    suite = load_eval_config(config_path)
    run_configs = expand_suite(suite)

    suite_name = suite.meta.name or Path(config_path).stem
    click.echo(f"Suite: {suite_name}")
    if suite.meta.description:
        click.echo(f"  {suite.meta.description}")
    click.echo(f"  {len(suite.models)} model(s) x {len(suite.benchmarks)} "
               f"benchmark(s) = {len(run_configs)} run(s)\n")

    # Ensure output directory exists
    output_dir = Path(suite.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for i, rc in enumerate(run_configs, 1):
        click.echo(f"--- [{i}/{len(run_configs)}] {rc.benchmark} / {rc.model} ---")
        try:
            summary = _run_single(rc)
            summaries.append(summary)
            click.echo(f"  {summary.accuracy:.4f} "
                       f"({summary.correct}/{summary.scored_samples})")
        except Exception as exc:
            click.echo(f"  FAILED: {exc}", err=True)

    # Print overall summary table
    if summaries:
        click.echo(f"\n{'=' * 60}")
        click.echo(f"Suite Results: {suite_name}")
        click.echo(f"{'=' * 60}")
        click.echo(f"  {'Benchmark':12s} {'Model':20s} {'Accuracy':>10s} {'Scored':>8s}")
        click.echo(f"  {'-' * 12} {'-' * 20} {'-' * 10} {'-' * 8}")
        for s in summaries:
            model_display = s.model[:20]
            click.echo(f"  {s.benchmark:12s} {model_display:20s} "
                       f"{s.accuracy:10.4f} "
                       f"{s.correct}/{s.scored_samples:>5}")
        click.echo(f"{'=' * 60}")


@click.group()
def main():
    """OpenJarvis Evaluation Framework."""


@main.command()
@click.option("-c", "--config", "config_path", default=None,
              type=click.Path(), help="TOML config file for suite runs")
@click.option("-b", "--benchmark", default=None,
              type=click.Choice(list(BENCHMARKS.keys())),
              help="Benchmark to run")
@click.option("--backend", default="jarvis-direct",
              type=click.Choice(list(BACKENDS.keys())),
              help="Inference backend")
@click.option("-m", "--model", default=None, help="Model identifier")
@click.option("-e", "--engine", "engine_key", default=None,
              help="Engine key (ollama, vllm, cloud, ...)")
@click.option("--agent", "agent_name", default="orchestrator",
              help="Agent name for jarvis-agent backend")
@click.option("--tools", default="", help="Comma-separated tool names")
@click.option("-n", "--max-samples", type=int, default=None,
              help="Maximum samples to evaluate")
@click.option("-w", "--max-workers", type=int, default=4,
              help="Parallel workers")
@click.option("--judge-model", default="gpt-5-mini-2025-08-07",
              help="LLM judge model")
@click.option("-o", "--output", "output_path", default=None,
              help="Output JSONL path")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--split", "dataset_split", default=None,
              help="Dataset split override")
@click.option("--temperature", type=float, default=0.0,
              help="Generation temperature")
@click.option("--max-tokens", type=int, default=2048,
              help="Max output tokens")
@click.option("--telemetry/--no-telemetry", default=False,
              help="Enable telemetry collection during eval")
@click.option("--gpu-metrics/--no-gpu-metrics", default=False,
              help="Enable GPU metrics collection")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
@click.pass_context
def run(ctx, config_path, benchmark, backend, model, engine_key, agent_name,
        tools, max_samples, max_workers, judge_model, output_path, seed,
        dataset_split, temperature, max_tokens, telemetry, gpu_metrics, verbose):
    """Run a single benchmark evaluation, or a full suite from a TOML config."""
    _setup_logging(verbose)

    # Config-driven mode
    if config_path is not None:
        _run_from_config(config_path, verbose)
        return

    # CLI-driven mode: validate required args
    if benchmark is None:
        raise click.UsageError(
            "Missing option '-b' / '--benchmark' "
            "(required when --config is not provided)"
        )
    if model is None:
        raise click.UsageError(
            "Missing option '-m' / '--model' "
            "(required when --config is not provided)"
        )

    from evals.core.types import RunConfig

    tool_list = [t.strip() for t in tools.split(",") if t.strip()] if tools else []

    config = RunConfig(
        benchmark=benchmark,
        backend=backend,
        model=model,
        max_samples=max_samples,
        max_workers=max_workers,
        temperature=temperature,
        max_tokens=max_tokens,
        judge_model=judge_model,
        engine_key=engine_key,
        agent_name=agent_name,
        tools=tool_list,
        output_path=output_path,
        seed=seed,
        dataset_split=dataset_split,
        telemetry=telemetry,
        gpu_metrics=gpu_metrics,
    )

    summary = _run_single(config)
    _print_summary(summary)


@main.command("run-all")
@click.option("-m", "--model", required=True, help="Model identifier")
@click.option("-e", "--engine", "engine_key", default=None,
              help="Engine key")
@click.option("-n", "--max-samples", type=int, default=None,
              help="Max samples per benchmark")
@click.option("-w", "--max-workers", type=int, default=4,
              help="Parallel workers")
@click.option("--judge-model", default="gpt-5-mini-2025-08-07", help="LLM judge model")
@click.option("--output-dir", default="results/",
              help="Output directory for results")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
def run_all(model, engine_key, max_samples, max_workers, judge_model,
            output_dir, seed, verbose):
    """Run all benchmarks."""
    _setup_logging(verbose)

    from evals.core.runner import EvalRunner
    from evals.core.types import RunConfig

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    model_slug = model.replace("/", "-").replace(":", "-")
    summaries = []

    for bench_name in BENCHMARKS:
        click.echo(f"\n--- Running {bench_name} ---")
        output_path = output_dir_path / f"{bench_name}_{model_slug}.jsonl"

        config = RunConfig(
            benchmark=bench_name,
            backend="jarvis-direct",
            model=model,
            max_samples=max_samples,
            max_workers=max_workers,
            judge_model=judge_model,
            engine_key=engine_key,
            output_path=str(output_path),
            seed=seed,
        )

        eval_backend = _build_backend("jarvis-direct", engine_key, "orchestrator", [])
        dataset = _build_dataset(bench_name)
        judge_backend = _build_judge_backend(judge_model)
        scorer = _build_scorer(bench_name, judge_backend, judge_model)

        runner = EvalRunner(config, dataset, eval_backend, scorer)
        try:
            summary = runner.run()
            summaries.append(summary)
            click.echo(f"  {bench_name}: {summary.accuracy:.4f} "
                       f"({summary.correct}/{summary.scored_samples})")
        except Exception as exc:
            click.echo(f"  {bench_name}: FAILED — {exc}", err=True)
        finally:
            eval_backend.close()
            judge_backend.close()

    # Print overall summary
    if summaries:
        click.echo(f"\n{'=' * 60}")
        click.echo("Overall Results:")
        for s in summaries:
            click.echo(f"  {s.benchmark:12s} {s.accuracy:.4f} "
                       f"({s.correct}/{s.scored_samples})")
        click.echo(f"{'=' * 60}")


@main.command()
@click.argument("jsonl_path", type=click.Path(exists=True))
def summarize(jsonl_path):
    """Summarize results from a JSONL output file."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        click.echo("No records found.")
        return

    total = len(records)
    scored = [r for r in records if r.get("is_correct") is not None]
    correct = [r for r in scored if r["is_correct"]]
    errors = [r for r in records if r.get("error")]
    accuracy = len(correct) / len(scored) if scored else 0.0

    click.echo(f"File:     {jsonl_path}")
    click.echo(f"Benchmark: {records[0].get('benchmark', '?')}")
    click.echo(f"Model:     {records[0].get('model', '?')}")
    click.echo(f"Total:     {total}")
    click.echo(f"Scored:    {len(scored)}")
    click.echo(f"Correct:   {len(correct)}")
    click.echo(f"Accuracy:  {accuracy:.4f}")
    click.echo(f"Errors:    {len(errors)}")


@main.command("list")
def list_cmd():
    """List available benchmarks and backends."""
    click.echo("Benchmarks:")
    for name, info in BENCHMARKS.items():
        click.echo(f"  {name:12s} [{info['category']:10s}] {info['description']}")

    click.echo("\nBackends:")
    for name, desc in BACKENDS.items():
        click.echo(f"  {name:16s} {desc}")


if __name__ == "__main__":
    main()
