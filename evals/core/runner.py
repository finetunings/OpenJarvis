"""EvalRunner — parallel execution of evaluation samples."""

from __future__ import annotations

import json
import logging
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from evals.core.backend import InferenceBackend
from evals.core.dataset import DatasetProvider
from evals.core.scorer import Scorer
from evals.core.types import EvalRecord, EvalResult, MetricStats, RunConfig, RunSummary

LOGGER = logging.getLogger(__name__)


class EvalRunner:
    """Runs an evaluation benchmark with parallel sample execution."""

    def __init__(
        self,
        config: RunConfig,
        dataset: DatasetProvider,
        backend: InferenceBackend,
        scorer: Scorer,
    ) -> None:
        self._config = config
        self._dataset = dataset
        self._backend = backend
        self._scorer = scorer
        self._results: List[EvalResult] = []
        self._output_file: Optional[Any] = None

    def run(self) -> RunSummary:
        """Execute the evaluation and return a summary."""
        cfg = self._config
        started_at = time.time()

        self._dataset.load(
            max_samples=cfg.max_samples,
            split=cfg.dataset_split,
            seed=cfg.seed,
        )
        records = list(self._dataset.iter_records())
        LOGGER.info(
            "Running %s: %d samples, backend=%s, model=%s, workers=%d",
            cfg.benchmark, len(records), cfg.backend, cfg.model, cfg.max_workers,
        )

        # Open output file for incremental JSONL writing
        output_path = self._resolve_output_path()
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._output_file = open(output_path, "w")

        try:
            with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
                futures = {
                    pool.submit(self._process_one, r): r for r in records
                }
                for future in as_completed(futures):
                    result = future.result()
                    self._results.append(result)
                    self._flush_result(result)
        finally:
            if self._output_file:
                self._output_file.close()
                self._output_file = None

        ended_at = time.time()
        summary = self._compute_summary(records, started_at, ended_at)

        # Write summary JSON alongside JSONL
        if output_path:
            summary_path = output_path.with_suffix(".summary.json")
            with open(summary_path, "w") as f:
                json.dump(_summary_to_dict(summary), f, indent=2)
            LOGGER.info("Results written to %s", output_path)
            LOGGER.info("Summary written to %s", summary_path)

        return summary

    def _process_one(self, record: EvalRecord) -> EvalResult:
        """Process a single evaluation sample."""
        cfg = self._config
        try:
            full = self._backend.generate_full(
                record.problem,
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            content = full.get("content", "")
            usage = full.get("usage", {})
            latency = full.get("latency_seconds", 0.0)
            cost = full.get("cost_usd", 0.0)

            is_correct, scoring_meta = self._scorer.score(record, content)

            return EvalResult(
                record_id=record.record_id,
                model_answer=content,
                is_correct=is_correct,
                score=1.0 if is_correct else (0.0 if is_correct is not None else None),
                latency_seconds=latency,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                cost_usd=cost,
                scoring_metadata=scoring_meta,
                ttft=full.get("ttft", 0.0),
                energy_joules=full.get("energy_joules", 0.0),
                power_watts=full.get("power_watts", 0.0),
                gpu_utilization_pct=full.get("gpu_utilization_pct", 0.0),
                throughput_tok_per_sec=full.get("throughput_tok_per_sec", 0.0),
            )
        except Exception as exc:
            LOGGER.error("Error processing %s: %s", record.record_id, exc)
            return EvalResult(
                record_id=record.record_id,
                model_answer="",
                error=str(exc),
            )

    def _flush_result(self, result: EvalResult) -> None:
        """Append a single result to the output JSONL file."""
        if not self._output_file:
            return
        record_dict = {
            "record_id": result.record_id,
            "benchmark": self._config.benchmark,
            "model": self._config.model,
            "backend": self._config.backend,
            "model_answer": result.model_answer,
            "is_correct": result.is_correct,
            "score": result.score,
            "latency_seconds": result.latency_seconds,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "cost_usd": result.cost_usd,
            "error": result.error,
            "scoring_metadata": result.scoring_metadata,
            "ttft": result.ttft,
            "energy_joules": result.energy_joules,
            "power_watts": result.power_watts,
            "gpu_utilization_pct": result.gpu_utilization_pct,
            "throughput_tok_per_sec": result.throughput_tok_per_sec,
            "mfu_pct": result.mfu_pct,
            "mbu_pct": result.mbu_pct,
            "ipw": result.ipw,
            "ipj": result.ipj,
        }
        self._output_file.write(json.dumps(record_dict) + "\n")
        self._output_file.flush()

    def _resolve_output_path(self) -> Optional[Path]:
        """Determine the output file path."""
        if self._config.output_path:
            return Path(self._config.output_path)
        # Auto-generate based on benchmark + model
        model_slug = self._config.model.replace("/", "-").replace(":", "-")
        name = f"{self._config.benchmark}_{model_slug}.jsonl"
        return Path(name)

    def _compute_summary(
        self,
        records: List[EvalRecord],
        started_at: float,
        ended_at: float,
    ) -> RunSummary:
        """Compute aggregate statistics from results."""
        cfg = self._config
        results = self._results

        scored = [r for r in results if r.is_correct is not None]
        correct = [r for r in scored if r.is_correct]
        errors = [r for r in results if r.error]

        latencies = [r.latency_seconds for r in results if r.latency_seconds > 0]
        mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
        total_cost = sum(r.cost_usd for r in results)

        # Per-subject breakdown
        record_map = {r.record_id: r for r in records}
        subject_groups: Dict[str, List[EvalResult]] = defaultdict(list)
        for r in results:
            rec = record_map.get(r.record_id)
            subj = rec.subject if rec and rec.subject else "general"
            subject_groups[subj].append(r)

        per_subject: Dict[str, Dict[str, float]] = {}
        for subj, subj_results in sorted(subject_groups.items()):
            subj_scored = [r for r in subj_results if r.is_correct is not None]
            subj_correct = [r for r in subj_scored if r.is_correct]
            subj_acc = len(subj_correct) / len(subj_scored) if subj_scored else 0.0
            per_subject[subj] = {
                "accuracy": round(subj_acc, 4),
                "total": float(len(subj_results)),
                "scored": float(len(subj_scored)),
                "correct": float(len(subj_correct)),
            }

        # Determine category from records
        categories = {r.category for r in records}
        category = categories.pop() if len(categories) == 1 else cfg.benchmark

        accuracy = len(correct) / len(scored) if scored else 0.0

        # Compute MetricStats for each metric
        accuracy_vals = [1.0 if r.is_correct else 0.0 for r in scored]
        latency_vals = [r.latency_seconds for r in results if r.latency_seconds > 0]
        ttft_vals = [r.ttft for r in results if r.ttft > 0]
        energy_vals = [r.energy_joules for r in results if r.energy_joules > 0]
        power_vals = [r.power_watts for r in results if r.power_watts > 0]
        gpu_util_vals = [r.gpu_utilization_pct for r in results if r.gpu_utilization_pct > 0]
        throughput_vals = [r.throughput_tok_per_sec for r in results if r.throughput_tok_per_sec > 0]
        mfu_vals = [r.mfu_pct for r in results if r.mfu_pct > 0]
        mbu_vals = [r.mbu_pct for r in results if r.mbu_pct > 0]
        ipw_vals = [r.ipw for r in results if r.ipw > 0]
        ipj_vals = [r.ipj for r in results if r.ipj > 0]

        total_energy = sum(r.energy_joules for r in results)

        return RunSummary(
            benchmark=cfg.benchmark,
            category=category,
            backend=cfg.backend,
            model=cfg.model,
            total_samples=len(results),
            scored_samples=len(scored),
            correct=len(correct),
            accuracy=round(accuracy, 4),
            errors=len(errors),
            mean_latency_seconds=round(mean_latency, 4),
            total_cost_usd=round(total_cost, 6),
            per_subject=per_subject,
            started_at=started_at,
            ended_at=ended_at,
            accuracy_stats=_metric_stats(accuracy_vals),
            latency_stats=_metric_stats(latency_vals),
            ttft_stats=_metric_stats(ttft_vals),
            energy_stats=_metric_stats(energy_vals),
            power_stats=_metric_stats(power_vals),
            gpu_utilization_stats=_metric_stats(gpu_util_vals),
            throughput_stats=_metric_stats(throughput_vals),
            mfu_stats=_metric_stats(mfu_vals),
            mbu_stats=_metric_stats(mbu_vals),
            ipw_stats=_metric_stats(ipw_vals),
            ipj_stats=_metric_stats(ipj_vals),
            total_energy_joules=round(total_energy, 6),
        )


def _metric_stats(values: List[float]) -> Optional[MetricStats]:
    """Compute MetricStats from a list of float values."""
    if not values:
        return None
    return MetricStats(
        mean=statistics.mean(values),
        median=statistics.median(values),
        min=min(values),
        max=max(values),
        std=statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def _metric_stats_to_dict(ms: Optional[MetricStats]) -> Optional[Dict[str, float]]:
    """Convert MetricStats to a JSON-serializable dict."""
    if ms is None:
        return None
    return {
        "mean": ms.mean,
        "median": ms.median,
        "min": ms.min,
        "max": ms.max,
        "std": ms.std,
    }


def _summary_to_dict(s: RunSummary) -> Dict[str, Any]:
    """Convert a RunSummary to a JSON-serializable dict."""
    return {
        "benchmark": s.benchmark,
        "category": s.category,
        "backend": s.backend,
        "model": s.model,
        "total_samples": s.total_samples,
        "scored_samples": s.scored_samples,
        "correct": s.correct,
        "accuracy": s.accuracy,
        "errors": s.errors,
        "mean_latency_seconds": s.mean_latency_seconds,
        "total_cost_usd": s.total_cost_usd,
        "per_subject": s.per_subject,
        "started_at": s.started_at,
        "ended_at": s.ended_at,
        "accuracy_stats": _metric_stats_to_dict(s.accuracy_stats),
        "latency_stats": _metric_stats_to_dict(s.latency_stats),
        "ttft_stats": _metric_stats_to_dict(s.ttft_stats),
        "energy_stats": _metric_stats_to_dict(s.energy_stats),
        "power_stats": _metric_stats_to_dict(s.power_stats),
        "gpu_utilization_stats": _metric_stats_to_dict(s.gpu_utilization_stats),
        "throughput_stats": _metric_stats_to_dict(s.throughput_stats),
        "mfu_stats": _metric_stats_to_dict(s.mfu_stats),
        "mbu_stats": _metric_stats_to_dict(s.mbu_stats),
        "ipw_stats": _metric_stats_to_dict(s.ipw_stats),
        "ipj_stats": _metric_stats_to_dict(s.ipj_stats),
        "total_energy_joules": s.total_energy_joules,
    }


__all__ = ["EvalRunner"]
