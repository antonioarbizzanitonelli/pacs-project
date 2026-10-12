"""Measure unchanged Stage-1 algorithms on shared, explicitly synthetic inputs.

Timing includes public runner validation and P_DM problem/model construction.
Warmups are retained but excluded from summaries. Failed or independently
rejected outcomes remain evidence; only accepted outcomes enter quality means.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from math import isfinite
import os
from pathlib import Path
import platform
from statistics import median, quantiles
from time import perf_counter

from llm_resource_allocation.algorithms.agh import run_agh
from llm_resource_allocation.algorithms.gh import run_gh
from llm_resource_allocation.algorithms.gh_phase1 import GHPhase1Config
from llm_resource_allocation.data.synthetic_benchmark_instances import (
    SyntheticBenchmarkCase, make_synthetic_benchmark_cases,
)
from llm_resource_allocation.math.objective import ObjectiveBreakdown, PaperVersion
from llm_resource_allocation.math.placement import PlacementDecisionState
from llm_resource_allocation.optimization.pdm_run import execute_and_record_pdm_run
from llm_resource_allocation.optimization.problem_data import PdmProblemData
from llm_resource_allocation.optimization.solver import (
    NoSolverIncumbentError, SolverBackend, SolverConfig, SolverResult, SolverStatus,
)
from llm_resource_allocation.validation.placement import PlacementValidationReport

METHODS = ("P_DM", "GH", "AGH")
UNMET_REPORTING_THRESHOLD = 0.01


@dataclass(frozen=True, slots=True)
class SyntheticBenchmarkConfig:
    """ASSUMED measurement choices; scientific runner defaults are preserved."""

    seeds: tuple[int, ...] = (37, 41, 43)
    query_counts: tuple[int, ...] = (2, 4, 8, 16)
    warmup_repeats: int = 1
    gh_budget_fraction: float = 0.8
    backend: SolverBackend = SolverBackend.HIGHS
    solver_time_limit_seconds: float = 20.0
    paper_version: PaperVersion = PaperVersion.V2
    data_residency_hours: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "query_counts", tuple(self.query_counts))
        if not self.seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in self.seeds
        ):
            raise ValueError("seeds must be non-empty non-negative integers")
        if (isinstance(self.warmup_repeats, bool)
                or not isinstance(self.warmup_repeats, int)
                or self.warmup_repeats < 0):
            raise ValueError("warmup_repeats must be a non-negative integer")
        if self.paper_version is not PaperVersion.V2:
            raise ValueError("coherent benchmarks require PaperVersion.V2")
        if (isinstance(self.data_residency_hours, bool)
                or not isinstance(self.data_residency_hours, (int, float))
                or not isfinite(self.data_residency_hours)
                or self.data_residency_hours < 0):
            raise ValueError("data_residency_hours must be finite and non-negative")
        GHPhase1Config(self.gh_budget_fraction)
        if (isinstance(self.solver_time_limit_seconds, bool)
                or not isinstance(self.solver_time_limit_seconds, (int, float))
                or not isfinite(self.solver_time_limit_seconds)
                or self.solver_time_limit_seconds <= 0):
            raise ValueError("solver_time_limit_seconds must be finite and positive")
        SolverConfig(backend=self.backend)
        counts = self.query_counts
        if (any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in counts)
                or len(set(counts)) != len(counts)):
            raise ValueError("query_counts must be distinct positive integers")


@dataclass(frozen=True, slots=True)
class SyntheticBenchmarkRun:
    """Retain plain decision/validation evidence without an AGH search graph."""

    case_id: str
    family: str
    method: str
    repeat: int
    seed: int
    elapsed_seconds: float
    completed: bool
    accepted: bool
    status: str
    placement: PlacementDecisionState | None
    validation: PlacementValidationReport | None
    solver: SolverResult | None
    objective_agrees: bool | None
    instance_sha256: str
    error_type: str | None = None
    error_message: str | None = None
    agh_executed_start_count: int | None = None
    agh_unexecuted_start_count: int | None = None
    agh_scheduled_random_start_count: int | None = None
    agh_executed_random_start_count: int | None = None
    agh_strict_improvement_count: int | None = None

    @property
    def is_warmup(self) -> bool:
        """Identify warmup rounds by their negative repeat indices."""

        return self.repeat < 0

    @property
    def independently_feasible(self) -> bool:
        return self.validation is not None and self.validation.is_feasible

    @property
    def objective(self) -> ObjectiveBreakdown | None:
        return None if self.validation is None else self.validation.objective_breakdown

    @property
    def optimal(self) -> bool:
        return (self.method == "P_DM" and self.accepted and self.solver is not None
                and self.solver.status is SolverStatus.OPTIMAL)


def _run_one(
    case: SyntheticBenchmarkCase, config: SyntheticBenchmarkConfig,
    method: str, repeat: int, seed: int,
) -> SyntheticBenchmarkRun:
    fingerprint = _instance_fingerprint(case.instance)
    start = perf_counter()
    gh_config = GHPhase1Config(config.gh_budget_fraction)
    if method == "P_DM":
        try:
            result = execute_and_record_pdm_run(
                PdmProblemData(case.instance),
                SolverConfig(backend=config.backend,
                             time_limit_seconds=config.solver_time_limit_seconds,
                             relative_mip_gap=0.0, absolute_mip_gap=0.0),
                config.paper_version,
            )
        except NoSolverIncumbentError as error:
            return SyntheticBenchmarkRun(
                case.case_id, case.family, method, repeat, seed,
                perf_counter() - start, False, False, "no_solver_incumbent",
                None, None, None, None, fingerprint, type(error).__name__, str(error),
            )
    elif method == "GH":
        result = run_gh(case.instance, gh_config, paper_version=config.paper_version)
    elif method == "AGH":
        result = run_agh(case.instance, gh_config, random_seed=seed,
                         paper_version=config.paper_version)
    else:
        raise ValueError(f"unknown benchmark method: {method!r}")
    elapsed = perf_counter() - start
    if method == "P_DM":
        completed, accepted = True, result.accepted_run
        state, report = result.solution, result.scientific_evidence.validation_report
        solver, agrees = result.solver_evidence.result, result.objective_agrees
        status = solver.status.value
    elif method == "GH":
        completed = result.algorithm_completed
        accepted = completed and result.independently_feasible
        state, report = result.placement_decisions, result.validation_report
        solver, agrees, status = None, None, result.completion_status.value
    else:
        completed = result.completed
        report, state = result.final_validation, result.final_placement
        accepted = completed and report is not None and report.is_feasible
        solver, agrees, status = None, None, result.completion_status.value
    agh_counts = {}
    if method == "AGH":
        traces = result.multistart_result.start_traces
        agh_counts = dict(
            agh_executed_start_count=result.executed_start_count,
            agh_unexecuted_start_count=result.unexecuted_start_count,
            agh_scheduled_random_start_count=result.scheduled_random_start_count,
            agh_executed_random_start_count=sum(
                trace.ordering.source.value == "seeded_random" for trace in traces),
            agh_strict_improvement_count=sum(trace.strict_improvement for trace in traces),
        )
    return SyntheticBenchmarkRun(
        case.case_id, case.family, method, repeat, seed, elapsed,
        completed, accepted, status, state, report, solver, agrees, fingerprint,
        **agh_counts,
    )


def run_synthetic_benchmark(
    config: SyntheticBenchmarkConfig,
    cases: tuple[SyntheticBenchmarkCase, ...] | None = None,
    progress: Callable[[str], None] | None = None,
    *,
    on_run: Callable[[SyntheticBenchmarkRun], None] | None = None,
) -> tuple[SyntheticBenchmarkRun, ...]:
    """Run all methods serially on each exact shared instance; preserve failures.

    Warmups use the first seed. Measured rounds use caller-ordered seeds and
    rotate method execution order to reduce systematic order bias. No retries.
    """
    if not isinstance(config, SyntheticBenchmarkConfig):
        raise TypeError("config must be SyntheticBenchmarkConfig")
    selected = (make_synthetic_benchmark_cases(
                    config.query_counts, data_residency_hours=config.data_residency_hours)
                if cases is None else tuple(cases))
    if any(not isinstance(case, SyntheticBenchmarkCase) for case in selected):
        raise TypeError("cases must contain SyntheticBenchmarkCase records")
    if not selected or len({case.case_id for case in selected}) != len(selected):
        raise ValueError("cases must be non-empty with unique case IDs")
    runs = []
    for case in selected:
        if progress is not None:
            progress(f"{case.case_id}: {case.instance.dimensions}")
        for repeat in range(-config.warmup_repeats, len(config.seeds)):
            warmup = repeat < 0
            seed = config.seeds[0] if warmup else config.seeds[repeat]
            offset = 0 if warmup else repeat % len(METHODS)
            order = METHODS[offset:] + METHODS[:offset]
            for method in order:
                if progress is not None:
                    progress(f"{case.case_id}: {method} repeat={repeat} seed={seed} starting")
                run = _run_one(case, config, method, repeat, seed)
                runs.append(run)
                if on_run is not None:
                    on_run(run)
                if progress is not None:
                    progress(f"{case.case_id}: {method} status={run.status} "
                             f"accepted={run.accepted} time={run.elapsed_seconds:.3f}s")
    return tuple(runs)


def objective_gap_percent(
    run: SyntheticBenchmarkRun, baseline: SyntheticBenchmarkRun | None,
) -> float | None:
    """Return signed 100*(J-J*)/abs(J*) only for comparable accepted optima.

    A zero reference makes the relative gap undefined. Numerical negative
    values are retained, not clipped. Solver optimum concerns the built model.
    """
    if baseline is None or not baseline.optimal:
        return None
    return pdm_incumbent_comparison_percent(run, baseline)


def pdm_incumbent_comparison_percent(
    run: SyntheticBenchmarkRun, baseline: SyntheticBenchmarkRun | None,
) -> float | None:
    """Compare accepted costs with the same run's accepted P_DM incumbent.

    Return signed ``100*(J-J_incumbent)/abs(J_incumbent)``. A feasible
    time-limited incumbent is eligible; this is not an optimality gap.
    Zero reference objectives remain undefined.
    """
    if (not run.accepted or run.objective is None or baseline is None
            or baseline.method != "P_DM" or not baseline.accepted
            or baseline.objective is None or baseline.solver is None
            or not baseline.solver.has_incumbent):
        return None
    if (
        run.validation.paper_version is not baseline.validation.paper_version
        or run.validation.unit_convention is not baseline.validation.unit_convention
        or run.validation.data_residency_hours != baseline.validation.data_residency_hours
        or run.instance_sha256 != baseline.instance_sha256
        or (run.case_id, run.repeat, run.seed, run.is_warmup)
        != (baseline.case_id, baseline.repeat, baseline.seed, baseline.is_warmup)
    ):
        return None
    reference = baseline.objective.total
    if reference == 0.0:
        return None
    return 100.0 * (run.objective.total - reference) / abs(reference)


def _median_or_none(values: list[float]) -> float | None:
    return median(values) if values else None


def _quartiles(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    q1, _, q3 = quantiles(values, n=4, method="inclusive")
    return q1, q3


def _service_metrics(run: SyntheticBenchmarkRun) -> tuple[float | None, float | None]:
    if run.placement is None or run.validation is None or not run.validation.domain_diagnostic.is_valid:
        return None, None
    values = tuple(run.placement.unmet_fractions.get(query, 0.0)
                   for query in run.validation.processing_delays_seconds)
    if not values:
        return None, None
    return (sum(values) / len(values),
            sum(value > UNMET_REPORTING_THRESHOLD for value in values) / len(values))


def summarize_benchmark(
    runs: tuple[SyntheticBenchmarkRun, ...], cases: tuple[SyntheticBenchmarkCase, ...],
) -> tuple[dict[str, object], ...]:
    """Summarize every measured attempt; quality summaries use accepted runs."""
    measured = tuple(run for run in runs if not run.is_warmup)
    baselines = {(run.case_id, run.repeat): run for run in measured if run.method == "P_DM"}
    rows = []
    for case in cases:
        for method in METHODS:
            group = [run for run in measured if run.case_id == case.case_id and run.method == method]
            if not group:
                continue
            accepted = [run for run in group if run.accepted]
            times = [run.elapsed_seconds for run in group]
            q1, q3 = _quartiles(times)
            gaps = [value for run in accepted
                    if (value := objective_gap_percent(run, baselines.get((run.case_id, run.repeat)))) is not None]
            comparisons = [value for run in accepted
                           if (value := pdm_incumbent_comparison_percent(
                               run, baselines.get((run.case_id, run.repeat)))) is not None]
            service = [_service_metrics(run) for run in accepted]
            row = dict(case_id=case.case_id, family=case.family, method=method,
                       query_count=case.instance.dimensions[0],
                       model_count=case.instance.dimensions[1], tier_count=case.instance.dimensions[2],
                       run_count=len(group), accepted_count=len(accepted),
                       feasible_count=sum(run.independently_feasible for run in group),
                       optimal_count=sum(run.optimal for run in group),
                       time_median_seconds=median(times), time_q1_seconds=q1, time_q3_seconds=q3,
                       objective_median=_median_or_none([run.objective.total for run in accepted]),
                       gap_median_percent=_median_or_none(gaps), gap_count=len(gaps),
                       pdm_incumbent_comparison_median_percent=_median_or_none(comparisons),
                       pdm_incumbent_comparison_count=len(comparisons),
                       mean_unmet_fraction_median=_median_or_none([item[0] for item in service if item[0] is not None]),
                       unmet_over_1pct_rate_median=_median_or_none([item[1] for item in service if item[1] is not None]))
            for index, attribute in enumerate(("gpu_rental_cost", "model_storage_cost", "data_storage_cost", "delay_penalty", "unmet_demand_penalty"), 1):
                row[f"c{index}_median"] = _median_or_none([getattr(run.objective, attribute) for run in accepted])
            rows.append(row)
    return tuple(rows)


def _plain(value: object) -> object:
    """Serialize domain mappings losslessly, including tuple-indexed tensors."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        result = {descriptor.name: _plain(getattr(value, descriptor.name))
                  for descriptor in fields(value)
                  if not descriptor.name.startswith("_")}
        if isinstance(value, SyntheticBenchmarkRun):
            result.update(is_warmup=value.is_warmup,
                          independently_feasible=value.independently_feasible,
                          optimal=value.optimal, objective=_plain(value.objective))
        elif isinstance(value, ObjectiveBreakdown):
            result["total"] = value.total
        elif isinstance(value, PlacementValidationReport):
            result["is_feasible"] = value.is_feasible
            result["violations"] = _plain(value.violations)
            result["max_violation_by_family"] = _plain(value.max_violation_by_family)
        return result
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: _plain(item) for key, item in value.items()}
        return [{"key": _plain(key), "value": _plain(item)} for key, item in value.items()]
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _instance_fingerprint(instance: object) -> str:
    """Hash the complete immutable input, including unit policy and coefficients."""

    payload = json.dumps(_plain(instance), sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def serialize_benchmark_run(run: SyntheticBenchmarkRun) -> dict[str, object]:
    """Return plain per-attempt evidence without an algorithm search graph."""

    return _plain(run)


def _write_csv(path: Path, rows: list[dict[str, object]] | tuple[dict[str, object], ...]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_benchmark_results(
    output_dir: Path, config: SyntheticBenchmarkConfig,
    cases: tuple[SyntheticBenchmarkCase, ...], runs: tuple[SyntheticBenchmarkRun, ...],
    repository_root: Path,
) -> tuple[dict[str, object], ...]:
    """Write manifest, plain decisions/validation, raw observations and summaries.

    Search trace graphs are deliberately absent. The recorded placement and
    complete independent report preserve scientific evidence for each outcome.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    baselines = {(run.case_id, run.repeat, run.is_warmup): run for run in runs if run.method == "P_DM"}
    raw_rows, query_rows = [], []
    with (output_dir / "evidence.jsonl").open("w", encoding="utf-8") as handle:
        for run in runs:
            baseline = baselines.get((run.case_id, run.repeat, run.is_warmup))
            gap = objective_gap_percent(run, baseline)
            evidence = _plain(run)
            evidence["gap_percent"] = gap
            comparison = pdm_incumbent_comparison_percent(run, baseline)
            evidence["pdm_incumbent_comparison_percent"] = comparison
            handle.write(json.dumps(evidence, ensure_ascii=False, allow_nan=False) + "\n")
            mean_unmet, rate = _service_metrics(run)
            row = dict(case_id=run.case_id, family=run.family, method=run.method,
                       repeat=run.repeat, seed=run.seed, is_warmup=run.is_warmup,
                       instance_sha256=run.instance_sha256,
                       time_seconds=run.elapsed_seconds, completed=run.completed,
                       accepted=run.accepted, independently_feasible=run.independently_feasible,
                       optimal=run.optimal, status=run.status, objective_agrees=run.objective_agrees,
                       solver_time_seconds=None if run.solver is None else run.solver.solve_time_seconds,
                       solver_status=None if run.solver is None else run.solver.status.value,
                       solver_termination=None if run.solver is None else run.solver.termination_condition,
                       solver_best_bound=None if run.solver is None else run.solver.best_bound,
                       solver_relative_gap=None if run.solver is None else run.solver.relative_gap,
                       objective_total=None if run.objective is None else run.objective.total,
                       gap_percent=gap, pdm_incumbent_comparison_percent=comparison,
                       mean_unmet_fraction=mean_unmet,
                       unmet_over_1pct_rate=rate, error_type=run.error_type, error_message=run.error_message,
                       violated_families="" if run.validation is None else "|".join(item.value for item in run.validation.violated_families))
            for index, attribute in enumerate(("gpu_rental_cost", "model_storage_cost", "data_storage_cost", "delay_penalty", "unmet_demand_penalty"), 1):
                row[f"c{index}"] = None if run.objective is None else getattr(run.objective, attribute)
            for attribute in ("agh_executed_start_count", "agh_unexecuted_start_count",
                              "agh_scheduled_random_start_count", "agh_executed_random_start_count",
                              "agh_strict_improvement_count"):
                row[attribute] = getattr(run, attribute)
            raw_rows.append(row)
            if run.validation is not None and run.placement is not None:
                error_by_query = {item.query_id: item for item in run.validation.error_slo_diagnostics}
                delay_by_query = {item.query_id: item for item in run.validation.delay_slo_diagnostics}
                for query_id, delay in run.validation.processing_delays_seconds.items():
                    error, delay_slo = error_by_query[query_id], delay_by_query[query_id]
                    unmet = run.placement.unmet_fractions.get(query_id, 0.0)
                    query_rows.append(dict(case_id=run.case_id, method=run.method,
                                           repeat=run.repeat, seed=run.seed, is_warmup=run.is_warmup,
                                           accepted=run.accepted, query_id=query_id,
                                           unmet_fraction=unmet, unmet_over_1pct=unmet > UNMET_REPORTING_THRESHOLD,
                                           delay_seconds=delay, delay_slo_seconds=delay_slo.delay_limit,
                                           modeled_weighted_error=error.weighted_error, error_slo=error.error_limit))
    summary = summarize_benchmark(runs, cases)
    _write_csv(output_dir / "runs.csv", raw_rows)
    _write_csv(output_dir / "query_metrics.csv", query_rows)
    _write_csv(output_dir / "summary.csv", summary)
    dependencies = {}
    for name in ("pyomo", "highspy", "gurobipy", "matplotlib"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    source_paths = sorted((repository_root / "src").rglob("*.py")) + sorted((repository_root / "scripts").glob("*.py"))
    source_hashes = {str(path.relative_to(repository_root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    manifest = dict(schema_version=1, classification="SYNTHETIC_REFERENCE_BENCHMARK",
                    config=_plain(config),
                    methods=METHODS, python=platform.python_version(), platform=platform.platform(),
                    processor=platform.processor(), logical_cpu_count=os.cpu_count(), dependencies=dependencies,
                    timing_scope="public runner including input/model construction and validation; excludes imports, export and plots",
                    method_order="serial, rotated P_DM/GH/AGH by measured repeat; warmups excluded from summaries",
                    summary_policy="all measured attempts in timings; only accepted runs in quality; median and inclusive quartiles",
                    gap_policy="signed 100*(J-J*)/abs(J*) against accepted OPTIMAL P_DM with matching input SHA, unit/paper policy, repeat and seed; absent if undefined",
                    incumbent_comparison_policy="separate signed percent difference against accepted same-run P_DM incumbent; does not claim optimality",
                    objective_units="USD; coherent units, primary Paper v2, explicit data residency",
                    service_metric="unweighted per-query mean unmet fraction; u>0.01 classification, distinct from constraint feasibility",
                    solver_options=dict(time_limit_seconds=config.solver_time_limit_seconds, relative_mip_gap=0.0, absolute_mip_gap=0.0, threading="backend defaults"),
                    no_incumbent_evidence="public P_DM runner raises without a record; exact exception retained, structured solver timing/bound/status unavailable; messages are not parsed",
                    cases=[dict(case_id=case.case_id, family=case.family, provenance=dict(case.provenance),
                                instance=_plain(case.instance),
                                instance_sha256=_instance_fingerprint(case.instance)) for case in cases],
                    source_sha256=source_hashes)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary
