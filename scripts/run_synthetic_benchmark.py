"""Run P_DM, GH and AGH on shared coherent synthetic instances."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from importlib.util import find_spec
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "synthetic_benchmark.json"
DEFAULT_OUTPUT = ROOT / "results" / "synthetic_benchmark"


def resolve_output(path: Path) -> Path:
    """Require a fresh destination outside the repository's source directories."""

    destination = path.expanduser().resolve()
    protected = (".git", "src", "tests", "scripts", "configs", "docs", "examples", ".github")
    if destination == ROOT or any(destination.is_relative_to(ROOT / name) for name in protected):
        raise ValueError("output must be outside repository source directories")
    if destination.exists():
        raise ValueError(f"output already exists; choose a new directory: {destination}")
    return destination


def read_case_specs(values: dict) -> tuple[dict, ...]:
    """Validate explicit case labels and dimensions before algorithm execution."""

    specifications = values.pop("case_specs", None)
    if not isinstance(specifications, list) or not specifications:
        raise ValueError("case_specs must be a non-empty array")
    seen = set()
    allowed = {
        "case_id", "family", "dimensions", "load_multiplier",
        "delay_slo_multiplier", "budget_multiplier",
    }
    for specification in specifications:
        if not isinstance(specification, dict) or set(specification).difference(allowed):
            raise ValueError("each case must contain only documented case-specification fields")
        for key in ("case_id", "family"):
            label = specification.get(key)
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"each case requires a non-empty {key}")
        case_id = specification["case_id"]
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        dimensions = specification.get("dimensions")
        if not isinstance(dimensions, list) or len(dimensions) != 3 or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 2
            for count in dimensions
        ):
            raise ValueError(f"{case_id}: dimensions must be three integers at least two")
    return tuple(specifications)


def capture_source_snapshot(output: Path, config_bytes: bytes) -> tuple[Path, dict[str, str]]:
    """Freeze executable Python and the exact configuration before scientific imports."""

    payloads = {}
    for directory in ("src", "scripts"):
        for source in sorted((ROOT / directory).rglob("*.py")):
            if not source.resolve().is_relative_to(ROOT):
                raise ValueError(f"snapshot source resolves outside the repository: {source.name}")
            payloads[source.relative_to(ROOT).as_posix()] = source.read_bytes()
    payloads["configs/synthetic_benchmark.json"] = config_bytes
    payloads["pyproject.toml"] = (ROOT / "pyproject.toml").read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    for relative, payload in payloads.items():
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    hashes = {path: hashlib.sha256(payload).hexdigest() for path, payload in payloads.items()}
    return snapshot, hashes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plots", action="store_true", help="also save PNG/SVG figures using matplotlib")
    args = parser.parse_args(argv)

    try:
        output = resolve_output(args.output)
        config_bytes = args.config.read_bytes()
        values = json.loads(config_bytes)
        if not isinstance(values, dict):
            raise ValueError("configuration must be a JSON object")
        case_specs = read_case_specs(values)
        if args.plots and find_spec("matplotlib") is None:
            raise ValueError('plotting requires matplotlib; install python -m pip install -e ".[plots]"')
        if any(name == "llm_resource_allocation" or name.startswith("llm_resource_allocation.") for name in sys.modules):
            raise ValueError("run the benchmark CLI in a fresh Python process to load its captured source")
        snapshot, source_hashes = capture_source_snapshot(output, config_bytes)
    except (OSError, ValueError) as error:
        print(f"Benchmark setup failed: {error}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(snapshot / "src"))
    from llm_resource_allocation.data.scaling_experiment_instances import make_scaling_experiment_case
    from llm_resource_allocation.experiments.synthetic_benchmark import (
        SyntheticBenchmarkConfig, run_synthetic_benchmark, serialize_benchmark_run,
        write_benchmark_results,
    )
    from llm_resource_allocation.math.objective import PaperVersion
    from llm_resource_allocation.optimization.solver import SolverBackend

    try:
        values["backend"] = SolverBackend(values.get("backend", "highs"))
        values["paper_version"] = PaperVersion(values.get("paper_version", "v2"))
        config = SyntheticBenchmarkConfig(**values)
        cases = tuple(
            replace(
                make_scaling_experiment_case(
                    *specification["dimensions"],
                    load_multiplier=specification.get("load_multiplier", 1.0),
                    delay_slo_multiplier=specification.get("delay_slo_multiplier", 1.0),
                    budget_multiplier=specification.get("budget_multiplier", 1.0),
                    data_residency_hours=config.data_residency_hours,
                ),
                case_id=specification["case_id"],
                family=specification["family"],
            )
            for specification in case_specs
        )
    except (TypeError, ValueError) as error:
        print(f"Invalid benchmark configuration: {error}", file=sys.stderr)
        return 1

    partial_manifest = {
        "status": "running",
        "source_snapshot": "source_snapshot",
        "source_snapshot_sha256": source_hashes,
        "input_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "input_config_snapshot": "source_snapshot/configs/synthetic_benchmark.json",
        "completed_attempts": 0,
    }
    partial_manifest_path = output / "partial_manifest.json"
    partial_manifest_path.write_text(json.dumps(partial_manifest, indent=2) + "\n", encoding="utf-8")
    with (output / "partial_runs.jsonl").open("w", encoding="utf-8") as partial:
        def record_completed_attempt(run):
            record = serialize_benchmark_run(run)
            partial.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            partial.flush()
            partial_manifest["completed_attempts"] += 1
            partial_manifest_path.write_text(json.dumps(partial_manifest, indent=2) + "\n", encoding="utf-8")

        try:
            runs = run_synthetic_benchmark(
                config, cases, progress=lambda text: print(text, flush=True),
                on_run=record_completed_attempt,
            )
        except KeyboardInterrupt:
            partial_manifest["status"] = "interrupted"
            partial_manifest_path.write_text(json.dumps(partial_manifest, indent=2) + "\n", encoding="utf-8")
            print(f"Benchmark interrupted; completed attempts remain in {output / 'partial_runs.jsonl'}", file=sys.stderr)
            return 130
    summary = write_benchmark_results(output, config, cases, runs, snapshot)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        source_snapshot="source_snapshot",
        source_snapshot_scope="copied before scientific imports and loaded directly from the snapshot",
        source_snapshot_sha256=source_hashes,
        input_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        input_config_snapshot="source_snapshot/configs/synthetic_benchmark.json",
        requested_case_specs=case_specs,
        plots_requested=args.plots,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    partial_manifest["status"] = "completed"
    partial_manifest_path.write_text(json.dumps(partial_manifest, indent=2) + "\n", encoding="utf-8")

    if args.plots:
        sys.path.insert(0, str(snapshot / "scripts"))
        from plot_synthetic_benchmark import render_figures
        paths = render_figures(output / "summary.csv", output / "figures")
        print(f"Saved {len(paths)} figures")

    measured = tuple(run for run in runs if not run.is_warmup)
    print(f"Saved {len(measured)} measured attempts; {sum(run.accepted for run in measured)} accepted")
    print(f"Summary: {output / 'summary.csv'} ({len(summary)} rows)")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
