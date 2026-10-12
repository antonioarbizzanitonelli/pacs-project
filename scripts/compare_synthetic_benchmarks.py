"""Compare two separately measured synthetic campaigns with identical scientific inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import isfinite
from pathlib import Path
from statistics import median, quantiles
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("P_DM", "GH", "AGH")
FLAGS = ("completed", "accepted", "independently_feasible", "optimal")
NUMBERS = ("time_seconds", "solver_time_seconds", "solver_best_bound", "solver_relative_gap",
           "objective_total", "c1", "c2", "c3", "c4", "c5")
OBSERVATIONS = ("status", "solver_status", "solver_termination", "error_type", "error_message")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def instance_sha256(instance: dict) -> str:
    return sha256(json.dumps(instance, sort_keys=True, allow_nan=False).encode("utf-8"))


def _relative_file(directory: Path, name: str) -> Path:
    target = (directory / name).resolve()
    if not target.is_relative_to(directory.resolve()):
        raise ValueError("campaign paths must remain within their directory")
    return target


def _verify_scientific_snapshot(directory: Path, manifest: dict, hashes: dict) -> None:
    archive = manifest.get("publication", {}).get("source_snapshot_archive")
    prefix = manifest.get("source_snapshot", "source_snapshot")
    if archive:
        with zipfile.ZipFile(_relative_file(directory, archive)) as snapshot:
            for name, expected in hashes.items():
                if sha256(snapshot.read(f"{prefix}/{name}")) != expected:
                    raise ValueError(f"scientific snapshot hash mismatch: {name}")
    else:
        for name, expected in hashes.items():
            if sha256(_relative_file(directory, f"{prefix}/{name}").read_bytes()) != expected:
                raise ValueError(f"scientific snapshot hash mismatch: {name}")


def read_campaign(directory: Path) -> dict:
    """Check recorded inputs, snapshot bytes, and the complete measured schedule."""

    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    config = manifest["config"]
    if config.get("paper_version") != "v2" or tuple(manifest["methods"]) != METHODS:
        raise ValueError("comparison requires the coherent V2 three-method protocol")
    if config.get("backend") not in ("highs", "gurobi"):
        raise ValueError("unsupported campaign backend")
    hashes = {name: digest for name, digest in manifest["source_sha256"].items() if name.startswith("src/")}
    if not hashes:
        raise ValueError("scientific source hashes are missing")
    _verify_scientific_snapshot(directory, manifest, hashes)
    cases = {}
    for case in manifest["cases"]:
        identity, instance = case["case_id"], case["instance"]
        if identity in cases:
            raise ValueError("duplicate case identity in manifest")
        if (instance.get("unit_convention") != "coherent"
                or instance.get("data_residency_hours") != config.get("data_residency_hours")):
            raise ValueError("case unit/residency policy disagrees with the coherent configuration")
        if instance_sha256(instance) != case["instance_sha256"]:
            raise ValueError(f"case input hash mismatch: {identity}")
        cases[identity] = case
    seeds = config["seeds"]
    if not cases or not seeds:
        raise ValueError("comparison requires nonempty cases and measured seeds")
    expected = {(case, method, repeat, seed) for case in cases for method in METHODS
                for repeat, seed in enumerate(seeds)}
    rows = {}
    runs_bytes = (directory / "runs.csv").read_bytes()
    declared_runs_hash = manifest.get("publication", {}).get("files", {}).get("runs.csv", {}).get("sha256")
    if declared_runs_hash is not None and sha256(runs_bytes) != declared_runs_hash:
        raise ValueError("published runs.csv hash mismatch")
    with (directory / "runs.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "family", "method", "repeat", "seed", "is_warmup", "instance_sha256", "objective_agrees",
                    *FLAGS, *NUMBERS, *OBSERVATIONS}
        if required.difference(reader.fieldnames or ()):
            raise ValueError("runs.csv lacks required scientific observation columns")
        for raw in reader:
            row = dict(raw)
            row["repeat"], row["seed"] = int(raw["repeat"]), int(raw["seed"])
            for name in (*FLAGS, "is_warmup"):
                if raw[name] not in ("True", "False"):
                    raise ValueError(f"invalid boolean observation: {name}")
                row[name] = raw[name] == "True"
            if raw["objective_agrees"] not in ("True", "False", ""):
                raise ValueError("invalid optional objective-agreement observation")
            row["objective_agrees"] = None if not raw["objective_agrees"] else raw["objective_agrees"] == "True"
            if row["is_warmup"] != (row["repeat"] < 0):
                raise ValueError("warmup flag disagrees with repeat index")
            if row["is_warmup"]:
                continue
            for name in NUMBERS:
                row[name] = float(raw[name]) if raw[name] else None
                if row[name] is not None and not isfinite(row[name]):
                    raise ValueError(f"nonfinite observation: {name}")
            key = (row["case_id"], row["method"], row["repeat"], row["seed"])
            if key not in expected or key in rows:
                raise ValueError("unexpected or duplicate case/method/repeat/seed")
            case = cases[row["case_id"]]
            if row["instance_sha256"] != case["instance_sha256"] or row["family"] != case["family"]:
                raise ValueError("run input identity disagrees with manifest")
            if row["time_seconds"] is None or row["time_seconds"] < 0:
                raise ValueError("every measured attempt requires nonnegative runtime")
            if row["accepted"] and (not row["completed"] or not row["independently_feasible"] or row["objective_total"] is None):
                raise ValueError("accepted attempt lacks completion, feasibility, or objective")
            if row["method"] == "P_DM" and row["accepted"] and row["objective_agrees"] is not True:
                raise ValueError("accepted P_DM attempt requires independent objective agreement")
            if row["optimal"] != (row["method"] == "P_DM" and row["accepted"] and row["solver_status"] == "optimal"):
                raise ValueError("optimal flag disagrees with independently accepted solver status")
            row["rejection_reason"] = rejection_reason(row)
            rows[key] = row
    if set(rows) != expected:
        raise ValueError("measured schedule is incomplete or repeats/seeds do not match configuration")
    return dict(manifest=manifest, cases=cases, rows=rows, scientific_hashes=hashes,
                manifest_sha256=sha256(manifest_bytes), runs_sha256=sha256(runs_bytes))


def percent_difference(candidate: float | None, reference: float | None) -> float | None:
    """Retain signed differences; a zero or unavailable reference is undefined."""

    if candidate is None or reference is None or reference == 0:
        return None
    return 100.0 * (candidate - reference) / abs(reference)


def rejection_reason(row: dict) -> str:
    """Distinguish feasibility from objective agreement without repairing evidence."""

    if row["accepted"]:
        return ""
    if row["status"] == "no_solver_incumbent":
        return "no_solver_incumbent"
    if not row["independently_feasible"]:
        return "constraint_violation"
    if row["method"] == "P_DM" and row["objective_agrees"] is False:
        return "objective_disagreement"
    return "incomplete_or_unaccepted"


def _within_backend_metrics(row: dict, campaign: dict) -> tuple[float | None, float | None]:
    baseline = campaign["rows"][(row["case_id"], "P_DM", row["repeat"], row["seed"])]
    if not row["accepted"] or not baseline["accepted"]:
        return None, None
    comparison = percent_difference(row["objective_total"], baseline["objective_total"])
    return (comparison if baseline["optimal"] else None), comparison


def _median(values: list) -> float | None:
    selected = [value for value in values if value is not None]
    return median(selected) if selected else None


def compare_campaigns(reference: dict, candidate: dict) -> tuple[list[dict], list[dict]]:
    """Compare like inputs; runtimes remain observations from separate campaigns."""

    left, right = reference["manifest"], candidate["manifest"]
    comparable_keys = ("methods", "solver_options", "timing_scope", "method_order", "summary_policy")
    if any(left[key] != right[key] for key in comparable_keys):
        raise ValueError("campaign measurement/solver protocols differ")
    if ({key: value for key, value in left["config"].items() if key != "backend"}
            != {key: value for key, value in right["config"].items() if key != "backend"}):
        raise ValueError("campaign configurations differ beyond backend")
    if left["config"]["backend"] == right["config"]["backend"]:
        raise ValueError("comparison requires two different backends")
    if reference["scientific_hashes"] != candidate["scientific_hashes"]:
        raise ValueError("scientific source hashes differ")
    if reference["cases"] != candidate["cases"]:
        raise ValueError("case inputs, policies, provenance, or identities differ")
    if reference["rows"].keys() != candidate["rows"].keys():
        raise ValueError("campaign measured schedules differ")
    paired, summaries = [], []
    for key, left_row in reference["rows"].items():
        right_row = candidate["rows"][key]
        row = dict(zip(("case_id", "method", "repeat", "seed"), key))
        row.update(family=left_row["family"], instance_sha256=left_row["instance_sha256"])
        for label, campaign, observed in (("reference", reference, left_row), ("candidate", candidate, right_row)):
            row[f"{label}_backend"] = campaign["manifest"]["config"]["backend"]
            row.update({f"{label}_{name}": observed[name] for name in
                        (*FLAGS, *NUMBERS, *OBSERVATIONS, "objective_agrees", "rejection_reason")})
            gap, comparison = _within_backend_metrics(observed, campaign)
            row[f"{label}_gap_percent"] = gap
            row[f"{label}_pdm_incumbent_comparison_percent"] = comparison
        accepted_pair = left_row["accepted"] and right_row["accepted"]
        row["objective_delta_usd"] = right_row["objective_total"] - left_row["objective_total"] if accepted_pair else None
        row["objective_difference_percent"] = percent_difference(right_row["objective_total"], left_row["objective_total"]) if accepted_pair else None
        paired.append(row)
    for label, campaign in (("reference", reference), ("candidate", candidate)):
        for identity, case in campaign["cases"].items():
            for method in METHODS:
                group = [row for row in campaign["rows"].values() if row["case_id"] == identity and row["method"] == method]
                accepted = [row for row in group if row["accepted"]]
                times = [row["time_seconds"] for row in group]
                solver_times = [row["solver_time_seconds"] for row in group if row["solver_time_seconds"] is not None]
                q1, _, q3 = quantiles(times, n=4, method="inclusive") if len(times) > 1 else (times[0], times[0], times[0])
                metrics = [_within_backend_metrics(row, campaign) for row in accepted]
                summary = dict(case_id=identity, family=case["family"], campaign=label,
                               backend=campaign["manifest"]["config"]["backend"], method=method,
                               run_count=len(group), completed_count=sum(row["completed"] for row in group),
                               accepted_count=len(accepted), feasible_count=sum(row["independently_feasible"] for row in group),
                               rejected_count=len(group) - len(accepted),
                               objective_disagreement_count=sum(row["rejection_reason"] == "objective_disagreement" for row in group),
                               optimal_count=sum(row["optimal"] for row in group),
                               feasible_nonoptimal_count=sum(method == "P_DM" and row["accepted"] and not row["optimal"] for row in group),
                               time_median_seconds=median(times), time_q1_seconds=q1, time_q3_seconds=q3,
                               solver_time_median_seconds=_median(solver_times), solver_time_count=len(solver_times),
                               objective_median=_median([row["objective_total"] for row in accepted]),
                               solver_best_bound_median=_median([row["solver_best_bound"] for row in accepted]),
                               solver_relative_gap_median=_median([row["solver_relative_gap"] for row in accepted]),
                               gap_median_percent=_median([metric[0] for metric in metrics]),
                               gap_count=sum(metric[0] is not None for metric in metrics),
                               pdm_incumbent_comparison_median_percent=_median([metric[1] for metric in metrics]),
                               pdm_incumbent_comparison_count=sum(metric[1] is not None for metric in metrics))
                summary.update({f"c{index}_median": _median([row[f"c{index}"] for row in accepted]) for index in range(1, 6)})
                summaries.append(summary)
    return paired, summaries


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_comparison(output: Path, reference: dict, candidate: dict) -> dict:
    """Create a fresh portable artifact with input and output byte fingerprints."""

    paired, summary = compare_campaigns(reference, candidate)
    destination = output.expanduser().resolve()
    if destination == ROOT or any(destination.is_relative_to(ROOT / name) for name in
                                  (".git", "src", "tests", "scripts", "configs", "docs", "examples", ".github")):
        raise ValueError("comparison output must be outside source directories")
    destination.mkdir(parents=True, exist_ok=False)
    _write_csv(destination / "paired_runs.csv", paired)
    _write_csv(destination / "summary.csv", summary)
    manifest = dict(schema_version=1, comparison="separate synthetic backend campaigns",
                    runtime_interpretation="separately measured observations; no simultaneous paired timing experiment or causal speedup claim",
                    quality_policy="all measured attempts in counts/runtime; accepted runs only in objective summaries; paired cost differences require both accepted",
                    gap_policy="heuristic gap uses accepted OPTIMAL P_DM in the same backend/repeat/seed; comparison to feasible P_DM incumbent is separate",
                    scientific_source_sha256=reference["scientific_hashes"],
                    case_input_sha256={key: case["instance_sha256"] for key, case in reference["cases"].items()},
                    comparison_script_sha256=sha256(Path(__file__).read_bytes()), inputs={}, files={})
    for label, campaign in (("reference", reference), ("candidate", candidate)):
        observed = campaign["manifest"]
        manifest["inputs"][label] = dict(backend=observed["config"]["backend"], config=observed["config"],
                                       manifest_sha256=campaign["manifest_sha256"], runs_sha256=campaign["runs_sha256"],
                                       input_config_sha256=observed["input_config_sha256"],
                                       environment={key: observed.get(key) for key in ("python", "platform", "processor", "logical_cpu_count", "dependencies")})
    for name in ("paired_runs.csv", "summary.csv"):
        payload = (destination / name).read_bytes()
        manifest["files"][name] = dict(bytes=len(payload), sha256=sha256(payload))
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "benchmarks/reference")
    parser.add_argument("--candidate", type=Path, default=ROOT / "benchmarks/gurobi")
    parser.add_argument("--output", type=Path, default=ROOT / "results/backend_comparison")
    arguments = parser.parse_args()
    try:
        reference, candidate = read_campaign(arguments.reference), read_campaign(arguments.candidate)
        write_comparison(arguments.output, reference, candidate)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"Backend comparison failed: {error}", file=sys.stderr)
        return 1
    print(f"Saved {len(reference['rows'])} matched measured attempt pairs to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
