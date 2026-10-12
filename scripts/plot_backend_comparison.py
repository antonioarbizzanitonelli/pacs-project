"""Plot recorded P_DM backend evidence and GH/AGH observations across campaigns."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import re
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("highs", "gurobi")
COLORS = {"highs": "#315C91", "gurobi": "#955172"}
LABELS = {"highs": "HiGHS", "gurobi": "Gurobi"}
COUNTS = ("run_count", "accepted_count", "feasible_count", "optimal_count", "feasible_nonoptimal_count")
VALUES = (
    "time_median_seconds", "time_q1_seconds", "time_q3_seconds", "objective_median",
    "solver_best_bound_median", "solver_relative_gap_median", "solver_time_median_seconds",
)


@dataclass(frozen=True, slots=True)
class Observation:
    case_id: str
    family: str
    backend: str
    method: str
    run_count: int
    accepted_count: int
    feasible_count: int
    optimal_count: int
    feasible_nonoptimal_count: int | None
    time_median_seconds: float | None
    time_q1_seconds: float | None
    time_q3_seconds: float | None
    objective_median: float | None
    solver_best_bound_median: float | None
    solver_relative_gap_median: float | None
    solver_time_median_seconds: float | None


@dataclass(frozen=True, slots=True)
class QualityPair:
    case_id: str
    method: str
    seed: int
    objective_delta_usd: float | None


def _optional_number(value: str) -> float | None:
    if not value.strip():
        return None
    number = float(value)
    if not isfinite(number):
        raise ValueError("numeric values must be finite or empty")
    return number


def read_comparison(directory: Path) -> tuple[tuple[Observation, ...], tuple[QualityPair, ...]]:
    """Read comparison tables with unavailable values distinct from numeric zero."""

    observations = []
    seen = set()
    with (directory / "summary.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "family", "backend", "method", *COUNTS, *VALUES}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError("comparison summary columns missing: " + ", ".join(sorted(missing)))
        for line, row in enumerate(reader, start=2):
            try:
                labels = {name: row[name].strip() for name in ("case_id", "family", "backend", "method")}
                if not all(labels.values()) or labels["backend"] not in BACKENDS or labels["method"] not in ("P_DM", "GH", "AGH"):
                    raise ValueError("case, family, backend and method must identify a supported observation")
                key = (labels["case_id"], labels["method"], labels["backend"])
                if key in seen:
                    raise ValueError("duplicate case/method/backend")
                seen.add(key)
                counts = {name: int(row[name]) if row[name].strip() else None for name in COUNTS}
                if counts["run_count"] is None or counts["run_count"] <= 0:
                    raise ValueError("run_count must be positive")
                if any(counts[name] is None or not 0 <= counts[name] <= counts["run_count"] for name in COUNTS[1:4]):
                    raise ValueError("acceptance, feasibility and optimum counts must lie within run_count")
                if counts["feasible_nonoptimal_count"] is not None and not 0 <= counts["feasible_nonoptimal_count"] <= counts["run_count"]:
                    raise ValueError("feasible_nonoptimal_count must lie within run_count")
                if labels["method"] == "P_DM" and counts["feasible_nonoptimal_count"] is None:
                    raise ValueError("P_DM requires a feasible_nonoptimal_count")
                values = {name: _optional_number(row[name]) for name in VALUES}
                observations.append(Observation(**labels, **counts, **values))
            except (TypeError, ValueError, AttributeError) as error:
                raise ValueError(f"comparison summary line {line}: {error}") from error
    if not observations:
        raise ValueError("comparison summary has no measured observations")

    pairs = []
    with (directory / "paired_runs.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "method", "repeat", "seed", "objective_delta_usd"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError("paired observation columns missing: " + ", ".join(sorted(missing)))
        for line, row in enumerate(reader, start=2):
            try:
                if int(row["repeat"]) < 0:
                    continue
                if row["method"] in ("GH", "AGH"):
                    pairs.append(QualityPair(row["case_id"], row["method"], int(row["seed"]), _optional_number(row["objective_delta_usd"])))
            except (TypeError, ValueError, AttributeError) as error:
                raise ValueError(f"paired observation line {line}: {error}") from error
    return tuple(observations), tuple(pairs)


def _style(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#DFE5EB", linewidth=0.7)


def _marker(axis, position: float, label: str = "undefined") -> None:
    axis.plot(position, 0.025, "x", color="#9B3C3C", transform=axis.get_xaxis_transform())
    axis.annotate(label, (position, 0.025), xycoords=axis.get_xaxis_transform(),
                  xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7, rotation=90)


def _case_label(case_id: str, family: str) -> str:
    dimensions = re.search(r"\d+x\d+x\d+$", case_id)
    if dimensions is None:
        return case_id
    prefix = {"nominal": "N", "compute_pressure": "P"}.get(family, family)
    return f"{prefix}\n{dimensions.group().replace('x', '×')}"


def render_figures(comparison_dir: Path, output_dir: Path) -> tuple[Path, ...]:
    """Save four PNG/SVG figures from recorded metrics without executing algorithms."""

    observations, pairs = read_comparison(comparison_dir)
    if output_dir.exists():
        raise ValueError(f"figure output already exists; choose a new directory: {output_dir}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError('install plotting support with python -m pip install -e ".[plots]"') from error

    cases = tuple(dict.fromkeys(row.case_id for row in observations))
    families = tuple(dict.fromkeys(row.family for row in observations))
    case_families = {row.case_id: row.family for row in observations}
    lookup = {(row.case_id, row.method, row.backend): row for row in observations}
    family_titles = {"nominal": "Nominal load", "compute_pressure": "Compute-pressure load"}
    backend_handles = [plt.Line2D([0], [0], marker="o", color=COLORS[backend], linestyle="none", label=LABELS[backend]) for backend in BACKENDS]
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = []

    def case_axis(axis, selected_cases=cases):
        axis.set_xticks(range(len(selected_cases)), [_case_label(case, case_families[case]) for case in selected_cases], fontsize=8)
        axis.set_xlim(-0.6, len(selected_cases) - 0.4)
        _style(axis)

    def finish(figure, name, caption, *, top=0.98):
        figure.text(0.07, 0.015, caption, fontsize=8.5)
        figure.tight_layout(rect=(0.02, 0.06, 0.98, top))
        for extension in ("png", "svg"):
            target = output_dir / f"{name}.{extension}"
            figure.savefig(target, dpi=160, facecolor="white", metadata={"Date": None} if extension == "svg" else None)
            paths.append(target)
        plt.close(figure)

    def runtime_axis(axis, method):
        axis.set_yscale("log")
        axis.set_autoscaley_on(False)
        positive = []
        for backend_index, backend in enumerate(BACKENDS):
            for index, case in enumerate(cases):
                position = index + (backend_index - 0.5) * 0.22
                row = lookup.get((case, method, backend))
                if row is None or row.time_median_seconds is None:
                    _marker(axis, position)
                    continue
                median = row.time_median_seconds
                if median <= 0:
                    _marker(axis, position, f"{median:g} seconds")
                    continue
                axis.plot(position, median, "o", color=COLORS[backend])
                positive.append(median)
                lower, upper = row.time_q1_seconds, row.time_q3_seconds
                if lower is not None and upper is not None:
                    if lower <= 0:
                        _marker(axis, position, "IQR reaches nonpositive")
                    else:
                        positive.extend((lower, upper))
                        axis.errorbar(position, median, yerr=[[median - lower], [upper - median]], color=COLORS[backend], capsize=2)
                if method == "P_DM":
                    solver_time = row.solver_time_median_seconds
                    if solver_time is None:
                        _marker(axis, position, "solver time undefined")
                    elif solver_time <= 0:
                        _marker(axis, position, f"solver {solver_time:g} seconds")
                    else:
                        positive.append(solver_time)
                        axis.plot(position, solver_time, "s", markersize=5, markerfacecolor="none", color=COLORS[backend])
        if positive:
            axis.set_ylim(min(positive) / 2, max(positive) * 2)
        axis.set_ylabel("Total time [seconds, log scale]")
        case_axis(axis)

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"}):
        figure, axis = plt.subplots(figsize=(11.5, 5.8))
        runtime_axis(axis, "P_DM")
        figure.suptitle("P_DM total runtime by exact-solver backend", x=0.07, y=0.97, ha="left")
        timing_handles = [plt.Line2D([0], [0], marker="o", color="#657080", linestyle="none", label="Total time: median and IQR"),
                          plt.Line2D([0], [0], marker="s", color="#657080", markerfacecolor="none", linestyle="none", label="Solver call: median")]
        figure.legend(handles=[*backend_handles, *timing_handles], frameon=False, ncols=4, loc="upper left", bbox_to_anchor=(0.07, 0.92), fontsize=9)
        finish(figure, "pdm_runtime", "Warmups excluded. Solver-call markers are separate observed timings. N: nominal, P: compute pressure.", top=0.86)

        figure, axes = plt.subplots(1, len(families), figsize=(5.7 * len(families), 5.8), squeeze=False)
        for family, axis in zip(families, axes[0], strict=True):
            selected = tuple(case for case in cases if case_families[case] == family)
            for backend_index, backend in enumerate(BACKENDS):
                for index, case in enumerate(selected):
                    position = index + (backend_index - 0.5) * 0.28
                    row = lookup.get((case, "P_DM", backend))
                    if row is None or row.accepted_count == 0 or row.objective_median is None:
                        _marker(axis, position, "rejected" if row is not None and row.accepted_count == 0 else "undefined")
                    else:
                        axis.bar(position, row.objective_median, 0.25, color=COLORS[backend])
            axis.set_title(f"Accepted P_DM objective — {family_titles.get(family, family)}", loc="left", fontsize=11)
            axis.set_ylabel("Median objective [USD]")
            case_axis(axis, selected)
        figure.legend(handles=backend_handles, frameon=False, ncols=2, loc="upper left", bbox_to_anchor=(0.07, 0.985))
        finish(figure, "pdm_objective", "Independent linear scales by load family; accepted incumbents are shown regardless of optimality status.", top=0.92)

        figure = plt.figure(figsize=(max(11.5, len(families) * 5.7), 11.6))
        grid = figure.add_gridspec(3, len(families), height_ratios=(1.05, 0.85, 0.85))
        bound_axes = [figure.add_subplot(grid[0, index]) for index in range(len(families))]
        gap_axis = figure.add_subplot(grid[1, :])
        status_axis = figure.add_subplot(grid[2, :])
        for family, axis in zip(families, bound_axes, strict=True):
            selected = tuple(case for case in cases if case_families[case] == family)
            for backend_index, backend in enumerate(BACKENDS):
                for index, case in enumerate(selected):
                    position = index + (backend_index - 0.5) * 0.22
                    row = lookup.get((case, "P_DM", backend))
                    if row is None or row.accepted_count == 0 or row.objective_median is None:
                        _marker(axis, position, "rejected" if row is not None and row.accepted_count == 0 else "undefined")
                        continue
                    axis.plot(position, row.objective_median, "o", color=COLORS[backend])
                    if row.solver_best_bound_median is None:
                        _marker(axis, position, "bound undefined")
                    else:
                        bound = row.solver_best_bound_median
                        axis.plot(position, bound, "_", markersize=10, color=COLORS[backend])
                        axis.plot((position, position), (bound, row.objective_median), color=COLORS[backend], linewidth=1.2)
            axis.set_title(f"Accepted incumbent and reported bound — {family_titles.get(family, family)}", loc="left", fontsize=10)
            axis.set_ylabel("Median value [USD]")
            case_axis(axis, selected)
        status_handles = [
            plt.Rectangle((0, 0), 1, 1, color="#657080", label="Accepted optimum"),
            plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#657080", hatch="//", label="Accepted feasible, unproven"),
            plt.Rectangle((0, 0), 1, 1, color="#B8BEC6", label="Not accepted"),
        ]
        figure.legend(handles=[*backend_handles, *status_handles], frameon=False, ncols=3,
                      loc="upper left", bbox_to_anchor=(0.07, 0.985), fontsize=9)
        for backend_index, backend in enumerate(BACKENDS):
            for index, case in enumerate(cases):
                position = index + (backend_index - 0.5) * 0.28
                row = lookup.get((case, "P_DM", backend))
                if row is None or row.solver_relative_gap_median is None:
                    _marker(gap_axis, position, "rejected" if row is not None and row.accepted_count == 0 else "undefined")
                else:
                    gap_axis.plot(position, 100 * row.solver_relative_gap_median, "o", color=COLORS[backend])
                if row is None:
                    _marker(status_axis, position)
                    continue
                optimum = 100 * row.optimal_count / row.run_count
                feasible = 100 * row.feasible_nonoptimal_count / row.run_count
                rejected = 100 * (row.run_count - row.accepted_count) / row.run_count
                status_axis.bar(position, optimum, 0.25, color=COLORS[backend])
                status_axis.bar(position, feasible, 0.25, bottom=optimum, facecolor="white", edgecolor=COLORS[backend], hatch="//")
                status_axis.bar(position, rejected, 0.25, bottom=optimum + feasible, color="#B8BEC6")
                status_axis.text(position, 103, f"O:{row.optimal_count}/{row.run_count}\nF:{row.feasible_count}/{row.run_count}", ha="center", fontsize=7)
        gap_axis.set_title("Reported relative MIP gap", loc="left", fontsize=11)
        gap_axis.set_ylabel("Median solver gap [%]")
        gap_axis.axhline(0, color="#657080", linewidth=0.7, linestyle="--")
        case_axis(gap_axis)
        status_axis.set_title("P_DM outcome counts", loc="left", fontsize=11)
        status_axis.set_ylabel("Measured attempts [%]")
        status_axis.set_ylim(0, 126)
        status_axis.set_yticks((0, 25, 50, 75, 100))
        case_axis(status_axis)
        finish(figure, "solver_evidence", "Dots: accepted incumbent; marks: reported bound. Linear scales differ by load. O: accepted optima, F: independently feasible / attempted.", top=0.91)

        figure, axes = plt.subplots(2, 2, figsize=(12.4, 9.4))
        for method, top, bottom in zip(("GH", "AGH"), axes[0], axes[1], strict=True):
            runtime_axis(top, method)
            top.set_title(f"{method} runtime across campaigns", loc="left", fontsize=11)
            for index, case in enumerate(cases):
                selected = tuple(pair for pair in pairs if pair.case_id == case and pair.method == method)
                if not selected:
                    _marker(bottom, index)
                for pair_index, pair in enumerate(selected):
                    position = index + (pair_index - (len(selected) - 1) / 2) * 0.08
                    if pair.objective_delta_usd is None:
                        _marker(bottom, position)
                    else:
                        bottom.plot(position, pair.objective_delta_usd, "o", color=COLORS["gurobi"], markersize=4)
            bottom.set_title(f"{method} paired objective change", loc="left", fontsize=11)
            bottom.set_ylabel("Gurobi campaign − HiGHS campaign [USD]")
            bottom.axhline(0, color="#657080", linewidth=0.7, linestyle="--")
            bottom.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            case_axis(bottom)
        campaign_handles = [plt.Line2D([0], [0], marker="o", color=COLORS[backend], linestyle="none",
                                      label=f"{LABELS[backend]} campaign") for backend in BACKENDS]
        figure.legend(handles=campaign_handles, frameon=False, ncols=2, loc="upper left", bbox_to_anchor=(0.07, 0.985))
        finish(figure, "heuristic_campaigns", "Campaign labels identify the exact-solver setting; GH/AGH do not invoke that solver. Objective differences retain every comparable measured seed.", top=0.93)
    return tuple(paths)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=ROOT / "benchmarks" / "backend_comparison")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "backend_comparison" / "plots")
    args = parser.parse_args(argv)
    try:
        paths = render_figures(args.comparison, args.output)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Backend plotting failed: {error}", file=sys.stderr)
        return 1
    print(f"Saved {len(paths)} backend-comparison figures in {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
