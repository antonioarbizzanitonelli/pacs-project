"""Plot recorded synthetic benchmark metrics with their availability preserved."""

from __future__ import annotations

import argparse
import csv
from math import isfinite
from pathlib import Path
import sys
from typing import Sequence


METHODS = ("P_DM", "GH", "AGH")
COLORS = {"P_DM": "#315C91", "GH": "#D58829", "AGH": "#24816D"}
COUNTS = ("run_count", "accepted_count", "feasible_count", "optimal_count", "gap_count")
VALUES = (
    "time_median_seconds", "time_q1_seconds", "time_q3_seconds", "objective_median",
    "gap_median_percent", "c1_median", "c2_median", "c3_median", "c4_median", "c5_median",
)


def read_summary(path: Path) -> tuple[dict, ...]:
    """Read finite measurements, keeping empty fields unavailable."""

    rows = []
    seen = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "family", "method", *COUNTS, *VALUES}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError("summary columns missing: " + ", ".join(sorted(missing)))
        for line, record in enumerate(reader, start=2):
            try:
                case_id = record["case_id"]
                method = record["method"]
                if not case_id or method not in METHODS:
                    raise ValueError("case_id and a supported method are required")
                key = (case_id, method)
                if key in seen:
                    raise ValueError("duplicate case_id/method")
                seen.add(key)
                counts = {name: int(record[name]) for name in COUNTS}
                if counts["run_count"] <= 0 or any(
                    not 0 <= counts[name] <= counts["run_count"] for name in COUNTS[1:]
                ):
                    raise ValueError("outcome counts must lie within the measured run count")
                values = {name: float(record[name]) if record[name].strip() else None for name in VALUES}
                if any(value is not None and not isfinite(value) for value in values.values()):
                    raise ValueError("measurements must be finite or empty")
                if any(values[name] is not None and values[name] < 0 for name in VALUES[:3]):
                    raise ValueError("execution times must be nonnegative")
                family = record["family"].strip()
                if not family:
                    raise ValueError("family is required")
                rows.append({"case_id": case_id, "family": family, "method": method, **counts, **values})
            except (TypeError, ValueError, AttributeError) as error:
                raise ValueError(f"summary line {line}: {error}") from error
    if not rows:
        raise ValueError("summary has no measured runs")
    return tuple(rows)


def _style(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E0E5EB", linewidth=0.7)
    axis.set_axisbelow(True)


def _unavailable(axis, position: float, label: str = "undefined") -> None:
    axis.plot(position, 0.025, "x", color="#9B3C3C", transform=axis.get_xaxis_transform())
    axis.annotate(
        label, (position, 0.025), xycoords=axis.get_xaxis_transform(),
        xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7, rotation=90,
    )


def render_figures(input_path: Path, output_dir: Path) -> tuple[Path, ...]:
    """Save runtime, accepted objective/gap, cost components and acceptance figures."""

    rows = read_summary(input_path)
    if output_dir.exists():
        raise ValueError(f"figure output already exists; choose a new directory: {output_dir}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError('install plotting support with python -m pip install -e ".[plots]"') from error

    cases = tuple(dict.fromkeys(row["case_id"] for row in rows))
    families = tuple(dict.fromkeys(row["family"] for row in rows))
    family_titles = {"nominal": "Nominal load", "compute_pressure": "Compute-pressure load"}
    lookup = {(row["case_id"], row["method"]): row for row in rows}
    width = 0.24
    size = (max(10.0, len(cases) * 0.7), 5.8)
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = []

    def finish(figure, name, caption, *, top=0.98):
        figure.text(0.07, 0.015, caption, fontsize=9)
        figure.tight_layout(rect=(0.02, 0.055, 0.98, top))
        for extension in ("png", "svg"):
            path = output_dir / f"{name}.{extension}"
            metadata = {"Date": None} if extension == "svg" else None
            figure.savefig(path, dpi=160, facecolor="white", metadata=metadata)
            outputs.append(path)
        plt.close(figure)

    def case_axis(axis, selected_cases=cases, *, compact=False):
        labels = tuple(case.rsplit("_", 1)[-1] for case in selected_cases) if compact else selected_cases
        axis.set_xticks(range(len(selected_cases)), labels, rotation=35, ha="right", fontsize=8)
        axis.set_xlim(-0.6, len(selected_cases) - 0.4)
        _style(axis)

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"}):
        figure, axis = plt.subplots(figsize=size)
        axis.set_yscale("log")
        axis.set_autoscaley_on(False)
        for index, method in enumerate(METHODS):
            for case_index, case in enumerate(cases):
                position = case_index + (index - 1) * width
                row = lookup.get((case, method))
                if row is None or row["time_median_seconds"] is None:
                    _unavailable(axis, position)
                    continue
                value = row["time_median_seconds"]
                if value <= 0:
                    _unavailable(axis, position, f"{value:g} seconds")
                    continue
                axis.plot(position, value, "o", color=COLORS[method])
                q1, q3 = row["time_q1_seconds"], row["time_q3_seconds"]
                if q1 is not None and q3 is not None:
                    if q1 <= 0:
                        _unavailable(axis, position, "IQR reaches 0")
                    else:
                        axis.errorbar(position, value, yerr=[[value - q1], [q3 - value]], color=COLORS[method], capsize=2)
        positive_times = [
            row[name] for row in rows for name in VALUES[:3]
            if row[name] is not None and row[name] > 0
        ]
        if positive_times:
            axis.set_ylim(min(positive_times) / 2, max(positive_times) * 2)
        figure.suptitle("Execution time by synthetic instance", x=0.07, y=0.97, ha="left")
        axis.set_ylabel("Total execution time [seconds, logarithmic scale]")
        figure.legend(
            handles=[plt.Line2D([0], [0], color=COLORS[method], marker="o", linestyle="none", label=method) for method in METHODS],
            frameon=False, ncols=3, loc="upper left", bbox_to_anchor=(0.07, 0.92),
        )
        case_axis(axis)
        finish(figure, "runtime", "Logarithmic axis; median and inclusive IQR. Nonpositive values have explicit markers; CSV values are unchanged.", top=0.86)

        figure = plt.figure(figsize=(max(11.0, len(families) * 5.7), 9.2))
        grid = figure.add_gridspec(2, len(families), height_ratios=(1, 1.1))
        objective_axes = [figure.add_subplot(grid[0, index]) for index in range(len(families))]
        gap_axis = figure.add_subplot(grid[1, :])
        for family, axis in zip(families, objective_axes, strict=True):
            selected_cases = tuple(dict.fromkeys(row["case_id"] for row in rows if row["family"] == family))
            for index, method in enumerate(METHODS):
                for case_index, case in enumerate(selected_cases):
                    position = case_index + (index - 1) * width
                    row = lookup.get((case, method))
                    if row is None or row["accepted_count"] == 0 or row["objective_median"] is None:
                        _unavailable(axis, position)
                    else:
                        axis.bar(position, row["objective_median"], width * 0.9, color=COLORS[method])
            axis.set_title(f"Accepted objective — {family_titles.get(family, family)}", loc="left", fontsize=11)
            axis.set_ylabel("Median objective [USD]")
            case_axis(axis, selected_cases, compact=True)
            axis.set_xlabel("Queries × models × GPU tiers")
        objective_axes[0].legend(
            handles=[plt.Line2D([0], [0], color=COLORS[method], linewidth=6, label=method) for method in METHODS],
            frameon=False,
        )
        for index, method in enumerate(METHODS):
            for case_index, case in enumerate(cases):
                position = case_index + (index - 1) * width
                row = lookup.get((case, method))
                if row is None or row["gap_count"] == 0 or row["gap_median_percent"] is None:
                    _unavailable(gap_axis, position)
                else:
                    gap_axis.plot(position, row["gap_median_percent"], "o", color=COLORS[method])
        gap_axis.set_title("Gap to accepted OPTIMAL P_DM", loc="left", fontsize=11)
        gap_axis.set_ylabel("Signed relative gap [%]")
        gap_axis.axhline(0, color="#657080", linewidth=0.7, linestyle="--")
        case_axis(gap_axis)
        finish(figure, "objective_and_gap", "Objectives use independent linear scales by family. Gaps require a proven optimum in the same case and round.")

        figure, component_axes = plt.subplots(len(families), 1, figsize=(12.0, 5.0 * len(families)), squeeze=False)
        labels = [f"{row['case_id']}\n{row['method']}" for row in rows]
        component_colors = ("#315C91", "#7C9CBC", "#24816D", "#D5A239", "#B46B77")
        component_labels = ("C1 GPU rental", "C2 model storage", "C3 data storage", "C4 delay", "C5 unmet demand")
        for family, axis in zip(families, component_axes[:, 0], strict=True):
            family_rows = tuple(row for row in rows if row["family"] == family)
            family_labels = [f"{row['case_id'].rsplit('_', 1)[-1]}\n{row['method']}" for row in family_rows]
            for position, row in enumerate(family_rows):
                values = tuple(row[f"c{number}_median"] for number in range(1, 6))
                if row["accepted_count"] == 0 or any(value is None for value in values):
                    _unavailable(axis, position)
                    continue
                positive = negative = 0.0
                for value, color in zip(values, component_colors, strict=True):
                    bottom = positive if value >= 0 else negative
                    axis.bar(position, value, bottom=bottom, color=color)
                    if value >= 0:
                        positive += value
                    else:
                        negative += value
            axis.set_title(f"Objective components — {family_titles.get(family, family)}", loc="left")
            axis.set_ylabel("Median component cost [USD]")
            axis.set_xticks(range(len(family_labels)), family_labels, rotation=35, ha="right", fontsize=8)
            axis.legend(
                handles=[plt.Rectangle((0, 0), 1, 1, color=color, label=label) for color, label in zip(component_colors, component_labels, strict=True)],
                frameon=False, ncols=3, fontsize=8,
            )
            _style(axis)
        finish(figure, "objective_components", "Independent linear scales by family; the large C4 segment is the delay charge. Component medians need not sum to the median total.")

        figure, axis = plt.subplots(figsize=(max(10.0, len(rows) * 0.4), 6.3))
        for position, row in enumerate(rows):
            color = COLORS[row["method"]]
            accepted = 100 * row["accepted_count"] / row["run_count"]
            feasible = 100 * row["feasible_count"] / row["run_count"]
            axis.bar(position - 0.18, accepted, 0.34, color=color)
            axis.bar(position + 0.18, feasible, 0.34, facecolor="white", edgecolor=color, hatch="//")
            axis.text(position, max(accepted, feasible) + 2, f"{row['accepted_count']}/{row['run_count']}", ha="center", fontsize=7)
        axis.set_title("Independent feasibility and method acceptance", loc="left")
        axis.set_ylabel("Measured attempts [%]")
        axis.set_ylim(0, 115)
        axis.set_yticks((0, 25, 50, 75, 100))
        axis.set_xticks(range(len(labels)), labels, rotation=50, ha="right", fontsize=7)
        axis.legend(
            handles=[plt.Rectangle((0, 0), 1, 1, color="#657080", label="Accepted"),
                     plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#657080", hatch="//", label="Independently feasible")],
            frameon=False,
        )
        _style(axis)
        finish(figure, "acceptance", "Every measured attempt remains in the denominator; labels show accepted/attempted.")
    return tuple(outputs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/synthetic_benchmark/summary.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/synthetic_benchmark/figures"))
    args = parser.parse_args(argv)
    try:
        paths = render_figures(args.input, args.output)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Plotting failed: {error}", file=sys.stderr)
        return 1
    print(f"Saved {len(paths)} figures in {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
