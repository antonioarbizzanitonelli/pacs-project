"""Check that comparison figures retain unavailable values and solver evidence."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.plot_backend_comparison import COUNTS, VALUES, read_comparison, render_figures


SUMMARY_FIELDS = ("case_id", "family", "backend", "method", *COUNTS, *VALUES)
PAIRED_FIELDS = ("case_id", "method", "repeat", "seed", "objective_delta_usd")


def write_table(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def recorded_observation(case_id, family, backend, method, objective):
    row = dict.fromkeys(SUMMARY_FIELDS, "")
    row.update(case_id=case_id, family=family, backend=backend, method=method,
               run_count=3, accepted_count=3, feasible_count=3, optimal_count=0,
               time_median_seconds="0.2", time_q1_seconds="0.1", time_q3_seconds="0.3",
               objective_median=objective)
    if method == "P_DM":
        row.update(optimal_count=1, feasible_nonoptimal_count=2,
                   solver_best_bound_median=objective / 2, solver_relative_gap_median="0.5")
    return row


class BackendComparisonPlotTests(unittest.TestCase):
    def setUp(self):
        self.temporary_parent = Path(__file__).resolve().parents[1] / "tmp"
        self.temporary_parent.mkdir(exist_ok=True)

    def test_reader_distinguishes_zero_missing_and_signed_pair_difference(self):
        optimum = recorded_observation("nominal_2x2x2", "nominal", "gurobi", "P_DM", 0)
        optimum.update(optimal_count=3, feasible_nonoptimal_count=0, solver_relative_gap_median="0")
        heuristic = recorded_observation("nominal_2x2x2", "nominal", "gurobi", "GH", 4)
        rejected = recorded_observation("nominal_2x2x2", "nominal", "highs", "P_DM", 2)
        rejected.update(accepted_count=0, feasible_count=3, optimal_count=0,
                        feasible_nonoptimal_count=0, objective_median="", solver_best_bound_median="", solver_relative_gap_median="")
        pairs = [dict(case_id="nominal_2x2x2", method="GH", repeat=0, seed=37, objective_delta_usd="-0.25"),
                 dict(case_id="nominal_2x2x2", method="GH", repeat=1, seed=41, objective_delta_usd="0"),
                 dict(case_id="nominal_2x2x2", method="GH", repeat=2, seed=43, objective_delta_usd=""),
                 dict(case_id="nominal_2x2x2", method="GH", repeat=-1, seed=37, objective_delta_usd="99")]
        with TemporaryDirectory(dir=self.temporary_parent) as directory:
            root = Path(directory)
            write_table(root / "summary.csv", SUMMARY_FIELDS, (optimum, heuristic, rejected))
            write_table(root / "paired_runs.csv", PAIRED_FIELDS, pairs)
            observations, quality = read_comparison(root)
        self.assertEqual(observations[0].objective_median, 0.0)
        self.assertEqual(observations[0].solver_relative_gap_median, 0.0)
        self.assertIsNone(observations[1].solver_relative_gap_median)
        self.assertIsNone(observations[1].feasible_nonoptimal_count)
        self.assertIsNone(observations[2].objective_median)
        self.assertEqual(observations[2].accepted_count, 0)
        self.assertEqual(observations[2].feasible_count, 3)
        self.assertEqual([pair.objective_delta_usd for pair in quality], [-0.25, 0.0, None])

    def test_existing_figure_directory_preserves_evidence(self):
        row = recorded_observation("nominal_2x2x2", "nominal", "highs", "P_DM", 2)
        with TemporaryDirectory(dir=self.temporary_parent) as directory:
            root = Path(directory)
            write_table(root / "summary.csv", SUMMARY_FIELDS, (row,))
            write_table(root / "paired_runs.csv", PAIRED_FIELDS, ())
            output = root / "plots"
            output.mkdir()
            retained = output / "evidence.txt"
            retained.write_text("original", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already exists"):
                render_figures(root, output)
            self.assertEqual(retained.read_text(encoding="utf-8"), "original")
            self.assertEqual(tuple(output.iterdir()), (retained,))

    def test_nonfinite_metric_is_rejected_before_rendering(self):
        row = recorded_observation("nominal_2x2x2", "nominal", "highs", "P_DM", 2)
        row["solver_best_bound_median"] = "nan"
        with TemporaryDirectory(dir=self.temporary_parent) as directory:
            root = Path(directory)
            write_table(root / "summary.csv", SUMMARY_FIELDS, (row,))
            write_table(root / "paired_runs.csv", PAIRED_FIELDS, ())
            with self.assertRaisesRegex(ValueError, "finite or empty"):
                read_comparison(root)

    def test_figures_keep_load_scales_status_and_unavailable_gaps_distinct(self):
        try:
            from matplotlib.figure import Figure
        except ImportError:
            self.skipTest("optional matplotlib is unavailable")
        observations, pairs = [], []
        for family, objective in (("nominal", 2.0), ("compute_pressure", 200000.0)):
            case = f"{family}_2x2x2"
            for backend in ("highs", "gurobi"):
                for method in ("P_DM", "GH", "AGH"):
                    row = recorded_observation(case, family, backend, method, objective)
                    if backend == "gurobi" and method == "P_DM":
                        row.update(optimal_count=3, feasible_nonoptimal_count=0,
                                   time_median_seconds="-0.1", time_q1_seconds="-0.1", time_q3_seconds="-0.1",
                                   solver_best_bound_median="", solver_relative_gap_median="")
                        if family == "compute_pressure":
                            row.update(accepted_count=0, optimal_count=0, feasible_nonoptimal_count=0,
                                       objective_median="")
                    observations.append(row)
            for method in ("GH", "AGH"):
                for repeat, delta in enumerate(("-0.1", "0", "")):
                    pairs.append(dict(case_id=case, method=method, repeat=repeat,
                                      seed=(37, 41, 43)[repeat], objective_delta_usd=delta))
        saved, metadata = {}, {}

        def capture(figure, output, **kwargs):
            saved[output.name] = figure
            metadata[output.name] = kwargs.get("metadata")

        with TemporaryDirectory(dir=self.temporary_parent) as directory:
            root = Path(directory)
            write_table(root / "summary.csv", SUMMARY_FIELDS, observations)
            write_table(root / "paired_runs.csv", PAIRED_FIELDS, pairs)
            with patch.object(Figure, "savefig", capture):
                paths = render_figures(root, root / "plots")
        self.assertEqual(len(paths), 8)
        runtime = saved["pdm_runtime.png"].axes[0]
        self.assertEqual(runtime.get_yscale(), "log")
        self.assertIn("-0.1 seconds", [text.get_text() for text in runtime.texts])
        objective = saved["pdm_objective.png"].axes
        self.assertEqual([axis.get_yscale() for axis in objective], ["linear", "linear"])
        self.assertLess(objective[0].get_ylim()[1], 10)
        self.assertGreater(objective[1].get_ylim()[1], 200000)
        self.assertIn("rejected", [text.get_text() for text in objective[1].texts])
        solver = saved["solver_evidence.png"].axes
        self.assertIn("bound undefined", [text.get_text() for text in solver[0].texts])
        self.assertIn("undefined", [text.get_text() for text in solver[2].texts])
        status_labels = [text.get_text() for text in solver[3].texts]
        self.assertIn("O:1/3\nF:3/3", status_labels)
        self.assertIn("O:3/3\nF:3/3", status_labels)
        self.assertIn("O:0/3\nF:3/3", status_labels)
        heuristic = saved["heuristic_campaigns.png"].axes
        self.assertEqual([axis.get_yscale() for axis in heuristic[:2]], ["log", "log"])
        self.assertIn("undefined", [text.get_text() for text in heuristic[2].texts])
        for name, value in metadata.items():
            self.assertEqual(value, {"Date": None} if name.endswith(".svg") else None)


if __name__ == "__main__":
    unittest.main()
