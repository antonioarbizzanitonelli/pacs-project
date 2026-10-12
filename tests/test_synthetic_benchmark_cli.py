"""Check benchmark output isolation, snapshot identity and missing measurements."""

from contextlib import redirect_stderr
import csv
import hashlib
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import run_synthetic_benchmark as runner
from scripts.plot_synthetic_benchmark import COUNTS, VALUES, read_summary, render_figures


class BenchmarkCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary_parent = Path(__file__).resolve().parents[1] / "tmp"
        self.temporary_parent.mkdir(exist_ok=True)

    def test_existing_output_preserves_its_contents_before_reading_config(self):
        with TemporaryDirectory(dir=self.temporary_parent) as directory:
            output = Path(directory) / "observations"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("retained evidence", encoding="utf-8")
            error = StringIO()
            with redirect_stderr(error):
                code = runner.main(["--output", str(output), "--config", str(Path(directory) / "absent.json")])
            self.assertEqual(code, 1)
            self.assertIn("output already exists", error.getvalue())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "retained evidence")
            self.assertEqual(tuple(output.iterdir()), (sentinel,))

    def test_source_snapshot_retains_original_bytes_after_live_edits(self):
        with TemporaryDirectory(dir=self.temporary_parent) as directory:
            root = Path(directory)
            source = root / "src" / "example.py"
            source.parent.mkdir()
            original = b"ANSWER = 37\r\n"
            source.write_bytes(original)
            (root / "scripts").mkdir()
            (root / "pyproject.toml").write_bytes(b"[project]\nname='fixture'\n")
            config = b'{"seeds": [37]}\n'
            with patch.object(runner, "ROOT", root):
                snapshot, hashes = runner.capture_source_snapshot(root / "observations", config)
            source.write_bytes(b"ANSWER = 99\n")
            self.assertEqual((snapshot / "src/example.py").read_bytes(), original)
            self.assertEqual(hashes["src/example.py"], hashlib.sha256(original).hexdigest())
            self.assertEqual((snapshot / "configs/synthetic_benchmark.json").read_bytes(), config)
            self.assertEqual(hashes["configs/synthetic_benchmark.json"], hashlib.sha256(config).hexdigest())

    def test_duplicate_case_identity_is_rejected(self):
        case = {"case_id": "shared", "family": "joint_size", "dimensions": [2, 2, 2]}
        with self.assertRaisesRegex(ValueError, "duplicate case_id"):
            runner.read_case_specs({"case_specs": [case, dict(case)]})

    def test_source_directory_is_not_a_result_destination(self):
        with self.assertRaisesRegex(ValueError, "source directories"):
            runner.resolve_output(runner.ROOT / "src" / "observations")

    def test_plot_input_preserves_zero_and_undefined_quality(self):
        fieldnames = ("case_id", "family", "method", *COUNTS, *VALUES)
        row = dict.fromkeys(fieldnames, "")
        row.update(case_id="case", family="nominal", method="P_DM", run_count=1, accepted_count=1,
                   feasible_count=1, optimal_count=1, gap_count=1,
                   time_median_seconds="0.1", time_q1_seconds="0.1", time_q3_seconds="0.1",
                   objective_median="2.5", gap_median_percent="0")
        failure = dict(row, method="GH", accepted_count=0, feasible_count=0,
                       optimal_count=0, gap_count=0, objective_median="", gap_median_percent="")
        with TemporaryDirectory(dir=self.temporary_parent) as directory:
            path = Path(directory) / "summary.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows((row, failure))
            optimum, rejected = read_summary(path)
        self.assertEqual(optimum["gap_median_percent"], 0.0)
        self.assertIsNone(rejected["objective_median"])
        self.assertIsNone(rejected["gap_median_percent"])
        self.assertEqual(rejected["time_median_seconds"], 0.1)

    def test_figures_separate_load_scales_and_mark_zero_log_runtime(self):
        try:
            from matplotlib.figure import Figure
        except ImportError:
            self.skipTest("optional matplotlib is unavailable")
        fieldnames = ("case_id", "family", "method", *COUNTS, *VALUES)
        records = []
        for family, objective in (("nominal", 2.0), ("compute_pressure", 200000.0)):
            for method in ("P_DM", "GH", "AGH"):
                record = dict.fromkeys(fieldnames, "")
                record.update(case_id=f"{family}_2x2x2", family=family, method=method,
                              run_count=1, accepted_count=1, feasible_count=1,
                              optimal_count=1 if method == "P_DM" else 0, gap_count=0,
                              time_median_seconds=0 if method == "P_DM" else 0.1,
                              time_q1_seconds=0 if method == "P_DM" else 0.1,
                              time_q3_seconds=0 if method == "P_DM" else 0.1,
                              objective_median=objective, c1_median=0, c2_median=0,
                              c3_median=0, c4_median=objective, c5_median=0)
                records.append(record)
        saved = {}
        with TemporaryDirectory(dir=self.temporary_parent) as directory:
            path = Path(directory) / "summary.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)
            with patch.object(Figure, "savefig", lambda figure, output, **kwargs: saved.setdefault(output.name, figure)):
                paths = render_figures(path, Path(directory) / "figures")
            self.assertEqual(read_summary(path)[0]["time_median_seconds"], 0.0)
        self.assertEqual(len(paths), 8)
        runtime = saved["runtime.png"].axes[0]
        self.assertEqual(runtime.get_yscale(), "log")
        self.assertIn("0 seconds", [text.get_text() for text in runtime.texts])
        quality = saved["objective_and_gap.png"].axes
        self.assertEqual(len(quality), 3)
        self.assertEqual(quality[0].get_yscale(), "linear")
        self.assertEqual(quality[1].get_yscale(), "linear")
        self.assertLess(quality[0].get_ylim()[1], 10)
        self.assertGreater(quality[1].get_ylim()[1], 200000)
        self.assertIn("undefined", [text.get_text() for text in quality[2].texts])
        self.assertEqual(len(saved["objective_components.png"].axes), 2)


if __name__ == "__main__":
    unittest.main()
