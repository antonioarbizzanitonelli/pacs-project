"""Hand-derived comparison arithmetic, schedule identity and incumbent semantics."""

import csv
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts import compare_synthetic_benchmarks as compare


def write_campaign(directory, backend, *, rejected_second_gh=False):
    directory.mkdir()
    snapshot = directory / "source_snapshot" / "src"
    snapshot.mkdir(parents=True)
    scientific_source = b"VALUE = 1\n"
    (snapshot / "fixture.py").write_bytes(scientific_source)
    instance = dict(id="fixture", unit_convention="coherent", data_residency_hours=1.0,
                    queries=[dict(id="q")], models=[dict(id="m")], gpu_tiers=[dict(id="k")])
    fingerprint = compare.instance_sha256(instance)
    config = dict(backend=backend, seeds=[37, 41], warmup_repeats=0,
                  paper_version="v2", data_residency_hours=1.0, solver_time_limit_seconds=20.0)
    manifest = dict(config=config, methods=list(compare.METHODS), solver_options=dict(time_limit_seconds=20.0),
                    timing_scope="public runner", method_order="rotated", summary_policy="accepted quality",
                    source_snapshot="source_snapshot", source_sha256={"src/fixture.py": compare.sha256(scientific_source)},
                    input_config_sha256="config-fingerprint",
                    cases=[dict(case_id="fixture", family="hand", instance=instance, instance_sha256=fingerprint)])
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = []
    for repeat, seed in enumerate(config["seeds"]):
        for method in compare.METHODS:
            optimum = method == "P_DM" and (backend == "gurobi" or repeat == 1)
            objective = ((10, 11) if backend == "gurobi" else (12, 14))[repeat] if method == "P_DM" else 20
            accepted = not (rejected_second_gh and repeat == 1 and method == "GH")
            if not accepted:
                objective = 100  # Retained rejected evidence, excluded from quality means.
            row = dict(case_id="fixture", family="hand", method=method, repeat=repeat, seed=seed,
                       is_warmup=False, instance_sha256=fingerprint, completed=True, accepted=accepted,
                       independently_feasible=accepted, optimal=optimum,
                       objective_agrees=True if method == "P_DM" else "",
                       time_seconds=(1, 3)[repeat] + (1 if backend == "gurobi" else 0),
                       solver_time_seconds=.5 if method == "P_DM" else "",
                       solver_best_bound=(objective if optimum else 8) if method == "P_DM" else "",
                       solver_relative_gap=(0 if optimum else 1 / 3) if method == "P_DM" else "",
                       objective_total=objective, status=("optimal" if optimum else "feasible") if method == "P_DM" else "completed",
                       solver_status=("optimal" if optimum else "feasible") if method == "P_DM" else "",
                       solver_termination="optimal" if optimum else "maxTimeLimit" if method == "P_DM" else "",
                       error_type="", error_message="", c1=objective, c2=0, c3=0, c4=0, c5=0)
            rows.append(row)
    with (directory / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return compare.read_campaign(directory)


class BackendComparisonTests(unittest.TestCase):
    def setUp(self):
        self.parent = Path(__file__).resolve().parents[1] / "tmp"
        self.parent.mkdir(exist_ok=True)
        self.temporary = TemporaryDirectory(dir=self.parent)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.reference = write_campaign(self.directory / "highs", "highs")
        self.candidate = write_campaign(self.directory / "gurobi", "gurobi")

    def test_hand_differences_and_same_backend_optimality_are_distinct(self):
        paired, summary = compare.compare_campaigns(self.reference, self.candidate)
        pdm = next(row for row in paired if row["method"] == "P_DM" and row["repeat"] == 0)
        self.assertEqual(pdm["objective_delta_usd"], -2)
        self.assertAlmostEqual(pdm["objective_difference_percent"], -100 / 6)
        self.assertFalse(pdm["reference_optimal"])
        self.assertTrue(pdm["candidate_optimal"])
        self.assertIsNone(pdm["reference_gap_percent"])
        self.assertEqual(pdm["reference_pdm_incumbent_comparison_percent"], 0)
        gh = next(row for row in paired if row["method"] == "GH" and row["repeat"] == 0)
        self.assertIsNone(gh["reference_gap_percent"])
        self.assertAlmostEqual(gh["reference_pdm_incumbent_comparison_percent"], 100 * 8 / 12)
        self.assertEqual(gh["candidate_gap_percent"], 100)
        self.assertEqual(gh["objective_delta_usd"], 0)
        highs = next(row for row in summary if row["backend"] == "highs" and row["method"] == "P_DM")
        gurobi = next(row for row in summary if row["backend"] == "gurobi" and row["method"] == "P_DM")
        self.assertEqual((highs["run_count"], highs["optimal_count"], highs["feasible_nonoptimal_count"]), (2, 1, 1))
        self.assertEqual((highs["time_median_seconds"], gurobi["time_median_seconds"]), (2, 3))
        self.assertEqual((highs["solver_time_median_seconds"], highs["solver_time_count"]), (.5, 2))
        heuristic = next(row for row in summary if row["backend"] == "highs" and row["method"] == "GH")
        self.assertIsNone(heuristic["solver_time_median_seconds"])
        self.assertEqual(heuristic["solver_time_count"], 0)
        self.assertEqual((highs["objective_median"], gurobi["objective_median"]), (13, 10.5))

    def test_rejected_attempt_retains_time_and_is_excluded_from_quality(self):
        candidate = write_campaign(self.directory / "rejected", "gurobi", rejected_second_gh=True)
        paired, summary = compare.compare_campaigns(self.reference, candidate)
        rejected = next(row for row in paired if row["method"] == "GH" and row["repeat"] == 1)
        self.assertIsNone(rejected["objective_difference_percent"])
        self.assertEqual(rejected["candidate_objective_total"], 100)
        row = next(row for row in summary if row["campaign"] == "candidate" and row["method"] == "GH")
        self.assertEqual((row["run_count"], row["accepted_count"], row["time_median_seconds"]), (2, 1, 3))
        self.assertEqual(row["objective_median"], 20)

    def test_mismatched_config_inputs_and_source_are_rejected(self):
        for area in ("configuration", "input", "source", "schedule"):
            changed = deepcopy(self.candidate)
            if area == "configuration":
                changed["manifest"]["config"]["solver_time_limit_seconds"] = 10.0
            elif area == "input":
                changed["cases"]["fixture"]["instance"]["data_residency_hours"] = 2.0
            elif area == "source":
                changed["scientific_hashes"]["src/fixture.py"] = "different"
            else:
                changed["rows"].pop(("fixture", "GH", 0, 37))
            with self.subTest(area=area), self.assertRaises(ValueError):
                compare.compare_campaigns(self.reference, changed)

    def test_reader_rejects_missing_repeat_and_bad_input_fingerprint(self):
        directory = self.directory / "highs"
        path = directory / "runs.csv"
        original = path.read_text(encoding="utf-8")
        path.write_text("\n".join(original.splitlines()[:-1]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "schedule"):
            compare.read_campaign(directory)
        path.write_text(original, encoding="utf-8")
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["cases"][0]["instance_sha256"] = "wrong"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "input hash"):
            compare.read_campaign(directory)

    def test_reader_checks_snapshot_bytes(self):
        (self.directory / "highs/source_snapshot/src/fixture.py").write_text("VALUE = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "snapshot hash"):
            compare.read_campaign(self.directory / "highs")

    def test_reader_checks_declared_published_csv_hash(self):
        directory = self.directory / "highs"
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["publication"] = {"files": {"runs.csv": {"sha256": "wrong"}}}
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "runs.csv hash"):
            compare.read_campaign(directory)

    def test_objective_disagreement_preserves_feasibility_and_rejects_quality(self):
        directory = self.directory / "gurobi"
        path = directory / "runs.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        rows[0].update(accepted="False", optimal="False", objective_agrees="False")
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        candidate = compare.read_campaign(directory)
        paired, summaries = compare.compare_campaigns(self.reference, candidate)
        observed = next(row for row in paired if row["method"] == "P_DM" and row["repeat"] == 0)
        self.assertTrue(observed["candidate_independently_feasible"])
        self.assertFalse(observed["candidate_accepted"])
        self.assertEqual(observed["candidate_rejection_reason"], "objective_disagreement")
        self.assertIsNone(observed["objective_difference_percent"])
        row = next(row for row in summaries if row["campaign"] == "candidate" and row["method"] == "P_DM")
        self.assertEqual((row["run_count"], row["accepted_count"], row["objective_disagreement_count"]), (2, 1, 1))
        self.assertEqual(row["time_median_seconds"], 3)
        self.assertEqual(row["objective_median"], 11)
        rows[0]["accepted"] = "True"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "objective agreement"):
            compare.read_campaign(directory)

    def test_zero_reference_signed_values_and_portable_export(self):
        self.assertIsNone(compare.percent_difference(10, 0))
        self.assertEqual(compare.percent_difference(9, 10), -10)
        output = self.directory / "comparison"
        manifest = compare.write_comparison(output, self.reference, self.candidate)
        self.assertEqual(manifest["inputs"]["reference"]["manifest_sha256"], self.reference["manifest_sha256"])
        for name, evidence in manifest["files"].items():
            payload = (output / name).read_bytes()
            self.assertNotIn(b"\r\n", payload)
            self.assertEqual(compare.sha256(payload), evidence["sha256"])
        self.assertNotIn(str(self.directory), json.dumps(manifest))
        with self.assertRaises(FileExistsError):
            compare.write_comparison(output, self.reference, self.candidate)


if __name__ == "__main__":
    unittest.main()
