"""Hand-derived checks of shared-input measurements and scientific evidence."""

import csv
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import llm_resource_allocation.experiments.synthetic_benchmark as benchmark
from llm_resource_allocation.data.synthetic_benchmark_instances import make_synthetic_benchmark_cases
from llm_resource_allocation.math.objective import ObjectiveBreakdown, PaperVersion
from llm_resource_allocation.optimization.solver import SolverBackend, SolverStatus
from llm_resource_allocation.validation.placement import validate_placement


class SyntheticBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = make_synthetic_benchmark_cases(())[0]
        cls.config = benchmark.SyntheticBenchmarkConfig(seeds=(37,), query_counts=(), warmup_repeats=0)
        cls.runs = benchmark.run_synthetic_benchmark(cls.config, (cls.case,))

    def test_public_defaults_are_explicit_and_coherent(self):
        config = benchmark.SyntheticBenchmarkConfig()
        self.assertEqual(config.seeds, (37, 41, 43))
        self.assertEqual(config.warmup_repeats, 1)
        self.assertIs(config.backend, SolverBackend.HIGHS)
        self.assertEqual(config.solver_time_limit_seconds, 20.0)
        self.assertEqual(config.data_residency_hours, 1.0)
        for changes in ({"seeds": ()}, {"seeds": (True,)}, {"warmup_repeats": -1},
                        {"query_counts": (2, 2)}, {"gh_budget_fraction": 1.1},
                        {"solver_time_limit_seconds": 0}, {"paper_version": PaperVersion.V1},
                        {"data_residency_hours": -1.0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                benchmark.SyntheticBenchmarkConfig(**changes)

    def test_minimal_costs_are_hand_derived_and_independently_validated(self):
        pdm, gh, agh = self.runs
        self.assertEqual(tuple(run.method for run in self.runs), benchmark.METHODS)
        # P_DM: TP2,x=.8 -> 2+.001+.000000024+12.8+1.
        self.assertAlmostEqual(pdm.objective.total, 15.801000024, places=9)
        self.assertAlmostEqual(pdm.placement.routing_fractions[("q_text", "m_small", "gpu_basic_fp16")], .8)
        # GH/AGH retain TP1 and serve x=1 -> 1+.001+.00000003+31+0.
        for run in (gh, agh):
            self.assertAlmostEqual(run.objective.total, 32.00100003, places=9)
            self.assertEqual(run.objective.delay_penalty, 31.0)
        for run in self.runs:
            self.assertTrue(run.completed)
            self.assertTrue(run.accepted)
            self.assertEqual(run.validation, validate_placement(self.case.instance, run.placement, PaperVersion.V2))
            self.assertEqual(run.instance_sha256, pdm.instance_sha256)
        self.assertEqual(agh.agh_executed_start_count, 6)
        self.assertEqual(agh.agh_executed_random_start_count, 0)
        self.assertEqual(agh.agh_strict_improvement_count, 0)

    def test_gap_requires_proven_optimum_but_incumbent_comparison_is_distinct(self):
        baseline, gh, _ = self.runs
        feasible = replace(baseline, solver=replace(baseline.solver, status=SolverStatus.FEASIBLE))
        self.assertIsNone(benchmark.objective_gap_percent(gh, feasible))
        self.assertIsNotNone(benchmark.pdm_incumbent_comparison_percent(gh, feasible))
        for changed in (None, replace(baseline, accepted=False),
                        replace(baseline, repeat=2), replace(baseline, seed=43),
                        replace(baseline, instance_sha256="different-input"),
                        replace(baseline, validation=replace(baseline.validation, data_residency_hours=2.0))):
            with self.subTest(baseline=changed):
                self.assertIsNone(benchmark.objective_gap_percent(gh, changed))
        zero = replace(baseline.validation, objective_breakdown=ObjectiveBreakdown(0, 0, 0, 0, 0))
        self.assertIsNone(benchmark.objective_gap_percent(gh, replace(baseline, validation=zero)))

    def test_signed_gap_is_not_clipped(self):
        baseline, gh, _ = self.runs
        baseline = replace(baseline, validation=replace(baseline.validation, objective_breakdown=ObjectiveBreakdown(10, 0, 0, 0, 0)))
        gh = replace(gh, validation=replace(gh.validation, objective_breakdown=ObjectiveBreakdown(9, 0, 0, 0, 0)))
        self.assertEqual(benchmark.objective_gap_percent(gh, baseline), -10.0)

    def test_summary_excludes_warmups_and_rejected_quality_but_keeps_attempt_times(self):
        baseline = replace(self.runs[0], elapsed_seconds=1.0)
        rejected = replace(baseline, repeat=1, accepted=False, elapsed_seconds=3.0)
        warmup = replace(baseline, repeat=-1, elapsed_seconds=100.0)
        row, = benchmark.summarize_benchmark((warmup, baseline, rejected), (self.case,))
        self.assertEqual((row["run_count"], row["accepted_count"]), (2, 1))
        self.assertEqual((row["time_median_seconds"], row["time_q1_seconds"], row["time_q3_seconds"]), (2.0, 1.5, 2.5))
        self.assertEqual(row["objective_median"], baseline.objective.total)
        self.assertEqual(row["gap_count"], 1)

    def test_shared_identity_seeds_rotating_order_and_callback(self):
        config = replace(self.config, seeds=(37, 41, 43), warmup_repeats=1)
        seen, delivered, progress = [], [], []
        def run_one(case, selected_config, method, repeat, seed):
            seen.append((case, method, repeat, seed))
            return replace(self.runs[0], method=method, repeat=repeat, seed=seed)
        with patch.object(benchmark, "_run_one", side_effect=run_one):
            runs = benchmark.run_synthetic_benchmark(config, (self.case,), progress.append, on_run=delivered.append)
        self.assertEqual(len(runs), 12)
        self.assertTrue(all(item[0] is self.case for item in seen))
        self.assertEqual(tuple(item[1] for item in seen[6:9]), ("GH", "AGH", "P_DM"))
        self.assertEqual(tuple(item[1] for item in seen[9:]), ("AGH", "P_DM", "GH"))
        self.assertEqual(tuple(run.seed for run in runs), (37,) * 6 + (41,) * 3 + (43,) * 3)
        self.assertEqual(tuple(delivered), runs)
        self.assertEqual(len(progress), 25)  # one case message plus before/after each attempt

    def test_unexpected_programming_errors_propagate(self):
        with patch.object(benchmark, "execute_and_record_pdm_run", side_effect=RuntimeError("broken code")):
            with self.assertRaisesRegex(RuntimeError, "broken code"):
                benchmark.run_synthetic_benchmark(self.config, (self.case,))

    def test_known_no_incumbent_is_recorded_without_invented_solver_fields(self):
        case = replace(self.case, instance=replace(self.case.instance, cost_budget_usd=0.0))
        run = benchmark._run_one(case, self.config, "P_DM", 0, 37)
        self.assertEqual((run.status, run.error_type), ("no_solver_incumbent", "NoSolverIncumbentError"))
        self.assertFalse(run.completed)
        self.assertFalse(run.accepted)
        self.assertIsNone(run.solver)
        self.assertIsNone(run.objective)

    def test_elapsed_time_wraps_public_runner_without_export(self):
        with patch.object(benchmark, "perf_counter", side_effect=(10.0, 10.25)):
            run = benchmark._run_one(self.case, self.config, "GH", 0, 37)
        self.assertEqual(run.elapsed_seconds, .25)

    def test_compact_export_preserves_decisions_validation_and_all_costs(self):
        repository = Path(__file__).resolve().parents[1]
        with TemporaryDirectory(prefix=".benchmark-test-", dir=repository) as temporary:
            output = Path(temporary)
            summary = benchmark.write_benchmark_results(output, self.config, (self.case,), self.runs, repository)
            self.assertEqual(len(summary), 3)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertNotIn("generated_at_utc", manifest)
            self.assertEqual(manifest["config"]["backend"], "highs")
            self.assertEqual(manifest["cases"][0]["instance_sha256"], self.runs[0].instance_sha256)
            evidence = [json.loads(line) for line in (output / "evidence.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(evidence), 3)
            self.assertNotIn("_violations", evidence[0]["validation"])
            self.assertIn("applied_tolerances", evidence[0]["validation"])
            self.assertTrue(evidence[0]["validation"]["is_feasible"])
            self.assertEqual(evidence[0]["objective"]["total"], self.runs[0].objective.total)
            self.assertNotIn("multistart_result", evidence[2])
            with (output / "runs.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["solver_status"], "optimal")
            self.assertTrue(rows[0]["solver_best_bound"])
            self.assertEqual(float(rows[1]["c4"]), 31.0)
            self.assertTrue(rows[1]["gap_percent"])
            self.assertTrue(rows[1]["pdm_incumbent_comparison_percent"])
            self.assertEqual(rows[2]["agh_executed_start_count"], "6")
            self.assertFalse((output / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
