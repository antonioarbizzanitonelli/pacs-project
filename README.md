# LLM Resource Allocation — Stage 1

This repository provides a Python reference implementation for allocating LLM
workloads across heterogeneous GPU tiers under latency, error, capacity and
budget constraints. Stage 1 includes deterministic `P_DM`, the M1–M3 mechanisms,
Greedy Heuristic (GH) and Adaptive Greedy Heuristic (AGH).

The formulation follows [paper v2](https://arxiv.org/abs/2604.07472v2), with
[v1](https://arxiv.org/abs/2604.07472v1) providing the secondary scientific
reference. The [units guide](docs/UNITS.md) explains the dimensional adjustments
and their assumptions. New instances use consistent units by default;
historical test fixtures explicitly retain the paper's original arithmetic.

## Install and run

Python 3.13 or later is required. From the repository root:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on Linux/macOS, then run:

```bash
python -m pip install -e .
python examples/stage1_quickstart.py
python -m unittest discover -s tests -q
```

The example runs GH, AGH and `P_DM` on a synthetic instance and validates each
result independently. It selects HiGHS explicitly and runs on a CPU. The base
dependencies are Pyomo 6.10.0 and highspy 1.14.0; tests use Python's `unittest`.

`SolverConfig()` defaults to Gurobi. Install
`python -m pip install -e ".[gurobi]"` and configure a license to use it. With the
base installation, pass `SolverConfig(backend=SolverBackend.HIGHS)`.
Generic solver tests use HiGHS, while dedicated Gurobi tests run when its package
and license are available. GH and AGH work directly with the project's input
and placement records.

## Read and verify

Start with the [reading guide](docs/STAGE1_GUIDE.md), then the
[example](examples/stage1_quickstart.py), [units](docs/UNITS.md) and
[validation results](docs/STAGE1_VALIDATION.md).

The [synthetic benchmark](docs/BENCHMARKS.md) compares P_DM, GH and AGH on shared
instances from 2×2×2 to 10×10×10 at two load levels. It records runtime,
independent feasibility, demand served, cost components and solver outcomes.
The supplied configurations and runner allow the comparison to be repeated
with the same unit convention.

The [HiGHS campaign](docs/BENCHMARKS.md#highs-campaign) includes 72
measured attempts, all independently accepted. GH's median runtime ranges from
2.6 to 352 milliseconds and AGH's from 0.055 to 15.6 seconds. P_DM proves
optimality for the two 2×2×2 cases; larger cases retain feasible incumbents at
the configured 20-second solve limit. GH and AGH serve all modeled demand,
while P_DM uses permitted unmet fractions. The coherent delay charge dominates
the objective, making its assumptions central to interpreting the comparison.

The [summary CSV](benchmarks/reference/summary.csv),
[raw attempts](benchmarks/reference/runs.csv),
[query observations](benchmarks/reference/query_metrics.csv) and
[manifest](benchmarks/reference/manifest.json) retain the measurements and
provenance. [Runtime](benchmarks/reference/plots/runtime.png) and
[objective/gap](benchmarks/reference/plots/objective_and_gap.png) figures show
execution times, accepted objectives and available optimality gaps.

The [Gurobi campaign and backend comparison](docs/BENCHMARKS.md#gurobi-campaign-and-backend-comparison)
repeat the same cases and scientific code with Gurobi 13.0.3. Gurobi also proves
the two 2×2×2 cases optimal and finds lower accepted incumbents on the larger
nominal cases; these larger results remain time limited. Its campaign accepts
69 of 72 measured attempts. Three P_DM attempts on compute-pressure 10×10×10
pass the constraints but fail objective agreement by `1.49e-8` USD at a
USD 51.2-million scale, just above the configured absolute tolerance. Their
objectives are excluded from quality summaries. GH and AGH are accepted
throughout, with identical paired objectives across campaigns.

The [paired CSV](benchmarks/backend_comparison/paired_runs.csv),
[comparison summary](benchmarks/backend_comparison/summary.csv) and
[manifest](benchmarks/backend_comparison/manifest.json) retain every measured
attempt. [Backend runtime](benchmarks/backend_comparison/plots/pdm_runtime.png)
and [solver evidence](benchmarks/backend_comparison/plots/solver_evidence.png)
figures distinguish accepted optima, feasible incumbents and rejected results.

| Source layer | Responsibility |
|---|---|
| `domain/` | Immutable inputs, units and input contracts. |
| `data/` | Tiny verification fixtures and controlled synthetic instances. |
| `math/` | Decision state and solver-independent equations. |
| `algorithms/` | M1–M3, GH, relocate, consolidation and AGH multi-start. |
| `optimization/` | P_DM, solver adapter, extraction and run evidence. |
| `validation/` | Independent constraint and objective checks. |
| `verification/` | Analytical optimum oracle restricted to 1×1×1. |
| `experiments/` | Shared-instance measurements, summaries and serialized evidence. |

These layers live in `src/llm_resource_allocation/`. All accepted complete
placements pass the same backend-neutral validator. Construction completion,
scientific feasibility and solver optimality remain separate outcomes.

## Repository contents and verification scope

The Stage-1 files are listed in an explicit
[allowlist](configs/stage1_publication.json). The
[preparation guide](docs/GITHUB_PREPARATION.md) explains how to export and verify
them. The generated manifest identifies every exported file and its SHA256 hash.

Tests use synthetic instances that can be checked by hand to verify equation
consistency, constraint satisfaction and algorithm behavior. Benchmark results
describe the supplied synthetic workloads and record their unit convention and
solver status. Historical fixtures retain the original arithmetic for regression
testing. A reuse license has yet to be selected, as recorded in the
[package metadata](pyproject.toml).
