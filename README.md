# LLM Resource Allocation — Stage 1

This repository provides a Python reference implementation for allocating LLM
workloads across heterogeneous GPU tiers under latency, error, capacity and
budget constraints. Stage 1 includes deterministic `P_DM`, the M1–M3 mechanisms,
Greedy Heuristic (GH) and Adaptive Greedy Heuristic (AGH).

The formulation follows [paper v2](https://arxiv.org/abs/2604.07472v2), with
[v1](https://arxiv.org/abs/2604.07472v1) providing the secondary scientific
reference. The [unit contract](docs/UNITS.md) records the project's dimensional
adjustments and their assumptions. New instances use coherent units by default;
historical fixtures select the literal convention explicitly.

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
and license are available. GH and AGH operate directly on plain domain records.

## Read and verify

Start with the [reading guide](docs/STAGE1_GUIDE.md), then the
[example](examples/stage1_quickstart.py), [units](docs/UNITS.md) and
[performed validation](docs/STAGE1_VALIDATION.md).

The [synthetic benchmark](docs/BENCHMARKS.md) compares P_DM, GH and AGH on shared
instances from 2×2×2 to 10×10×10 at two load levels. It records runtime,
independent feasibility, demand served, cost components and solver evidence.
The configuration and runner reproduce the comparison with the coherent unit
convention.

The [recorded comparison](docs/BENCHMARKS.md#recorded-comparison) includes 72
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
the measured time and quality together with their availability.

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

## Delivery scope

The reviewed Stage-1 delivery uses an explicit
[allowlist](configs/stage1_publication.json) and
[preparation procedure](docs/GITHUB_PREPARATION.md). The generated manifest
identifies every delivered file and its SHA256.

The verification uses synthetic, hand-checkable instances to establish equation
consistency, constraint satisfaction and algorithm behavior. This evidence is
limited to those checks; the reported results carry their unit convention and
solver status. Historical fixtures retain the original arithmetic for regression
testing. The reuse license remains an open project choice, recorded in the
[package metadata](pyproject.toml).
