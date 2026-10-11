# Reading and running the Stage-1 reference

The Python reference implements deterministic placement (`P_DM`), the M1–M3
mechanisms, GH and complete AGH. Every accepted complete placement passes the
same solver-independent feasibility validator. Synthetic instances with
hand-derived expected values establish the verification scope.

The publication manifest identifies the Stage-1 source, tests and documentation
delivered here.

## Start with a working example

Use Python 3.13 or later. Create and activate a virtual environment from the
repository root:

```bash
python -m venv .venv
```

In PowerShell, activation is `.venv\Scripts\Activate.ps1`; on Linux/macOS it is
`source .venv/bin/activate`. Then run:

```bash
python -m pip install -e .
python examples/stage1_quickstart.py
python -m unittest discover -s tests -q
```

The example selects HiGHS explicitly for `P_DM` and runs both heuristics on a tiny
dimensionally coherent instance. `make_coherent_tiny_instance` is the starting
fixture for coherent-unit checks; historical factories retain the literal-paper
convention for their golden cases. GH and AGH evaluate plain domain records.
Pyomo and highspy provide the exact-solve path, and the example runs on a CPU
using the supplied synthetic inputs.

Gurobi is optional: install `python -m pip install -e ".[gurobi]"` and configure
a usable license. `SolverConfig()` defaults to Gurobi, so pass
`SolverConfig(backend=SolverBackend.HIGHS)` when using the base installation.
Dedicated Gurobi integration cases skip if that backend is unavailable; the
generic placement regression explicitly uses HiGHS. Package requirements are
specified in `pyproject.toml`; the paper's reported software versions remain
part of its experimental provenance.

## Read the layers in this order

| Layer | Where to start | Responsibility |
|---|---|---|
| Inputs | [domain/types.py](../src/llm_resource_allocation/domain/types.py), [domain/validation.py](../src/llm_resource_allocation/domain/validation.py) | Query/model/GPU records, units and input contracts. |
| Tiny data | [data/tiny_instances.py](../src/llm_resource_allocation/data/tiny_instances.py) | Hand-checkable synthetic cases. |
| Decisions and equations | [math/placement.py](../src/llm_resource_allocation/math/placement.py), [math/objective.py](../src/llm_resource_allocation/math/objective.py) | Immutable decision state and pure scientific arithmetic. |
| Independent validation | [validation/placement.py](../src/llm_resource_allocation/validation/placement.py) | Domain checks, recomputed objective, residuals and numerical acceptance. |
| GH | [algorithms/gh.py](../src/llm_resource_allocation/algorithms/gh.py), then `m1.py`, `m2.py`, `m3.py` | Deployment construction, routing and parallelism upgrades. |
| AGH | [algorithms/agh.py](../src/llm_resource_allocation/algorithms/agh.py), then `agh_start.py` | Ordering, GH construction, relocate, consolidation and multi-start selection. |
| Exact solve | [optimization/pdm_run.py](../src/llm_resource_allocation/optimization/pdm_run.py) | Model construction, backend execution, extraction and independent acceptance. |
| Independent tiny oracle | [verification/tiny_enumeration.py](../src/llm_resource_allocation/verification/tiny_enumeration.py) | Analytic reference for the supported 1×1×1 scope. |

Both execution paths return a placement that is checked independently:

```text
ProblemInstance -> GH/AGH or P_DM -> PlacementDecisionState
                                      -> independent validation -> accepted result
```

Import public functions from their defining modules; the `algorithms` package
also exposes a curated API. `math`, `optimization` and `validation` initializers
stay lightweight. The only package name is `llm_resource_allocation`.

## Understand the scientific state

Identifiers express meaning independently of container order. A route key is
`(query_id, model_id, gpu_tier_id)`; a deployment pair is
`(model_id, gpu_tier_id)`; configuration keys add TP and PP degrees.

| Symbol | Meaning |
|---|---|
| `x`, `u` | Routed and unmet demand fractions. |
| `z`, `q` | Route admission and model/tier deployment indicators. |
| `w`, `y` | Chosen TP/PP configuration and allocated GPU count. |
| `v` | Auxiliary product of routing and configuration selection. |

Input and decision records own immutable copies. Missing known sparse keys mean
zero; unknown identifiers and unsupported configurations are invalid. The
validator evaluates the supplied `v` directly, preserving the extracted decision
and its constraint residuals.

GH's construction completion and independent feasibility are separate outcomes.
AGH retains only feasible starts and applies strict objective improvement. Its
local search transfers complete positive routes, uses deterministic traversal,
and keeps explicit seeded ordering evidence. Heuristic results represent
accepted feasible placements. Proven optimality requires the appropriate exact
solver termination; a time-limit incumbent records feasibility and termination
separately.

The monetary and resource unit contract, including any explicit divergence from
printed equations, belongs in [UNITS.md](UNITS.md). [STAGE1_VALIDATION.md](STAGE1_VALIDATION.md)
records the checks actually performed on the prepared publication. These should
be read before interpreting objectives or changing tolerances.

When reporting a result, retain instance identity, unit convention, selected
configuration, numerical policy and provenance alongside the validated placement.

The [preparation procedure](GITHUB_PREPARATION.md) explains how to assemble and
review the publication. Package metadata records the pending reuse-license
choice.
