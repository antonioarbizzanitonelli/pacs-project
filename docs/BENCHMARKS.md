# Synthetic benchmarks of P_DM, GH and AGH

The benchmark runs the three Python methods on identical synthetic inputs and
records execution time, feasibility, objective components and demand served.
The input factory uses the [coherent unit convention](UNITS.md). Each instance
is constructed once and passed to all methods; its full coefficients and
provenance are retained with the observations.

## Recorded comparison

The [reference observations](../benchmarks/reference/summary.csv) cover eight
cases, with one warmup and three measured repetitions per method: 24 warmup
attempts and 72 measured attempts. The campaign uses HiGHS with a 20-second
solver limit, seeds 37, 41 and 43, and one-hour data residency. All measured
results passed their method's acceptance criteria and the independent placement
validator. The [manifest](../benchmarks/reference/manifest.json) records Python
3.14.3, Pyomo 6.10.0, highspy 1.14.0 and the CPU environment.

Dimensions below are query types × models × GPU tiers. The compute-pressure
family multiplies the source arrival rates by 100,000 while preserving the
configured SLO and budget multipliers. Runtime medians include the complete
public method call. The solver's 20-second limit applies to the solve itself;
model construction, extraction and validation explain the larger total times.

| Load | Dimensions | P_DM termination | P_DM median [s] | GH median [ms] | AGH median [s] |
|---|---|---|---:|---:|---:|
| Nominal | 2×2×2 | Optimal | 0.391 | 2.92 | 0.067 |
| Nominal | 4×4×5 | Feasible, time limit | 20.111 | 21.53 | 0.640 |
| Nominal | 6×6×10 | Feasible, time limit | 20.511 | 104.87 | 3.790 |
| Nominal | 10×10×10 | Feasible, time limit | 21.510 | 351.72 | 15.625 |
| ×100,000 | 2×2×2 | Optimal | 0.313 | 2.62 | 0.055 |
| ×100,000 | 4×4×5 | Feasible, time limit | 20.124 | 19.31 | 0.597 |
| ×100,000 | 6×6×10 | Feasible, time limit | 20.659 | 122.92 | 4.601 |
| ×100,000 | 10×10×10 | Feasible, time limit | 21.223 | 332.40 | 14.542 |

GH has the lowest median runtime in each recorded case. AGH's search grows from
roughly 0.06 seconds on 2×2×2 to 14.5–15.6 seconds on 10×10×10. P_DM proves
optimality on both 2×2×2 cases; the other six cases retain feasible incumbents
at the time limit. Across the measured P_DM runs, six are optimal and eighteen
are feasible without an optimality certificate.

Every measured AGH run executes six deterministic starts. Its existing early
stopping rule ends the schedule before any random start, and subsequent starts
produce no strict incumbent improvement. The raw observations retain scheduled
and executed start counts. These measurements therefore describe the executed
deterministic search under the configured seeds.

The median accepted objectives are:

| Load | Dimensions | P_DM [USD] | GH [USD] | AGH [USD] |
|---|---|---:|---:|---:|
| Nominal | 2×2×2 | 120.62 | 359.42 | 358.42 |
| Nominal | 4×4×5 | 244.70 | 514.62 | 513.62 |
| Nominal | 6×6×10 | 534.56 | 767.44 | 766.44 |
| Nominal | 10×10×10 | 1,274.76 | 1,151.06 | 1,150.06 |
| ×100,000 | 2×2×2 | 10,240,018.25 | 35,040,009.05 | 35,040,008.05 |
| ×100,000 | 4×4×5 | 20,480,093.24 | 50,560,009.09 | 50,560,008.09 |
| ×100,000 | 6×6×10 | 56,407,996.57 | 75,840,009.14 | 75,840,008.14 |
| ×100,000 | 10×10×10 | 144,835,014.40 | 114,200,009.22 | 114,200,008.22 |

AGH reduces GH's GPU rental component by USD 1 per planning horizon in every
case, with identical modeled delay charges and unmet fractions. On both
10×10×10 cases, GH and AGH have lower objectives than the time-limited P_DM
incumbent. Those differences compare feasible solutions. Optimality gaps are
available only on 2×2×2: GH and AGH have gaps of 197.97% and 197.14% at nominal
load, and approximately 242.19% under compute pressure.

Service also differs between methods. In every measured P_DM run the unweighted
mean unmet fraction across query types is 7.5%, giving a mean served fraction
of 92.5%. Weighting the query fractions by their recorded arrival rates gives
8.75% unmet requests and 91.25% served requests. These values satisfy the
instance-specific unmet-demand limits. GH and AGH serve the full modeled demand
in every measured run. The [query observations](../benchmarks/reference/query_metrics.csv)
retain each fraction and its modeled delay/error constraints.

The coherent delay charge C4 drives most of the objective: its share across
measured accepted runs is 83.59–99.30% at nominal load and above 99.999% under
compute pressure. This follows from charging modeled processing delay for the
arrival count over the planning horizon. Changes in served fractions and
placement affect that large component. The [raw observations](../benchmarks/reference/runs.csv)
include all five costs, solver bounds, termination and status; the summary
retains their medians. These controlled synthetic results describe the recorded
instances and environment.

| View | PNG | SVG |
|---|---|---|
| Runtime, logarithmic axis with median and IQR | [runtime](../benchmarks/reference/plots/runtime.png) | [runtime](../benchmarks/reference/plots/runtime.svg) |
| Accepted objective by load family; proven optimality gaps | [objective and gap](../benchmarks/reference/plots/objective_and_gap.png) | [objective and gap](../benchmarks/reference/plots/objective_and_gap.svg) |
| C1–C5, independent linear scales by load family | [components](../benchmarks/reference/plots/objective_components.png) | [components](../benchmarks/reference/plots/objective_components.svg) |
| Acceptance and independent feasibility | [acceptance](../benchmarks/reference/plots/acceptance.png) | [acceptance](../benchmarks/reference/plots/acceptance.svg) |

## Run the benchmark

From the repository root, install the project and run the configured campaign:

```bash
python -m pip install -e .
python scripts/run_synthetic_benchmark.py
```

The default configuration is [configs/synthetic_benchmark.json](../configs/synthetic_benchmark.json).
It selects the cases, local random seeds, warmups, GH budget fraction, data
residency, solver backend and solver time limit. Each seed defines one measured
round of all three methods. Warmups use the first seed. Methods run serially;
their order rotates between measured rounds to reduce systematic order bias.
The seeds control AGH's local random ordering stream. GH is deterministic for
each supplied instance; the same round labels pair all methods' observations.
The round seeds do not set a solver seed; native solver defaults remain active.

The default output is `results/synthetic_benchmark/`. An existing directory is
refused so observations from separate campaigns remain distinct. To select a
configuration or a new output directory:

```bash
python scripts/run_synthetic_benchmark.py --config configs/synthetic_benchmark.json --output results/synthetic_benchmark_run2
```

The shipped benchmark configuration explicitly selects HiGHS, which uses the
base dependencies. `SolverConfig()` defaults to Gurobi; selecting that backend
for the benchmark requires `python -m pip install -e ".[gurobi]"` and a usable
license. The backend selected in the configuration is used directly.

## Cases and measurement scope

Each `case_specs` entry gives a unique `case_id`, a `family` label, three
`dimensions` in query/model/GPU-tier order, and explicit load, delay-SLO and
budget multipliers. Cardinalities are integers of at least two. The shared
`data_residency_hours` value applies to every case.

The joint-size family uses 2×2×2, 4×4×5, 6×6×10 and 10×10×10 dimensions at two
load levels. Query types alternate the two heterogeneous source templates.
Added models and tiers interpolate their source attributes and coefficients;
the original alternatives remain available. Budget and storage capacity scale
with the number of query types. The manifest records the exact generation rule
and every resulting instance.

These cases vary both the decision space and the workload. A larger instance
can therefore have a different optimum and resource mix. Comparisons between
methods use the same case and measured round. The fixtures represent controlled
synthetic inputs, with fixed coefficients describing modeled delay and error.

Execution time uses a monotonic clock around each public method call. It
includes input checks, P_DM problem/model construction, solve, extraction and
independent validation. Imports, source capture, file export and plotting occur
outside that interval. Solver-only time is recorded separately when available.
P_DM uses the configured time limit and zero requested absolute/relative MIP
gaps; other solver settings follow the backend defaults.

Runtime summaries include every measured attempt and exclude warmups. They
report the median and inclusive interquartile range. An interquartile range
describes the observed repetitions; timing also depends on the recorded machine
and software environment.

## Acceptance and quality

Completion, independent feasibility and solver termination remain separate
columns. P_DM acceptance requires an incumbent, independent feasibility and
agreement between solver and recomputed objectives. GH requires completion and
independent feasibility. AGH uses its complete, independently validated result.

Quality summaries use accepted runs. C1–C5 and their total are recomputed by the
common validator and expressed in USD under the configured physical convention.
They describe GPU rental, admitted model storage, resident data storage, delay
charges and unmet-demand charges. The sum of component medians can differ from
the median total. Per-query output records unmet fractions, modeled delay and
modeled weighted error alongside their limits.

The signed optimality gap is `100 * (J - J*) / abs(J*)`. It is available when
P_DM produced an accepted `OPTIMAL` result for the same case and measured round,
with a nonzero reference objective. Small negative numerical differences remain
visible. A comparison with a time-limited P_DM incumbent is recorded separately
as `pdm_incumbent_comparison_percent`; it measures a difference between accepted
objectives. Solver bounds, termination and relative MIP gap remain available to
interpret that incumbent.

Failed, incomplete and independently rejected attempts remain in the raw
observations and runtime denominator. Their status, violated families and error
details stay visible. Unavailable quality values are empty CSV fields and JSON
nulls. The synthetic results establish performance on the recorded instances
and environments; heuristic agreement with an exact result applies to the
particular case that was checked.

## Files and reproducibility

| File | Contents |
|---|---|
| `runs.csv` | Every warmup and measured attempt, timing, acceptance, objective, gaps and solver status. |
| `summary.csv` | Per-case/method counts, runtime quartiles and accepted quality metrics. |
| `query_metrics.csv` | Available query-level service, delay and error observations. |
| `evidence.jsonl` | Compact placement, validation, objective and solver records. |
| `manifest.json` | Configuration, complete instances, environment and source/configuration hashes. |
| `source_snapshot/` | Captured Python source, scripts, package metadata and the exact input configuration. |
| `partial_runs.jsonl` and `partial_manifest.json` | Completed attempts flushed during execution, with snapshot/configuration identity and campaign status. |

Source capture precedes scientific imports. The runner imports the captured
library directly, and the manifest identifies snapshot paths relative to the
output. This makes the measured code and inputs recoverable even when the
working repository changes afterward. The configuration hash covers the
original JSON bytes. Run the CLI in a fresh Python process so source loading is
unambiguous.

The console reports the case, method, round and outcome as execution progresses.
If execution is interrupted, the partial files retain completed attempts. The
active attempt has no completed observation until its method returns. Restart a
campaign in a new output directory and preserve the partial run as a separate
record.

The published [reference directory](../benchmarks/reference/) contains the three
CSVs, full manifest, [compressed evidence](../benchmarks/reference/evidence.jsonl.gz)
and [source archive](../benchmarks/reference/source_snapshot.zip). The archive
contains `source_snapshot/` with the exact measured scientific code, scripts,
configuration and package metadata, plus `measured_manifest.json` with the
original measurement manifest. The publication manifest adds archive and file
hashes to that recorded scientific information.

The published figures were regenerated from the recorded summary using the
current plotter to separate load scales and display logarithmic runtimes. Its
renderer hash is retained in the publication manifest. The archive preserves the
original captured plotter alongside the measured scientific library.

To inspect the full evidence and captured code, run this Python snippet from
the repository root, choosing a fresh recovery directory:

```python
from pathlib import Path
import gzip
import shutil
import zipfile

target = Path("results/recovered_benchmark")
target.mkdir(parents=True, exist_ok=False)
with zipfile.ZipFile("benchmarks/reference/source_snapshot.zip") as archive:
    archive.extractall(target)
with gzip.open("benchmarks/reference/evidence.jsonl.gz", "rb") as source:
    with (target / "evidence.jsonl").open("wb") as destination:
        shutil.copyfileobj(source, destination)
```

## Plot recorded results

Install the optional plotting dependency and create figures with the campaign:

```bash
python -m pip install -e ".[plots]"
python scripts/run_synthetic_benchmark.py --output results/synthetic_benchmark_with_plots --plots
```

Figures can also be generated from an existing summary without executing the
algorithms:

```bash
python scripts/plot_synthetic_benchmark.py --input results/synthetic_benchmark/summary.csv --output results/synthetic_benchmark/figures
```

The plotter writes PNG and SVG views of runtime, accepted objective and optimality
gap, all five cost components, and method acceptance/independent feasibility.
Missing objectives and undefined gaps have explicit markers. The figure output
directory must be new. Runtime uses a declared logarithmic axis with explicit
markers for nonpositive values; objectives and cost components use independent
linear scales for each load family. The recorded numerical values remain intact.
