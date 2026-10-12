# Synthetic benchmarks of P_DM, GH and AGH

The benchmark runs the three Python methods on identical synthetic inputs and
records execution time, feasibility, objective components and demand served.
The input factory uses the [coherent unit convention](UNITS.md). Each instance
is constructed once and passed to all methods; its full coefficients and
provenance are retained with the observations.

## HiGHS campaign

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
public method call. The 20-second limit is passed through as the backend's
time-limit option. The recorded solver duration covers the APPSI `solve(model)`
call, including any model transfer and work within that call. Both the call and
the complete method can exceed the requested limit. Model construction,
extraction and validation add to the complete method time.

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
retains component medians. These controlled synthetic results describe the recorded
instances and environment.

| View | PNG | SVG |
|---|---|---|
| Runtime, logarithmic axis with median and IQR | [runtime](../benchmarks/reference/plots/runtime.png) | [runtime](../benchmarks/reference/plots/runtime.svg) |
| Accepted objective by load family; proven optimality gaps | [objective and gap](../benchmarks/reference/plots/objective_and_gap.png) | [objective and gap](../benchmarks/reference/plots/objective_and_gap.svg) |
| C1–C5, independent linear scales by load family | [components](../benchmarks/reference/plots/objective_components.png) | [components](../benchmarks/reference/plots/objective_components.svg) |
| Acceptance and independent feasibility | [acceptance](../benchmarks/reference/plots/acceptance.png) | [acceptance](../benchmarks/reference/plots/acceptance.svg) |

## Gurobi campaign and backend comparison

The [Gurobi observations](../benchmarks/gurobi/summary.csv) repeat the same eight
cases with Gurobi 13.0.3, the same 20-second limit and seeds 37, 41 and 43. The
comparison verifies identical hashes for all 57 scientific Python source files
and all eight complete instances. Configuration values agree except for the
backend. The [comparison manifest](../benchmarks/backend_comparison/manifest.json)
retains those checks and each campaign's environment and input hashes.

Gurobi completes all 72 measured attempts. Independent constraint checks pass
for every placement, while 69 attempts meet the full acceptance criteria.
P_DM has six accepted optima, fifteen accepted time-limited incumbents and three
rejected incumbents. All GH and AGH attempts are accepted. The table compares
P_DM's total runtime and accepted objective medians; warmups are excluded.

| Load | Dimensions | HiGHS time [s] | Gurobi time [s] | HiGHS objective [USD] | Gurobi objective [USD] | Gurobi accepted / attempted |
|---|---|---:|---:|---:|---:|---:|
| Nominal | 2×2×2 | 0.391 | 0.076 | 120.62 | 120.62 | 3/3 |
| Nominal | 4×4×5 | 20.111 | 20.185 | 244.70 | 225.23 | 3/3 |
| Nominal | 6×6×10 | 20.511 | 20.507 | 534.56 | 329.86 | 3/3 |
| Nominal | 10×10×10 | 21.510 | 21.288 | 1,274.76 | 724.02 | 3/3 |
| ×100,000 | 2×2×2 | 0.313 | 0.085 | 10,240,018.25 | 10,240,018.25 | 3/3 |
| ×100,000 | 4×4×5 | 20.124 | 20.190 | 20,480,093.24 | 20,480,020.49 | 3/3 |
| ×100,000 | 6×6×10 | 20.659 | 20.504 | 56,407,996.57 | 30,720,022.74 | 3/3 |
| ×100,000 | 10×10×10 | 21.223 | 21.369 | 144,835,014.40 | Rejected | 0/3 |

Both backends prove optimality on the two 2×2×2 cases and recover the same
objective to floating-point precision. Gurobi's recorded total medians are
lower for those cases. Larger cases reach the solver time limit in both
campaigns. Their reported bounds leave large MIP gaps: at nominal load the
median reported gaps are approximately 97.40–99.61% for HiGHS and 88.86–99.38%
for Gurobi; accepted compute-pressure cases above 2×2×2 remain close to 100%.
These MIP gaps measure the distance between the incumbent and the solver's
bound. A heuristic's gap against an accepted optimum is reported separately.

Gurobi finds lower accepted median incumbents on nominal 4×4×5, 6×6×10 and
10×10×10. On nominal 10×10×10 its USD 724.02 incumbent is below GH's USD 1,151.06
and AGH's USD 1,150.06. This changes the ordering seen against HiGHS's
time-limited incumbent. The larger cases have no optimality certificate, so
these are comparisons of accepted solutions. Optimality gaps for GH and AGH
remain available only on 2×2×2 in each campaign.

### Rejected compute-pressure result

All three measured Gurobi P_DM attempts on 10×10×10 pass independent constraint
validation but fail the objective-agreement check. The native objective is
USD 51,200,027.25220001 and the independently recomputed value is
USD 51,200,027.25219999. Their binary64 difference is
`1.4901161193847656e-8` USD, two floating-point spacings at that scale, exceeding
the configured absolute agreement tolerance of `1e-8` USD. The warmup has the
same outcome.

The processing-delay calculation and its expanded model expression differ by
one floating-point spacing. The available records do not establish the source
of the second spacing in the native objective evaluation. Under the existing
acceptance policy, these attempts contribute runtime and feasibility
observations, with 0/3 accepted and 3/3 independently feasible. Their recomputed costs remain
diagnostic records in the [raw attempts](../benchmarks/gurobi/runs.csv).
Accepted objective, component and gap summaries for this case are unavailable;
the figures mark them as rejected.

### Heuristics and service across campaigns

The 48 matched GH and AGH attempts have exactly zero paired objective difference
in the exported records. All their placements are accepted and serve the full
modeled demand. GH remains the fastest method in each Gurobi case, with median
times from 2.62 to 387.53 milliseconds. AGH medians range from 0.056 to 16.94
seconds. Its six executed starts are deterministic, with no random start and no
strict improvement after the first incumbent, as in the HiGHS campaign.

Accepted Gurobi P_DM runs retain the same service fractions: unweighted mean
unmet demand is 7.5%, or 8.75% when weighted by arrival rates. The corresponding
served fractions are 92.5% and 91.25%. These summaries exclude the rejected
case. The [Gurobi query observations](../benchmarks/gurobi/query_metrics.csv)
preserve each query fraction and the acceptance flag.

C4 continues to dominate accepted objectives. Its share across accepted Gurobi
runs is 84.89–99.30% at nominal load and above 99.999% under compute pressure.
AGH again reduces GH's C1 rental cost by USD 1 per horizon while retaining the
same C4 and service fractions. The
[comparison summary](../benchmarks/backend_comparison/summary.csv)
contains C1–C5 medians, counts, runtime quartiles, accepted objectives, reported
bounds and MIP gaps for every method and campaign.

The [72 paired observations](../benchmarks/backend_comparison/paired_runs.csv)
match case, method, round and seed; objective differences require acceptance in
both campaigns. Runtime values are separate observations from serial campaigns
on the recorded machine. Their medians and IQRs describe those runs without
establishing a universal backend speedup.

| Comparison view | PNG | SVG |
|---|---|---|
| P_DM total runtime and solver-call medians, logarithmic axis | [runtime](../benchmarks/backend_comparison/plots/pdm_runtime.png) | [runtime](../benchmarks/backend_comparison/plots/pdm_runtime.svg) |
| Accepted P_DM objectives, independent load scales | [objectives](../benchmarks/backend_comparison/plots/pdm_objective.png) | [objectives](../benchmarks/backend_comparison/plots/pdm_objective.svg) |
| Incumbents, reported bounds/MIP gaps and outcome counts | [solver evidence](../benchmarks/backend_comparison/plots/solver_evidence.png) | [solver evidence](../benchmarks/backend_comparison/plots/solver_evidence.svg) |
| GH/AGH runtime and paired objective differences | [heuristic campaigns](../benchmarks/backend_comparison/plots/heuristic_campaigns.png) | [heuristic campaigns](../benchmarks/backend_comparison/plots/heuristic_campaigns.svg) |

The Gurobi campaign's own
[runtime](../benchmarks/gurobi/plots/runtime.png),
[objective/gap](../benchmarks/gurobi/plots/objective_and_gap.png),
[components](../benchmarks/gurobi/plots/objective_components.png) and
[acceptance](../benchmarks/gurobi/plots/acceptance.png) figures retain the
three-method view, including the distinction between acceptance and independent
feasibility.

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

The default output is `results/synthetic_benchmark/`. The runner requires a new
directory to keep observations from separate campaigns distinct. To select a
configuration or a new output directory:

```bash
python scripts/run_synthetic_benchmark.py --config configs/synthetic_benchmark.json --output results/synthetic_benchmark_run2
```

The shipped benchmark configuration explicitly selects HiGHS, which uses the
base dependencies. `SolverConfig()` defaults to Gurobi; selecting that backend
for the benchmark requires `python -m pip install -e ".[gurobi]"` and a usable
license. The backend selected in the configuration is used directly.

The [Gurobi configuration](../configs/synthetic_benchmark_gurobi.json) preserves
the cases, repetitions, seeds, physical inputs and 20-second limit. After
installing the optional package and configuring a usable license, run it in a
new output directory:

```bash
python scripts/run_synthetic_benchmark.py --config configs/synthetic_benchmark_gurobi.json --output results/synthetic_benchmark_gurobi_run2
```

P_DM uses the selected solver backend. GH and AGH use the same input records
and arithmetic in both campaigns. Their repeated observations provide
context for execution-time variation and reproducibility of the accepted
placements.

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
outside that interval. The duration of the APPSI solver call is recorded
separately when available.
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

Failed, incomplete and rejected attempts remain in the raw observations and
runtime summaries. Their status, violated constraint families and error details
remain available. Unavailable quality values are empty CSV fields and JSON
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

The [Gurobi directory](../benchmarks/gurobi/) provides the same archive layout,
with its [measurement manifest](../benchmarks/gurobi/manifest.json),
[compressed evidence](../benchmarks/gurobi/evidence.jsonl.gz) and
[captured source](../benchmarks/gurobi/source_snapshot.zip). Each archive retains
the code and configuration used in its own campaign. The comparison manifest
identifies both inputs, the comparator and the renderer, together with hashes
of the derived CSVs and figures. All 57 scientific source file hashes match;
scripts and configurations retain their separately captured versions.

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

To rebuild the backend comparison from the published observations and render
its four views, choose fresh output directories:

```bash
python scripts/compare_synthetic_benchmarks.py --reference benchmarks/reference --candidate benchmarks/gurobi --output results/backend_comparison_run2
python scripts/plot_backend_comparison.py --comparison results/backend_comparison_run2 --output results/backend_comparison_run2/plots
```

The comparator checks scientific source hashes, complete case identities and
configuration equivalence before pairing measured attempts. It reads the CSVs
and manifests; the plotter reads the derived tables. Both commands preserve the
recorded algorithm runs.
