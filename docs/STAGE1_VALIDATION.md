# Stage-1 verification checkpoint

Verification covers input and placement records, equations, independent
validation, `P_DM`, M1–M3, GH and complete AGH. The publication manifest identifies
the tested files.

The tests compare values derived by hand for KV concurrency, resident storage,
budget costs and period delay charges with the equations, M2 and Pyomo
expressions. Tiny analytical optima provide independent solver references, while GH and AGH
results pass the common placement validator. Literal fixtures preserve the
established regression values under their explicit convention.

The verified implementation passes the following checks:

| Check | Result | Time |
|---|---|---|
| Stage-1 publication with licensed Gurobi and plots | 680 tests passed | 5.595 s |
| Installed wheel with base dependencies | 676 tests run; 672 passed, 4 optional skips | 3.904 s |

Both runs completed successfully. The counts differ because an unavailable
Gurobi backend skips its test class before method discovery. The base run skips
two optional Gurobi checks and two figure checks requiring Matplotlib. HiGHS coverage
remains active in both environments. Verification used Python 3.14.3, Pyomo
6.10.0 and highspy 1.14.0; licensed tests also used gurobipy 13.0.3 and
Matplotlib 3.11.2 for the optional figure check.

Each regression runs `python -m unittest discover -s tests -q`, resolving the
library from the exported source or the installed wheel, respectively.
In-memory compilation covers all 115 publication Python files, whose SHA256
inventories remained stable during testing. All 57 installed library files
match the exported source after LF normalization. The
[machine-readable checkpoint](stage1_checkpoint.json) retains that Python
inventory and both run summaries; `stage1-manifest.json` records the final
source and documentation bytes.

The base environment contains the installed wheel and its declared dependencies.
`python -m pip check` passed, and the installed quickstart returned independently
accepted GH, six-start AGH and optimal `P_DM` output using HiGHS. A second review
confirmed consistent unit factors across plain math, M2, the symbolic model and
the analytical oracle, together with the GH/AGH transition semantics.

The bounded benchmark also completed three accepted attempts in the base
environment. Focused checks cover case interpolation, coherent inputs, method
acceptance, optimality-gap eligibility, summary statistics, immutable source
capture and missing values. The figure check covers logarithmic runtime with
zero observations, independent family scales and explicitly undefined gaps.
The recorded [larger-instance campaigns](BENCHMARKS.md) each contain 24 excluded
warmups and 72 measured attempts. HiGHS accepts all 72; Gurobi accepts 69.
Three Gurobi incumbents pass the independent constraints but exceed the fixed
objective-agreement threshold by floating-point roundoff. The benchmark guide
records the two-ULP discrepancy and its diagnostic status. Each campaign retains
complete placements, validation reports, input/configuration hashes and source.

Eight comparison tests with expected values derived by hand check input,
snapshot and measurement-schedule matching, exclusion of rejected objectives
from quality summaries, the distinction between incumbents and optima, and CSV
identity. Four plotting tests check input requirements and generated figures.
An independent audit checks all 72 matched attempt pairs and 48 summary rows,
confirming identical scientific inputs and source files and exactly zero
objective change for the 48 GH/AGH pairs. The comparison CLI also passes
in the base installation without invoking a licensed solver.

Hand-derived cases isolate binding memory, storage and budget constraints.
Their HiGHS and licensed Gurobi optima agree with the independent tiny oracle.

The local checkpoint covers Python 3.14.3. The GitHub workflow runs the regression,
example, bounded benchmark, recorded backend comparison and manifest check on
Ubuntu with Python 3.13 and 3.14; results are
available in the repository's [CI history](https://github.com/antonioarbizzanitonelli/pacs-project/actions/workflows/stage1.yml).
The final export retains the tested Python inventory and incorporates the
completed documentation record.

Immutable-property tests require assignment to be rejected and accept
`FrozenInstanceError` or `TypeError`, reflecting the exceptions raised by the
supported Python patch versions for frozen, slotted records. The tests verify
that the records remain immutable.

The standalone publication configuration passes the exporter's input check in
a directory without Git history. Rebuilding the delivery reads the published
files directly and preserves their scientific source bytes.

The evidence establishes synthetic mathematical, boundary and software
verification. Its scope is limited to the documented fixtures, independent
calculations and integrations. Historical literal values and coherent-unit
results remain separately identified.
