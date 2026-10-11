# Stage-1 verification checkpoint

Verification on 2026-10-11 covers domain records, equations, independent
validation, `P_DM`, M1–M3, GH and complete AGH. The publication manifest identifies
the tested delivery.

The D-0101 tests compare hand-derived KV concurrency, resident storage, budget
costs and period delay with the plain equations, M2 and Pyomo expressions. Tiny
analytical optima provide independent solver references, while GH and AGH
results pass the common placement validator. Literal fixtures preserve the
established regression values under their explicit convention.

The stable implementation passed these checks on 2026-10-11:

| Check | Result | Time |
|---|---|---|
| Stage-1 publication with licensed Gurobi | 647 tests passed | 4.024 s |
| Installed wheel with base dependencies | 643 tests run; 641 passed, 2 optional Gurobi skips | 3.489 s |

Both runs completed successfully. The counts differ because an unavailable
Gurobi backend skips its test class before method discovery. HiGHS coverage
remains active in both environments. Verification used Python 3.14.3, Pyomo
6.10.0 and highspy 1.14.0; licensed tests also used gurobipy 13.0.3.

Each regression runs `python -m unittest discover -s tests -q`, resolving the
library from the exported source or the installed wheel, respectively.
In-memory compilation covers all 102 publication Python files, whose SHA256
inventories remained stable during testing. All 53 installed library files
match the exported source after LF normalization. The
[machine-readable checkpoint](stage1_checkpoint.json) retains that Python
inventory and both run summaries; `stage1-manifest.json` records the final
source and documentation bytes.

The base environment contains the installed wheel and its declared dependencies.
`python -m pip check` passed, and the installed quickstart returned independently
accepted GH, six-start AGH and optimal `P_DM` output using HiGHS. A second review
confirmed consistent unit factors across plain math, M2, the symbolic model and
the analytical oracle, together with the GH/AGH transition semantics.

Hand-derived cases isolate binding memory, storage and budget constraints.
Their HiGHS and licensed Gurobi optima agree with the independent tiny oracle.

The GitHub workflow targets Python 3.13 and 3.14. The executed checkpoint covers
local Python 3.14.3; hosted CI and Python 3.13 execution remain unverified.
The final export retains the tested Python inventory and incorporates the
completed documentation record.

The standalone publication configuration passes the exporter's input check in
a directory without Git history. Rebuilding the delivery reads the published
files directly and preserves their scientific source bytes.

The evidence establishes synthetic mathematical, boundary and software
verification. Its scope is limited to the documented fixtures, independent
calculations and integrations. Historical literal values and coherent-unit
results remain separately identified.
