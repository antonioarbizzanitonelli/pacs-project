# Stage-1 units and scientific contract

D-0101, approved on 2026-10-11, is the current interpretation for dimensionally
coherent Stage-1 controls and costs. It explicitly departs from printed memory,
storage and delay-cost expressions. `UnitConvention.LITERAL_PAPER` preserves
the previous convention for historical evidence.

## Inputs and physical factors

New `ProblemInstance` records default to `UnitConvention.COHERENT` and
`data_residency_hours=1.0`. Use `make_coherent_tiny_instance()` for the current
synthetic example. Existing tiny factories explicitly retain literal units and
their hand-derived golden values. To convert one, use `dataclasses.replace`
with `unit_convention=UnitConvention.COHERENT` and the desired residency.

| Quantity | Unit |
|---|---|
| Arrival rate `λ` | requests/hour |
| Tokens `h`, `f`, `r` | tokens/request |
| Model weight `B` | decimal GB |
| KV and stored-data footprints `β`, `θ` | decimal GB/token |
| KV residency `T_res` | seconds/request |
| Data residency `R` | hours; configurable, nonnegative |
| Optimization horizon `H` | hours; positive |
| Compute coefficient `α` | GFLOP/token |
| GPU throughput `P` | TFLOP/second |
| Processing delay `D_proc` and SLO | seconds/request |
| Storage price `p_s` | USD/(GB·hour) |
| Delay penalty `ρ` | USD/(millisecond·request) |
| Unmet penalty `φ` | USD/hour for an unmet fraction of one |

`domain/units.py` owns 1,000,000 KB/GB, 1,000 GFLOP/TFLOP,
1,000 milliseconds/second and 3,600 seconds/hour. KB/GB are decimal units,
distinct from KiB/GiB. Convert footprints once at the input boundary.
The legacy `seconds_per_hour` field remains configurable for old synthetic
fixtures; coherent input validation requires exactly 3,600.

## Resource and monetary formulas

With TP/PP degrees `n,m` and the supplied auxiliary decision `v`:

```text
weight GB per GPU = Σ_(n,m) B_j/(n m) w[j,k,n,m]
KV GB per GPU = Σ_(i,n,m) β_j r_i T_res[i,j,k] (λ_i/3600) v[i,j,k,n,m]/(n m)
weight GB + KV GB <= C_gpu[k] q[j,k]

data rate_i GB/hour = Σ_(j,k) θ_i r_i λ_i x[i,j,k]
resident data_i GB = R * data rate_i
Σ_(j,k) B_j z[i,j,k] + resident data_i GB <= C_s       for each query i

required TFLOP/hour = Σ_i α[i,j,k] r_i λ_i x[i,j,k]/1000
available TFLOP/hour = η * 3600 * P_gpu[k] * y[j,k]

C1 = H Σ_(j,k) p_gpu[k] y[j,k]
C2 = H p_s Σ_(i,j,k) B_j z[i,j,k]
C3 = H p_s R Σ_(i,j,k) θ_i r_i λ_i x[i,j,k]
C4 = Σ_i ρ_i * 1000 * D_proc[i] * λ_i * H
C5 = H Σ_i φ_i u[i]
objective USD = C1+C2+C3+C4+C5
resource budget USD = C1+C2+C3 <= budget
```

KV stock represents average occupancy under the supplied arrival rate and
residency, `arrival/second * residency/seconds`. `T_res` is an explicit input.
Data storage represents steady resident stock with lifetime `R`; setting `R=0`
models immediate data deletion while admitted model weights remain in storage.

The per-query storage closure and query-indexed weights remain unchanged
(D-0016/D-0005). Each row accounts for its query's admitted model weights.
GPU capacity is measured per GPU and gated by `q`. `D_proc` includes served
fractions through `v`, so C4 applies the arrival count directly. M2 uses the same
resident-data and period-delay factors in its marginal ranking cost.

Coherent mode supports primary `PaperVersion.V2` and raises `ValueError` for V1.
Literal mode supports both versions with their original C5 difference.
Validation reports snapshot `unit_convention` and `data_residency_hours`.
The diagnostic `printed_*` names remain for compatibility; coherent callers can
read `memory_usage_gb`, `kv_memory_usage_gb`, `storage_usage_gb` and
`data_storage_usage_gb`. The GB-only properties reject literal mixed-unit values.

Unit factors belong to the scientific expressions. Acceptance tolerances
classify the resulting residuals, and validation evaluates the supplied `v`
directly. Solver tolerances, normalization, independent acceptance and strict
objective improvement have separate contracts.

## Provenance

SI factors are **DERIVED**. Mean concurrency, resident stock and period delay
cost are approved **SCIENTIFIC INTERPRETATION**. The configurable one-hour
residency is **ASSUMED**. Explicit enums, aliases and
preserved historical fixtures are **IMPLEMENTATION CHOICE**. D-0101 supersedes
literal dimensional defaults from D-0002/D-0008/D-0016 while preserving their
linearization and index structure.

Literal fixtures retain their original convention as regression evidence.
