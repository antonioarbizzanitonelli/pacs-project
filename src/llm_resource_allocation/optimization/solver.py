"""Invoke HiGHS or Gurobi and normalize solver evidence (D-0080).

Extraction and independent scientific validation are separate operations.
"""

from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from importlib.util import find_spec
from math import isfinite
from numbers import Real
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

from pyomo.contrib.appsi.base import Results, TerminationCondition
from pyomo.contrib.appsi.solvers import Gurobi, Highs


class SolverBackend(str, Enum):
    """Identify the supported HiGHS and licensed Gurobi backends."""

    HIGHS = "highs"
    GUROBI = "gurobi"


class SolverBackendAvailability(str, Enum):
    """Classify a backend probe without conflating absence and license failure."""

    AVAILABLE = "available"
    INSTALLED_BUT_UNUSABLE = "installed_but_unusable_or_unlicensed"
    UNAVAILABLE = "unavailable"
    NOT_RELIABLY_DETECTABLE = "not_reliably_detectable"


class SolverStatus(str, Enum):
    """Expose backend-independent solve outcomes to callers."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    INFEASIBLE_OR_UNBOUNDED = "infeasible_or_unbounded"
    NO_SOLUTION = "no_solution"
    ERROR = "error"
    UNKNOWN = "unknown"


class SolverAdapterError(RuntimeError):
    """Base error for solver-adapter setup or incumbent-safety failures."""


class UnsupportedSolverBackendError(SolverAdapterError):
    """Report a backend unsupported by the requested adapter operation."""


class SolverBackendUnavailableError(SolverAdapterError):
    """Report that the selected implemented backend cannot be invoked."""


class NoSolverIncumbentError(SolverAdapterError):
    """Prevent solution extraction after an outcome with no incumbent."""


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Configure backend, time limit in seconds, MIP gaps, and logging.

    Gurobi is the default (D-0096); ``None`` uses native backend defaults.
    """

    backend: SolverBackend = SolverBackend.GUROBI
    time_limit_seconds: float | None = None
    relative_mip_gap: float | None = None
    absolute_mip_gap: float | None = None
    tee: bool = False

    def __post_init__(self) -> None:
        """Require a backend enum, boolean logging, and finite non-negative options."""

        if not isinstance(self.backend, SolverBackend):
            raise ValueError("backend must be a SolverBackend")
        if not isinstance(self.tee, bool):
            raise ValueError("tee must be a boolean")
        for field_name in (
            "time_limit_seconds",
            "relative_mip_gap",
            "absolute_mip_gap",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(float(value))
                or value < 0.0
            ):
                raise ValueError(
                    f"{field_name} must be None or a finite non-negative real number"
                )


@dataclass(frozen=True, slots=True)
class SolverBackendInfo:
    """Store availability, version, and diagnostic detail for one backend."""

    backend: SolverBackend
    availability: SolverBackendAvailability
    version: str | None
    detail: str


@dataclass(frozen=True, slots=True)
class SolverResult:
    """Store solve status, objective/bound/gap, timing, and backend diagnostics.

    ``OPTIMAL`` applies to the constructed model; independent validation is
    still required. Solve time is in seconds.
    """

    backend: SolverBackend
    status: SolverStatus
    termination_condition: str
    has_incumbent: bool
    objective_value: float | None
    best_bound: float | None
    relative_gap: float | None
    solve_time_seconds: float
    raw_status_metadata: Mapping[str, str | None]

    def __post_init__(self) -> None:
        """Validate result invariants and freeze diagnostic metadata."""

        if not isinstance(self.backend, SolverBackend):
            raise ValueError("backend must be a SolverBackend")
        if not isinstance(self.status, SolverStatus):
            raise ValueError("status must be a SolverStatus")
        if not isinstance(self.termination_condition, str):
            raise ValueError("termination_condition must be a string")
        if not isinstance(self.has_incumbent, bool):
            raise ValueError("has_incumbent must be a boolean")
        for field_name in ("objective_value", "best_bound", "relative_gap"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(float(value))
            ):
                raise ValueError(f"{field_name} must be None or a finite real number")
        if self.relative_gap is not None and self.relative_gap < 0.0:
            raise ValueError("relative_gap must be non-negative")
        if (
            isinstance(self.solve_time_seconds, bool)
            or not isinstance(self.solve_time_seconds, Real)
            or not isfinite(float(self.solve_time_seconds))
            or self.solve_time_seconds < 0.0
        ):
            raise ValueError("solve_time_seconds must be finite and non-negative")
        if not self.has_incumbent and self.objective_value is not None:
            raise ValueError("objective_value requires an incumbent")
        object.__setattr__(
            self,
            "raw_status_metadata",
            MappingProxyType(dict(self.raw_status_metadata)),
        )


def probe_solver_backend(backend: SolverBackend) -> SolverBackendInfo:
    """Probe backend availability and version without solving a model.

    Gurobi probing imports ``gurobipy`` and starts a quiet licensed environment.
    """

    if not isinstance(backend, SolverBackend):
        raise ValueError("backend must be a SolverBackend")
    if backend is SolverBackend.HIGHS:
        solver = Highs()
        availability = solver.available()
        version_tuple = solver.version() if bool(availability) else None
        return SolverBackendInfo(
            backend=backend,
            availability=(
                SolverBackendAvailability.AVAILABLE
                if bool(availability)
                else SolverBackendAvailability.UNAVAILABLE
            ),
            version=(
                ".".join(str(component) for component in version_tuple)
                if version_tuple is not None
                else None
            ),
            detail=f"Pyomo APPSI HiGHS availability: {availability.name}",
        )

    if find_spec("gurobipy") is None:
        return SolverBackendInfo(
            backend=backend,
            availability=SolverBackendAvailability.UNAVAILABLE,
            version=None,
            detail="gurobipy is not installed",
        )

    try:
        gurobipy = import_module("gurobipy")
        version_tuple = gurobipy.gurobi.version()
        environment = gurobipy.Env(empty=True)
        try:
            environment.setParam("OutputFlag", 0)
            environment.start()
        finally:
            environment.dispose()
    except Exception as error:  # optional proprietary backend boundary
        return SolverBackendInfo(
            backend=backend,
            availability=SolverBackendAvailability.INSTALLED_BUT_UNUSABLE,
            version=None,
            detail=f"{type(error).__name__}: {error}",
        )
    return SolverBackendInfo(
        backend=backend,
        availability=SolverBackendAvailability.AVAILABLE,
        version=".".join(str(component) for component in version_tuple),
        detail="gurobipy imported and a licensed environment started",
    )


def translate_highs_options(config: SolverConfig) -> Mapping[str, float]:
    """Map configured options to HiGHS ``time_limit/mip_rel_gap/mip_abs_gap``.

    Omit backend-default options and return an immutable mapping.
    """

    if not isinstance(config, SolverConfig):
        raise TypeError("config must be a SolverConfig")
    if config.backend is not SolverBackend.HIGHS:
        raise UnsupportedSolverBackendError(
            f"HiGHS option translation does not support {config.backend.value!r}"
        )
    options: dict[str, float] = {}
    if config.time_limit_seconds is not None:
        options["time_limit"] = float(config.time_limit_seconds)
    if config.relative_mip_gap is not None:
        options["mip_rel_gap"] = float(config.relative_mip_gap)
    if config.absolute_mip_gap is not None:
        options["mip_abs_gap"] = float(config.absolute_mip_gap)
    return MappingProxyType(options)


def translate_gurobi_options(config: SolverConfig) -> Mapping[str, float]:
    """Map configured options to Gurobi ``TimeLimit/MIPGap/MIPGapAbs``.

    Omit backend-default options and return an immutable mapping.
    """

    if not isinstance(config, SolverConfig):
        raise TypeError("config must be a SolverConfig")
    if config.backend is not SolverBackend.GUROBI:
        raise UnsupportedSolverBackendError(
            f"Gurobi option translation does not support {config.backend.value!r}"
        )
    options: dict[str, float] = {}
    if config.time_limit_seconds is not None:
        options["TimeLimit"] = float(config.time_limit_seconds)
    if config.relative_mip_gap is not None:
        options["MIPGap"] = float(config.relative_mip_gap)
    if config.absolute_mip_gap is not None:
        options["MIPGapAbs"] = float(config.absolute_mip_gap)
    return MappingProxyType(options)


def _finite_optional(value: object) -> float | None:
    """Normalize unavailable or non-finite backend metadata to ``None``."""

    if value is None or isinstance(value, bool) or not isinstance(value, Real):
        return None
    plain_value = float(value)
    return plain_value if isfinite(plain_value) else None


def _normalize_appsi_status(
    termination_condition: TerminationCondition,
    has_incumbent: bool,
) -> SolverStatus:
    """Classify APPSI termination using incumbent presence."""

    if termination_condition is TerminationCondition.optimal:
        return SolverStatus.OPTIMAL if has_incumbent else SolverStatus.ERROR
    if termination_condition is TerminationCondition.infeasible:
        return SolverStatus.INFEASIBLE
    if termination_condition is TerminationCondition.unbounded:
        return SolverStatus.UNBOUNDED
    if termination_condition is TerminationCondition.infeasibleOrUnbounded:
        return SolverStatus.INFEASIBLE_OR_UNBOUNDED
    if termination_condition in (
        TerminationCondition.error,
        TerminationCondition.licensingProblems,
    ):
        return SolverStatus.ERROR
    if has_incumbent:
        # An incumbent after a limit does not establish optimality.
        return SolverStatus.FEASIBLE
    if termination_condition is TerminationCondition.unknown:
        return SolverStatus.UNKNOWN
    return SolverStatus.NO_SOLUTION


def _normalized_relative_gap(
    objective_value: float | None,
    best_bound: float | None,
) -> float | None:
    """Return ``abs(objective-bound)/abs(objective)`` when defined.

    Equal values give zero; missing values or a zero objective with a gap give None.
    """

    if objective_value is None or best_bound is None:
        return None
    absolute_gap = abs(objective_value - best_bound)
    if absolute_gap == 0.0:
        return 0.0
    if objective_value == 0.0:
        return None
    return absolute_gap / abs(objective_value)


def _prepare_solver(
    config: SolverConfig,
) -> tuple[Highs | Gurobi, dict[str, str | None]]:
    """Check availability, configure the selected APPSI solver, and record it."""

    try:
        if config.backend is SolverBackend.HIGHS:
            solver = Highs()
            solver.highs_options.update(translate_highs_options(config))
            interface = "pyomo.contrib.appsi.solvers.Highs"
        else:
            solver = Gurobi()
            solver.gurobi_options.update(translate_gurobi_options(config))
            interface = "pyomo.contrib.appsi.solvers.Gurobi"
        availability = solver.available()
        if not bool(availability):
            raise SolverBackendUnavailableError(
                f"Pyomo APPSI {config.backend.value} availability: {availability.name}"
            )
        version = ".".join(str(component) for component in solver.version())
    except SolverBackendUnavailableError:
        raise
    except Exception as error:
        raise SolverBackendUnavailableError(
            f"Cannot prepare {config.backend.value}: {type(error).__name__}: {error}"
        ) from error
    solver.config.load_solution = False
    solver.config.stream_solver = config.tee
    raw_metadata = {
        "interface": interface,
        "solver_version": version,
        "backend_availability": SolverBackendAvailability.AVAILABLE.value,
    }
    return solver, raw_metadata


def _solver_error_result(
    config: SolverConfig,
    termination_condition: str,
    error: Exception,
    elapsed: float,
    raw_metadata: Mapping[str, str | None],
    best_bound: float | None = None,
) -> SolverResult:
    """Retain exception evidence without exposing an unusable incumbent."""

    return SolverResult(
        backend=config.backend,
        status=SolverStatus.ERROR,
        termination_condition=termination_condition,
        has_incumbent=False,
        objective_value=None,
        best_bound=best_bound,
        relative_gap=None,
        solve_time_seconds=elapsed,
        raw_status_metadata={
            **raw_metadata,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
        },
    )


def _collect_appsi_result(
    appsi_result: Results,
    config: SolverConfig,
    elapsed: float,
    raw_metadata: dict[str, str | None],
) -> SolverResult:
    """Interpret termination evidence and load only a finite-objective incumbent."""

    objective_value = _finite_optional(appsi_result.best_feasible_objective)
    best_bound = _finite_optional(appsi_result.best_objective_bound)
    has_incumbent = objective_value is not None
    status = _normalize_appsi_status(
        appsi_result.termination_condition,
        has_incumbent,
    )
    raw_metadata["pyomo_termination_condition"] = (
        appsi_result.termination_condition.name
    )

    if has_incumbent:
        try:
            appsi_result.solution_loader.load_vars()
        except Exception as error:
            return _solver_error_result(
                config, "incumbent_load_error", error, elapsed,
                raw_metadata, best_bound,
            )

    return SolverResult(
        backend=config.backend,
        status=status,
        termination_condition=appsi_result.termination_condition.name,
        has_incumbent=has_incumbent,
        objective_value=objective_value,
        best_bound=best_bound,
        relative_gap=_normalized_relative_gap(objective_value, best_bound),
        solve_time_seconds=elapsed,
        raw_status_metadata=raw_metadata,
    )


def solve_pyomo_model(model: object, config: SolverConfig) -> SolverResult:
    """Solve a constructed model through the configured APPSI backend.

    Load variables only when a finite best feasible objective establishes an
    incumbent.
    """

    if not isinstance(config, SolverConfig):
        raise TypeError("config must be a SolverConfig")
    solver, raw_metadata = _prepare_solver(config)

    start = perf_counter()
    try:
        appsi_result = solver.solve(model)
    except Exception as error:
        return _solver_error_result(
            config, "execution_error", error, perf_counter() - start, raw_metadata,
        )
    elapsed = perf_counter() - start
    return _collect_appsi_result(appsi_result, config, elapsed, raw_metadata)


def require_solver_incumbent(result: SolverResult) -> SolverResult:
    """Require an incumbent before extraction; otherwise raise."""

    if not isinstance(result, SolverResult):
        raise TypeError("result must be a SolverResult")
    if not result.has_incumbent:
        raise NoSolverIncumbentError(
            "Cannot extract a solution because the solver result has no incumbent: "
            f"status={result.status.value}, termination={result.termination_condition}"
        )
    return result
