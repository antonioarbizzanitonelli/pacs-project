"""Solver-independent placement validation using the mathematical primitives."""

from dataclasses import dataclass, field, fields
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from llm_resource_allocation.domain import InstanceValidationError, ProblemInstance, UnitConvention
from llm_resource_allocation.math.budget import (
    BudgetDiagnostic,
    _budget_residual_from_objective,
)
from llm_resource_allocation.math.compute_capacity import (
    ComputeCapacityDiagnostic,
    _compute_compute_capacity_diagnostics,
)
from llm_resource_allocation.math.parallelism import (
    ConfigurationSelectionDiagnostic,
    GpuCountLinkingDiagnostic,
    _compute_configuration_selection_residuals,
    _compute_gpu_count_linking_residuals,
)
from llm_resource_allocation.math.routing import (
    DecisionDomainError,
    DemandBalanceDiagnostic,
    RoutingConsistencyDiagnostic,
    _compute_demand_balance_residuals,
    _compute_routing_consistency_residuals,
)
from llm_resource_allocation.math.processing_delay import (
    DelaySloDiagnostic,
    _delay_slo_residuals_from_processing_delays,
    _compute_processing_delays_seconds,
)
from llm_resource_allocation.math.error_slo import (
    ErrorSloDiagnostic,
    _compute_error_slo_residuals,
)
from llm_resource_allocation.math.gpu_memory import (
    GpuMemoryDiagnostic,
    _compute_gpu_memory_residuals,
)
from llm_resource_allocation.math.placement import (
    McCormickConsistencyDiagnostic,
    PlacementDecisionState,
    _compute_mccormick_consistency_diagnostics,
    validate_placement_decision_domains,
)
from llm_resource_allocation.math.objective import (
    ObjectiveBreakdown,
    ObjectiveState,
    PaperVersion,
    compute_objective_breakdown,
)
from llm_resource_allocation.math.storage_capacity import (
    QueryStorageCapacityDiagnostic,
    _compute_storage_capacity_residuals,
)
from .tolerances import (
    DEFAULT_FRACTION_ABSOLUTE_TOLERANCE,
    DEFAULT_BUDGET_ABSOLUTE_TOLERANCE,
    DEFAULT_COMPUTE_ABSOLUTE_TOLERANCE,
    DEFAULT_DELAY_ABSOLUTE_TOLERANCE,
    DEFAULT_ERROR_ABSOLUTE_TOLERANCE,
    DEFAULT_PRINTED_CAPACITY_ABSOLUTE_TOLERANCE,
    DEFAULT_SCALED_RELATIVE_TOLERANCE,
    _record_applied_tolerance,
)


# D-0076: dimensionless residual classification; supplied decisions stay unchanged.
DEFAULT_MCCORMICK_ABSOLUTE_TOLERANCE = DEFAULT_FRACTION_ABSOLUTE_TOLERANCE


class ConstraintFamily(str, Enum):
    """Identify placement constraint families."""

    DECISION_DOMAINS = "decision_domains"
    MCCORMICK = "mccormick_eq_7"
    DEMAND_BALANCE = "demand_balance_eq_8b"
    BUDGET = "budget_eq_8c"
    CONFIGURATION_SELECTION = "configuration_selection_eq_8d"
    GPU_COUNT_LINKING = "gpu_count_linking_eq_8e"
    GPU_MEMORY = "gpu_memory_eq_8f_d_0002"
    COMPUTE_CAPACITY = "compute_capacity_eq_8g"
    STORAGE_CAPACITY = "storage_capacity_eq_8h_d_0016"
    DELAY_SLO = "delay_slo_eq_8i"
    ERROR_SLO = "error_slo_eq_8j"
    ROUTING_CONSISTENCY = "routing_consistency_eq_8k"


_ToleranceKey = tuple[ConstraintFamily, tuple[str | int, ...]]


@dataclass(frozen=True, slots=True)
class ValidationTolerances:
    """Per-family absolute floors and relative coefficients; domains stay exact.

    Scaled rows use ``atol + rtol*max(abs(lhs), abs(rhs))``. Use ``exact()``
    to disable both terms; setting an absolute floor to zero alone keeps rtol.
    """

    mccormick: float = DEFAULT_MCCORMICK_ABSOLUTE_TOLERANCE
    demand_balance: float = DEFAULT_FRACTION_ABSOLUTE_TOLERANCE
    budget: float = DEFAULT_BUDGET_ABSOLUTE_TOLERANCE
    configuration_selection: float = 0.0
    gpu_count_linking: float = 0.0
    gpu_memory: float = DEFAULT_PRINTED_CAPACITY_ABSOLUTE_TOLERANCE
    compute_capacity: float = DEFAULT_COMPUTE_ABSOLUTE_TOLERANCE
    storage_capacity: float = DEFAULT_PRINTED_CAPACITY_ABSOLUTE_TOLERANCE
    delay_slo: float = DEFAULT_DELAY_ABSOLUTE_TOLERANCE
    error_slo: float = DEFAULT_ERROR_ABSOLUTE_TOLERANCE
    routing_consistency: float = DEFAULT_FRACTION_ABSOLUTE_TOLERANCE
    budget_relative: float = DEFAULT_SCALED_RELATIVE_TOLERANCE
    gpu_memory_relative: float = DEFAULT_SCALED_RELATIVE_TOLERANCE
    compute_capacity_relative: float = DEFAULT_SCALED_RELATIVE_TOLERANCE
    storage_capacity_relative: float = DEFAULT_SCALED_RELATIVE_TOLERANCE
    delay_slo_relative: float = DEFAULT_SCALED_RELATIVE_TOLERANCE
    error_slo_relative: float = DEFAULT_SCALED_RELATIVE_TOLERANCE

    @classmethod
    def exact(cls) -> "ValidationTolerances":
        """Request exact classification of every residual, including scaled rows."""

        return cls(**{descriptor.name: 0.0 for descriptor in fields(cls)})

    def __post_init__(self) -> None:
        """Reject non-finite, negative, boolean, or non-numeric tolerances."""

        for field in fields(self):
            value = getattr(self, field.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0.0
            ):
                raise ValueError(
                    f"{field.name} tolerance must be a finite non-negative real number"
                )


@dataclass(frozen=True, slots=True)
class PlacementDomainDiagnostic:
    """Record input/decision domain validity and any error."""

    is_valid: bool
    error_type: str | None = None
    message: str | None = None

@dataclass(frozen=True, slots=True)
class ValidationViolation:
    """Locate one violation, with slack in the family's native unit.

    Domain errors use ``None`` because they have no numerical magnitude.
    """

    family: ConstraintFamily
    semantic_location: tuple[str | int, ...]
    slack: float | None

    def __post_init__(self) -> None:
        """Own an immutable copy of the semantic location."""

        object.__setattr__(self, "semantic_location", tuple(self.semantic_location))

    @property
    def raw_violation(self) -> float | None:
        return None if self.slack is None else -self.slack


@dataclass(frozen=True, slots=True)
class PlacementValidationReport:
    """Store objective and constraint evidence for v2 Eqs. (7)-(8k).

    Retains the input unit policy; D-0101 supplies coherent units and
    D-0002/D-0016 remain the literal historical policy. Invalid domains
    leave the objective and mathematical diagnostics empty.
    """

    paper_version: PaperVersion
    unit_convention: UnitConvention
    data_residency_hours: float
    tolerances: ValidationTolerances
    domain_diagnostic: PlacementDomainDiagnostic
    processing_delays_seconds: Mapping[str, float]
    objective_breakdown: ObjectiveBreakdown | None
    mccormick_diagnostics: tuple[McCormickConsistencyDiagnostic, ...]
    demand_balance_diagnostics: tuple[DemandBalanceDiagnostic, ...]
    budget_diagnostic: BudgetDiagnostic | None
    configuration_selection_diagnostics: tuple[ConfigurationSelectionDiagnostic, ...]
    gpu_count_linking_diagnostics: tuple[GpuCountLinkingDiagnostic, ...]
    gpu_memory_diagnostics: tuple[GpuMemoryDiagnostic, ...]
    compute_capacity_diagnostics: tuple[ComputeCapacityDiagnostic, ...]
    storage_capacity_diagnostics: tuple[QueryStorageCapacityDiagnostic, ...]
    delay_slo_diagnostics: tuple[DelaySloDiagnostic, ...]
    error_slo_diagnostics: tuple[ErrorSloDiagnostic, ...]
    routing_consistency_diagnostics: tuple[RoutingConsistencyDiagnostic, ...]
    _violations: tuple[ValidationViolation, ...] = field(
        init=False, repr=False, compare=False
    )
    applied_tolerances: Mapping[_ToleranceKey, float] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Own immutable copies of the mapping and diagnostic sequences."""

        object.__setattr__(
            self,
            "processing_delays_seconds",
            MappingProxyType(dict(self.processing_delays_seconds)),
        )
        for name in (
            "mccormick_diagnostics",
            "demand_balance_diagnostics",
            "configuration_selection_diagnostics",
            "gpu_count_linking_diagnostics",
            "gpu_memory_diagnostics",
            "compute_capacity_diagnostics",
            "storage_capacity_diagnostics",
            "delay_slo_diagnostics",
            "error_slo_diagnostics",
            "routing_consistency_diagnostics",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        thresholds: dict[_ToleranceKey, float] = {}
        object.__setattr__(self, "_violations", self._collect_violations(thresholds))
        object.__setattr__(self, "applied_tolerances", MappingProxyType(thresholds))

    @property
    def violations(self) -> tuple[ValidationViolation, ...]:
        """Return violations derived once in deterministic constraint-family order."""

        return self._violations

    def _collect_violations(
        self, thresholds: dict[_ToleranceKey, float]
    ) -> tuple[ValidationViolation, ...]:
        """Derive violation locations from the report's immutable evidence."""

        if not self.domain_diagnostic.is_valid:
            return (
                ValidationViolation(
                    family=ConstraintFamily.DECISION_DOMAINS,
                    semantic_location=(self.domain_diagnostic.error_type,),
                    slack=None,
                ),
            )

        violations = self._mccormick_violations(thresholds)
        violations.extend(self._demand_balance_violations(thresholds))
        violations.extend(self._budget_violations(thresholds))
        violations.extend(self._configuration_violations(thresholds))
        violations.extend(self._capacity_and_slo_violations(thresholds))
        violations.extend(self._routing_consistency_violations(thresholds))
        return tuple(violations)

    def _mccormick_violations(
        self, thresholds: dict[_ToleranceKey, float]
    ) -> list[ValidationViolation]:
        """Locate each failed McCormick bound at its full configuration index."""

        violations: list[ValidationViolation] = []
        for diagnostic in self.mccormick_diagnostics:
            slacks = diagnostic.residuals
            location = (
                diagnostic.query_id,
                diagnostic.model_id,
                diagnostic.gpu_tier_id,
                diagnostic.tensor_parallel_degree,
                diagnostic.pipeline_parallel_depth,
            )
            named_slacks = (
                ("x_minus_v", slacks.x_minus_v),
                ("w_minus_v", slacks.w_minus_v),
                ("v_minus_x_minus_w_plus_one", slacks.v_minus_x_minus_w_plus_one),
                ("v_nonnegative", slacks.v_nonnegative),
            )
            for inequality_name, slack in named_slacks:
                tolerance = _record_applied_tolerance(
                    thresholds, ConstraintFamily.MCCORMICK,
                    (*location, inequality_name), self.tolerances.mccormick,
                )
                if slack < -tolerance:
                    violations.append(
                        ValidationViolation(
                            ConstraintFamily.MCCORMICK,
                            (*location, inequality_name),
                            slack,
                        )
                    )
        return violations

    def _demand_balance_violations(
        self, thresholds: dict[_ToleranceKey, float]
    ) -> list[ValidationViolation]:
        """Locate failed demand-balance equalities by query."""

        violations: list[ValidationViolation] = []
        for diagnostic in self.demand_balance_diagnostics:
            tolerance = _record_applied_tolerance(
                thresholds, ConstraintFamily.DEMAND_BALANCE,
                (diagnostic.query_id,), self.tolerances.demand_balance,
            )
            if not diagnostic.is_feasible(tolerance):
                violations.append(
                    ValidationViolation(
                        ConstraintFamily.DEMAND_BALANCE,
                        (diagnostic.query_id,),
                        -diagnostic.absolute_violation,
                    )
                )
        return violations

    def _budget_violations(
        self, thresholds: dict[_ToleranceKey, float]
    ) -> list[ValidationViolation]:
        """Report the global budget inequality when present and violated."""

        diagnostic = self.budget_diagnostic
        if diagnostic is None:
            return []
        tolerance = _record_applied_tolerance(
            thresholds, ConstraintFamily.BUDGET, ("global",),
            self.tolerances.budget, self.tolerances.budget_relative,
            diagnostic.total_budget_use, diagnostic.budget_limit,
        )
        if diagnostic.is_feasible(tolerance):
            return []
        return [
            ValidationViolation(
                ConstraintFamily.BUDGET,
                ("global",),
                diagnostic.slack,
            )
        ]

    def _configuration_violations(
        self, thresholds: dict[_ToleranceKey, float]
    ) -> list[ValidationViolation]:
        """Locate selection and GPU-count equalities by model/GPU pair."""

        violations: list[ValidationViolation] = []
        equality_groups = (
            (
                ConstraintFamily.CONFIGURATION_SELECTION,
                self.configuration_selection_diagnostics,
                self.tolerances.configuration_selection,
            ),
            (
                ConstraintFamily.GPU_COUNT_LINKING,
                self.gpu_count_linking_diagnostics,
                self.tolerances.gpu_count_linking,
            ),
        )
        for family, diagnostics, tolerance in equality_groups:
            for diagnostic in diagnostics:
                _record_applied_tolerance(
                    thresholds, family, (diagnostic.model_id, diagnostic.gpu_tier_id),
                    tolerance,
                )
                if not diagnostic.is_feasible(tolerance):
                    violations.append(
                        ValidationViolation(
                            family,
                            (diagnostic.model_id, diagnostic.gpu_tier_id),
                            -diagnostic.absolute_violation,
                        )
                    )
        return violations

    def _capacity_and_slo_violations(
        self, thresholds: dict[_ToleranceKey, float]
    ) -> list[ValidationViolation]:
        """Locate capacity and SLO inequalities by model/GPU pair or query."""

        violations: list[ValidationViolation] = []
        pair_inequality_groups = (
            (
                ConstraintFamily.GPU_MEMORY,
                self.gpu_memory_diagnostics,
                self.tolerances.gpu_memory,
                self.tolerances.gpu_memory_relative,
                "printed_memory_lhs", "available_memory_rhs_gb",
            ),
            (
                ConstraintFamily.COMPUTE_CAPACITY,
                self.compute_capacity_diagnostics,
                self.tolerances.compute_capacity,
                self.tolerances.compute_capacity_relative,
                "required_compute", "available_compute",
            ),
        )
        for (
            family, diagnostics, absolute, relative, lhs_name, rhs_name
        ) in pair_inequality_groups:
            for diagnostic in diagnostics:
                tolerance = _record_applied_tolerance(
                    thresholds, family, (diagnostic.model_id, diagnostic.gpu_tier_id),
                    absolute, relative, getattr(diagnostic, lhs_name),
                    getattr(diagnostic, rhs_name),
                )
                if not diagnostic.is_feasible(tolerance):
                    violations.append(
                        ValidationViolation(
                            family,
                            (diagnostic.model_id, diagnostic.gpu_tier_id),
                            diagnostic.slack,
                        )
                    )

        query_inequality_groups = (
            (
                ConstraintFamily.STORAGE_CAPACITY,
                self.storage_capacity_diagnostics,
                self.tolerances.storage_capacity,
                self.tolerances.storage_capacity_relative,
                "printed_storage_lhs", "storage_limit_gb",
            ),
            (
                ConstraintFamily.DELAY_SLO,
                self.delay_slo_diagnostics,
                self.tolerances.delay_slo,
                self.tolerances.delay_slo_relative,
                "processing_delay", "delay_limit",
            ),
            (
                ConstraintFamily.ERROR_SLO,
                self.error_slo_diagnostics,
                self.tolerances.error_slo,
                self.tolerances.error_slo_relative,
                "weighted_error", "error_limit",
            ),
        )
        for (
            family, diagnostics, absolute, relative, lhs_name, rhs_name
        ) in query_inequality_groups:
            for diagnostic in diagnostics:
                tolerance = _record_applied_tolerance(
                    thresholds, family, (diagnostic.query_id,), absolute, relative,
                    getattr(diagnostic, lhs_name), getattr(diagnostic, rhs_name),
                )
                if not diagnostic.is_feasible(tolerance):
                    violations.append(
                        ValidationViolation(
                            family,
                            (diagnostic.query_id,),
                            diagnostic.slack,
                        )
                    )
        return violations

    def _routing_consistency_violations(
        self, thresholds: dict[_ToleranceKey, float]
    ) -> list[ValidationViolation]:
        """Locate each failed routing bound at its query/model/GPU index."""

        violations: list[ValidationViolation] = []
        for diagnostic in self.routing_consistency_diagnostics:
            location = (diagnostic.query_id, diagnostic.model_id, diagnostic.gpu_tier_id)
            named_slacks = (
                ("routing_nonnegative_slack", diagnostic.routing_nonnegative_slack),
                ("admission_slack", diagnostic.admission_slack),
                ("deployment_slack", diagnostic.deployment_slack),
            )
            for inequality_name, slack in named_slacks:
                tolerance = _record_applied_tolerance(
                    thresholds, ConstraintFamily.ROUTING_CONSISTENCY,
                    (*location, inequality_name), self.tolerances.routing_consistency,
                )
                if slack < -tolerance:
                    violations.append(
                        ValidationViolation(
                            ConstraintFamily.ROUTING_CONSISTENCY,
                            (*location, inequality_name),
                            slack,
                        )
                    )

        return violations

    @property
    def is_feasible(self) -> bool:
        """Return true only when domains and every reported family pass."""

        return self.domain_diagnostic.is_valid and not self.violations

    @property
    def violation_count(self) -> int:
        """Return the number of independently located violations."""

        return len(self.violations)

    @property
    def violated_families(self) -> tuple[ConstraintFamily, ...]:
        """Return violated families once each in deterministic report order."""

        return tuple(dict.fromkeys(item.family for item in self.violations))

    @property
    def max_violation_by_family(self) -> Mapping[ConstraintFamily, float]:
        """Return native-unit maxima within families, never across families."""

        maxima: dict[ConstraintFamily, float] = {}
        for violation in self.violations:
            if violation.raw_violation is not None:
                maxima[violation.family] = max(
                    maxima.get(violation.family, 0.0), violation.raw_violation
                )
        return MappingProxyType(maxima)


def _empty_report(
    instance: ProblemInstance,
    paper_version: PaperVersion,
    tolerances: ValidationTolerances,
    domain_diagnostic: PlacementDomainDiagnostic,
) -> PlacementValidationReport:
    """Return a domain-invalid report with no mathematical results."""

    return PlacementValidationReport(
        paper_version=paper_version,
        unit_convention=instance.unit_convention,
        data_residency_hours=instance.data_residency_hours,
        tolerances=tolerances,
        domain_diagnostic=domain_diagnostic,
        processing_delays_seconds={},
        objective_breakdown=None,
        mccormick_diagnostics=(),
        demand_balance_diagnostics=(),
        budget_diagnostic=None,
        configuration_selection_diagnostics=(),
        gpu_count_linking_diagnostics=(),
        gpu_memory_diagnostics=(),
        compute_capacity_diagnostics=(),
        storage_capacity_diagnostics=(),
        delay_slo_diagnostics=(),
        error_slo_diagnostics=(),
        routing_consistency_diagnostics=(),
    )


@dataclass(frozen=True, slots=True)
class PlacementValidator:
    """Validate supplied placements with one immutable configuration."""

    instance: ProblemInstance
    paper_version: PaperVersion
    tolerances: ValidationTolerances = field(default_factory=ValidationTolerances)

    def __post_init__(self) -> None:
        """Reject invalid API configuration before scientific validation."""

        if not isinstance(self.paper_version, PaperVersion):
            raise ValueError("paper_version must be PaperVersion.V1 or PaperVersion.V2")
        if not isinstance(self.tolerances, ValidationTolerances):
            raise TypeError("tolerances must be a ValidationTolerances")
        if self.instance.uses_coherent_units and self.paper_version is PaperVersion.V1:
            raise ValueError("coherent units require PaperVersion.V2; use LITERAL_PAPER for historical v1")

    def validate(self, state: PlacementDecisionState) -> PlacementValidationReport:
        """Check domains, then recompute objective and constraint diagnostics.

        Invalid domains return an empty report; all constraint failures remain
        visible. Objective: v2 Eq. (8a) / v1 Eq. (5a).
        """

        instance = self.instance
        paper_version = self.paper_version
        selected_tolerances = self.tolerances

        try:
            validate_placement_decision_domains(instance, state)
        except (InstanceValidationError, DecisionDomainError) as error:
            return _empty_report(
                instance,
                paper_version,
                selected_tolerances,
                PlacementDomainDiagnostic(
                    is_valid=False,
                    error_type=type(error).__name__,
                    message=str(error),
                ),
            )

        # Derive D_proc from supplied v (v2 Eq. 6), without replacing it with x*w.
        processing_delays = _compute_processing_delays_seconds(instance, state)
        objective = compute_objective_breakdown(
            instance,
            ObjectiveState(
                gpu_counts=state.gpu_counts,
                admission_indicators=state.admission_indicators,
                routing_fractions=state.routing_fractions,
                unmet_fractions=state.unmet_fractions,
                processing_delays_seconds=processing_delays,
            ),
            paper_version,
        )

        return PlacementValidationReport(
            paper_version=paper_version,
            unit_convention=instance.unit_convention,
            data_residency_hours=instance.data_residency_hours,
            tolerances=selected_tolerances,
            domain_diagnostic=PlacementDomainDiagnostic(is_valid=True),
            processing_delays_seconds=processing_delays,
            objective_breakdown=objective,
            mccormick_diagnostics=_compute_mccormick_consistency_diagnostics(
                instance, state
            ),
            demand_balance_diagnostics=_compute_demand_balance_residuals(
                instance, state.routing_decisions
            ),
            budget_diagnostic=_budget_residual_from_objective(instance, objective),
            configuration_selection_diagnostics=(
                _compute_configuration_selection_residuals(
                    instance, state.parallelism_decisions
                )
            ),
            gpu_count_linking_diagnostics=_compute_gpu_count_linking_residuals(
                instance, state.parallelism_decisions
            ),
            gpu_memory_diagnostics=_compute_gpu_memory_residuals(instance, state),
            compute_capacity_diagnostics=_compute_compute_capacity_diagnostics(
                instance, state.parallelism_decisions
            ),
            storage_capacity_diagnostics=_compute_storage_capacity_residuals(
                instance, state
            ),
            delay_slo_diagnostics=_delay_slo_residuals_from_processing_delays(
                instance, processing_delays
            ),
            error_slo_diagnostics=_compute_error_slo_residuals(
                instance, state.routing_decisions
            ),
            routing_consistency_diagnostics=_compute_routing_consistency_residuals(
                instance, state.routing_decisions
            ),
        )


def validate_placement(
    instance: ProblemInstance,
    state: PlacementDecisionState,
    paper_version: PaperVersion,
    tolerances: ValidationTolerances | None = None,
) -> PlacementValidationReport:
    """Validate one placement through the configured independent authority."""

    selected_tolerances = ValidationTolerances() if tolerances is None else tolerances
    return PlacementValidator(instance, paper_version, selected_tolerances).validate(state)
