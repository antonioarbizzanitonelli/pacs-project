"""Build, solve, normalize, and independently validate one ``P_DM`` run.

Records retain plain immutable evidence, including objective agreement.
"""

from dataclasses import dataclass

from llm_resource_allocation.math.objective import (
    ObjectiveBreakdown,
    PaperVersion,
)
from llm_resource_allocation.math.placement import PlacementDecisionState
from llm_resource_allocation.validation.objective_comparison import (
    ObjectiveComparison,
)
from llm_resource_allocation.validation.placement import (
    PlacementValidationReport,
    ValidationTolerances,
    validate_placement,
)

from .extraction import (
    SolutionNormalizationTolerances,
    extract_raw_pdm_solution_from_pyomo,
    normalize_raw_pdm_solution,
)
from .problem_data import (
    PdmConstraintCounts,
    PdmProblemData,
    PdmVariableCounts,
)
from .pyomo_pdm import build_pdm_pyomo_model
from .solver import (
    SolverConfig,
    SolverResult,
    require_solver_incumbent,
    solve_pyomo_model,
)


@dataclass(frozen=True, slots=True)
class PdmRunIdentity:
    """Snapshot instance/paper identity, ordered IDs, and unpresolved counts."""

    instance_id: str
    paper_version: PaperVersion
    query_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    tier_ids: tuple[str, ...]
    variable_counts: PdmVariableCounts
    constraint_counts: PdmConstraintCounts

    def __post_init__(self) -> None:
        """Freeze ordered identity sequences already owned by problem data."""

        for field_name in ("query_ids", "model_ids", "tier_ids"):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))


@dataclass(frozen=True, slots=True)
class PdmScientificEvidence:
    """Store independent validation results and normalization tolerances.

    Retain C1-C5 costs computed from problem data and normalized decisions,
    together with their total's comparison to the solver objective.
    """

    validation_report: PlacementValidationReport
    objective_comparison: ObjectiveComparison
    normalization_tolerances: SolutionNormalizationTolerances

    @property
    def objective_breakdown(self) -> ObjectiveBreakdown | None:
        """Return the independently recomputed costs from the retained report."""

        return self.validation_report.objective_breakdown

@dataclass(frozen=True, slots=True)
class PdmSolverEvidence:
    """Store solver configuration and result metadata together."""

    config: SolverConfig
    result: SolverResult

@dataclass(frozen=True, slots=True)
class PdmRunRecord:
    """Store a completed ``P_DM`` incumbent with separate scientific evidence.

    Record construction follows successful normalization. Acceptance also
    requires independent feasibility and objective agreement.
    """

    identity: PdmRunIdentity
    solution: PlacementDecisionState
    scientific_evidence: PdmScientificEvidence
    solver_evidence: PdmSolverEvidence

    @property
    def has_incumbent(self) -> bool:
        """Return incumbent presence from the solver result."""

        return self.solver_evidence.result.has_incumbent

    @property
    def independently_feasible(self) -> bool:
        """Return the independent placement-feasibility classification."""

        return self.scientific_evidence.validation_report.is_feasible

    @property
    def objective_agrees(self) -> bool:
        """Return solver/independent objective agreement."""

        return self.scientific_evidence.objective_comparison.agrees

    @property
    def accepted_run(self) -> bool:
        """Require incumbent, independent feasibility, and objective agreement.

        Successful normalization precedes record construction.
        """

        return (
            self.has_incumbent
            and self.independently_feasible
            and self.objective_agrees
        )


def execute_and_record_pdm_run(
    problem_data: PdmProblemData,
    solver_config: SolverConfig,
    paper_version: PaperVersion = PaperVersion.V2,
    normalization_tolerances: SolutionNormalizationTolerances | None = None,
    *,
    validation_tolerances: ValidationTolerances | None = None,
) -> PdmRunRecord:
    """Execute the ``P_DM`` pipeline and snapshot decisions and evidence.

    No incumbent raises ``NoSolverIncumbentError`` before extraction.
    Validation failures and objective discrepancies remain in rejected records.
    ``validation_tolerances`` controls only independent residual acceptance;
    ``ValidationTolerances.exact()`` disables absolute and relative terms.
    """

    if not isinstance(problem_data, PdmProblemData):
        raise TypeError("problem_data must be PdmProblemData")
    if not isinstance(solver_config, SolverConfig):
        raise TypeError("solver_config must be SolverConfig")
    if not isinstance(paper_version, PaperVersion):
        raise ValueError("paper_version must be PaperVersion.V1 or V2")
    selected_validation = (
        ValidationTolerances() if validation_tolerances is None else validation_tolerances
    )
    if not isinstance(selected_validation, ValidationTolerances):
        raise TypeError("validation_tolerances must be ValidationTolerances or None")
    selected_tolerances = (
        normalization_tolerances or SolutionNormalizationTolerances()
    )
    if not isinstance(selected_tolerances, SolutionNormalizationTolerances):
        raise TypeError(
            "normalization_tolerances must be SolutionNormalizationTolerances or None"
        )

    model = build_pdm_pyomo_model(problem_data, paper_version)
    solver_result = solve_pyomo_model(model, solver_config)
    require_solver_incumbent(solver_result)
    raw_solution = extract_raw_pdm_solution_from_pyomo(model, problem_data)
    solution = normalize_raw_pdm_solution(
        raw_solution,
        problem_data,
        selected_tolerances,
    )
    validation_report = validate_placement(
        problem_data.instance,
        solution,
        paper_version,
        selected_validation,
    )
    objective_breakdown = validation_report.objective_breakdown
    if objective_breakdown is None:
        raise RuntimeError(
            "normalized solution produced no independent objective breakdown"
        )
    # Guard the optional value for static typing and manually built results.
    if solver_result.objective_value is None:
        raise RuntimeError("incumbent-bearing solver result has no objective")
    objective_comparison = ObjectiveComparison(
        solver_result.objective_value,
        objective_breakdown.total,
    )

    indices = problem_data.index_sets
    return PdmRunRecord(
        identity=PdmRunIdentity(
            instance_id=problem_data.instance.id,
            paper_version=paper_version,
            query_ids=indices.query_ids,
            model_ids=indices.model_ids,
            tier_ids=indices.tier_ids,
            variable_counts=problem_data.variable_counts,
            constraint_counts=problem_data.constraint_counts,
        ),
        solution=solution,
        scientific_evidence=PdmScientificEvidence(
            validation_report=validation_report,
            objective_comparison=objective_comparison,
            normalization_tolerances=selected_tolerances,
        ),
        solver_evidence=PdmSolverEvidence(
            config=solver_config,
            result=solver_result,
        ),
    )
