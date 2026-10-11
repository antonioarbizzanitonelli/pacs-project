"""Run a synthetic coherent-units placement through each Stage-1 entry point."""

from llm_resource_allocation.algorithms.agh import run_agh
from llm_resource_allocation.algorithms.gh import run_gh
from llm_resource_allocation.algorithms.gh_phase1 import GHPhase1Config
from llm_resource_allocation.data.tiny_instances import make_coherent_tiny_instance
from llm_resource_allocation.optimization.pdm_run import execute_and_record_pdm_run
from llm_resource_allocation.optimization.problem_data import PdmProblemData
from llm_resource_allocation.optimization.solver import SolverBackend, SolverConfig


def main() -> None:
    instance = make_coherent_tiny_instance()
    phase1 = GHPhase1Config(budget_fraction=0.8)
    gh = run_gh(instance, phase1)
    agh = run_agh(instance, phase1, random_seed=37)
    exact = execute_and_record_pdm_run(
        PdmProblemData(instance), SolverConfig(backend=SolverBackend.HIGHS)
    )
    print("Unit convention:", instance.unit_convention.value)
    print("GH completed / feasible:", gh.algorithm_completed, gh.independently_feasible)
    print("AGH completed:", agh.completed, "starts:", agh.executed_start_count)
    print("P_DM accepted / solver status:", exact.accepted_run,
          exact.solver_evidence.result.status.value)
    if not (gh.algorithm_completed and gh.independently_feasible
            and agh.completed and exact.accepted_run):
        raise RuntimeError("the synthetic example did not pass independent acceptance")


if __name__ == "__main__":
    main()
