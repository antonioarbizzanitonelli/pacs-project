"""Verify the published API and scientific layers in fresh processes."""

from pathlib import Path
import subprocess
import sys
import unittest


class Stage1ImportTests(unittest.TestCase):
    def test_scientific_imports_do_not_load_solvers(self):
        source = Path(__file__).resolve().parents[1] / "src"
        result = subprocess.run(
            [sys.executable, "-I", "-c",
             f"import sys; sys.path.insert(0, {str(source)!r}); "
             "from llm_resource_allocation.domain import UnitConvention; "
             "from llm_resource_allocation.algorithms import run_gh, run_agh; "
             "from llm_resource_allocation.data import make_coherent_tiny_instance; "
             "from llm_resource_allocation.validation.placement import PlacementValidator; "
             "assert make_coherent_tiny_instance().unit_convention is UnitConvention.COHERENT; "
             "assert not any(n.startswith(('pyomo', 'highspy', 'gurobipy')) for n in sys.modules)"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
