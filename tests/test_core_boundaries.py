"""Shared architectural contracts for the published scientific core."""

import ast
from pathlib import Path
import subprocess
import sys
import unittest


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"


class CoreBoundaryTests(unittest.TestCase):
    def run_fresh(self, code):
        result = subprocess.run(
            [sys.executable, "-I", "-c",
             f"import sys; sys.path.insert(0, {str(SOURCE_ROOT)!r});\n{code}"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_gh_and_oracle_import_without_solver_backends(self):
        for module in ("algorithms.gh", "verification.tiny_enumeration",
                       "validation.placement"):
            with self.subTest(module=module):
                self.run_fresh(
                    f"import llm_resource_allocation.{module}\n"
                    "assert not any(name.startswith(('pyomo', 'highspy', 'gurobipy')) "
                    "for name in sys.modules)\n"
                )

    def test_package_initializers_do_not_eagerly_load_submodules(self):
        for package in ("math", "optimization", "validation"):
            with self.subTest(package=package):
                self.run_fresh(
                    f"import llm_resource_allocation.{package}\n"
                    f"assert not any(name.startswith('llm_resource_allocation.{package}.') "
                    "for name in sys.modules)\n"
                )

    def test_algorithm_exports_match_their_defining_objects(self):
        self.run_fresh(
            "import importlib\n"
            "from llm_resource_allocation import algorithms\n"
            "for name in algorithms.__all__:\n"
            "    value = getattr(algorithms, name)\n"
            "    if not getattr(value, '__module__', '').startswith('llm_resource_allocation.'):\n"
            "        continue  # Type aliases have no project defining-module object.\n"
            "    owner = importlib.import_module(value.__module__)\n"
            "    assert getattr(owner, name) is value, name\n"
        )

    def test_mechanisms_and_oracle_do_not_import_execution_layers(self):
        modules = ("algorithms/m1.py", "algorithms/m2.py", "algorithms/m3.py",
                   "algorithms/phase1.py", "verification/tiny_enumeration.py")
        forbidden = ("pyomo", "highspy", "gurobipy",
                     "llm_resource_allocation.optimization")
        for module in modules:
            with self.subTest(module=module):
                path = SOURCE_ROOT / "llm_resource_allocation" / module
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        imports = [node.module or ""]
                    else:
                        continue
                    for imported in imports:
                        self.assertFalse(imported.startswith(forbidden), imported)


if __name__ == "__main__":
    unittest.main()
