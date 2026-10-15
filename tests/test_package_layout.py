"""Package migration invariants and guarded argparse help checks."""
import ast
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from project_layout import ROOT, SCRIPTS, module_name, module_path, rewrite_imports, adapt_shared_io
from test_project_paths import PROBE

CHECKPOINT = "7f571153a14b78791294d18c6abb74c70645df38"
HELP_SCRIPTS = ("audit_trip_data", "normalize_station_ids", "select_study_area",
                "build_inventory_scenarios", "single_period_lp", "compare_lp_milp")


class PackageLayoutTests(unittest.TestCase):
    def test_formal_document_links_and_module_commands(self):
        for relative in ("README.md", "03_SHOWCASE/REPRODUCIBILITY.md"):
            path = ROOT / relative
            source = path.read_text(encoding="utf-8")
            for module in re.findall(r"python -m (citibike\.[a-z_]+\.[a-z_]+)", source):
                self.assertTrue((ROOT / (module.replace(".", "/") + ".py")).is_file(), module)
            for target in re.findall(r"\]\(([^)]+)\)", source):
                if "://" not in target and not target.startswith("#"):
                    self.assertTrue((path.parent / target.split("#")[0]).is_file(), target)
            self.assertNotRegex(source, r"python 01_CODE/\w+\.py")

    def test_whole_program_ast_matches_reviewed_refactors(self):
        for name in SCRIPTS + ["project_paths"]:
            old = subprocess.check_output(["git", "show", CHECKPOINT + ":01_CODE/" + name + ".py"], cwd=ROOT).decode("utf-8-sig")
            new = module_path(name).read_text(encoding="utf-8-sig")
            tree = ast.parse(new)
            compile(tree, str(module_path(name)), "exec")
            expected = adapt_shared_io(ast.parse(rewrite_imports(old)), name)
            # Only module-level usage examples may change with the invocation.
            # Function docstrings and every executable statement still compare.
            for parsed in (expected, tree):
                if parsed.body and isinstance(parsed.body[0], ast.Expr) and isinstance(parsed.body[0].value, ast.Constant) and isinstance(parsed.body[0].value.value, str):
                    parsed.body.pop(0)
            self.assertEqual(ast.dump(expected), ast.dump(tree), name)

    def test_internal_imports_resolve_to_canonical_modules(self):
        for name in SCRIPTS:
            for node in ast.walk(ast.parse(module_path(name).read_text(encoding="utf-8-sig"))):
                imports = ([node.module] if isinstance(node, ast.ImportFrom) else
                           [a.name for a in node.names] if isinstance(node, ast.Import) else [])
                for target in imports:
                    if target in SCRIPTS + ["project_paths"]:
                        self.assertEqual(module_name(target), target, "Bare import of migrated module")
                    if target and target.startswith("citibike."):
                        self.assertTrue((ROOT / (target.replace(".", "/") + ".py")).is_file(), target)
        for path in (ROOT / "citibike").rglob("__init__.py"):
            # Package initializers must not eagerly load runnable modules.
            tree = ast.parse(path.read_text(encoding="utf-8"))
            self.assertTrue(all(isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) for n in tree.body))

    def test_guarded_package_cli_help(self):
        # Only six previously inspected argparse entry points. No --help for
        # scripts that ignore arguments and would execute experiments.
        for name in HELP_SCRIPTS:
            qualified = module_name(name)
            with tempfile.TemporaryDirectory() as tmp:
                scratch = Path(tmp)
                env = os.environ.copy()
                env.update(PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(scratch / "mpl"),
                           CITIBIKE_PROJECT_ROOT=str(scratch), TEMP=tmp, TMP=tmp,
                           PYTHONIOENCODING="utf-8")
                # The same import guard also rejects project data reads/writes,
                # optimizer construction and any project function except main.
                code = PROBE.split("result = {}", 1)[0].replace('"<module>", "get_project_root"', '"<module>", "get_project_root", "main"')
                code += '\nimport runpy\nsys.argv = [' + repr(qualified) + ', "--help"]\nrunpy.run_module(' + repr(qualified) + ', run_name="__main__")\n'
                result = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT), tmp],
                                        cwd=scratch, env=env, capture_output=True, text=True, encoding="utf-8")
                self.assertEqual(result.returncode, 0, (qualified, result.stderr))
                self.assertIn("--help", result.stdout)
                self.assertNotIn("RuntimeWarning", result.stderr)
                # After the guarded invocation proves --help exits before any
                # business function, verify the actual public -m command too.
                env["PYTHONPATH"] = str(ROOT)
                for cwd in (ROOT, scratch):
                    direct = subprocess.run([sys.executable, "-B", "-m", qualified, "--help"],
                                            cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8")
                    self.assertEqual(direct.returncode, 0, (qualified, direct.stderr))
                    self.assertIn("--help", direct.stdout)
                    self.assertNotIn("RuntimeWarning", direct.stderr)


if __name__ == "__main__":
    unittest.main()
