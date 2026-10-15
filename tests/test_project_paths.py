"""Lightweight path/import checks. Run: python -B -m unittest discover -s tests -v.

No project entry points are called. Real installed dependencies are imported;
Gurobi environment/model creation is blocked. All caches use temporary storage.
"""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "01_CODE"
CHECKPOINT = "4d147977ebb89303718e5e402ad942807257d78d"
SCRIPTS = sorted(p.stem for p in CODE.glob("*.py")
                 if p.stem not in {"project_paths"}
                 and not p.stem.startswith(("lesson", "day7_")))

PROBE = r'''
import importlib, json, os, pathlib, sys
code, scratch = map(pathlib.Path, sys.argv[1:3])
sys.path.insert(0, str(code))
import gurobipy as gp
def forbidden(*args, **kwargs):
    raise AssertionError("Solver/environment construction during import")
gp.Model = gp.Env = forbidden
def audit(event, args):
    if event == "open":
        path, mode, flags = args
        if isinstance(path, (str, bytes, os.PathLike)):
            p = pathlib.Path(os.fsdecode(path)).resolve()
            if any(part in {"00_DATA", "02_RESULTS"} for part in p.parts):
                raise AssertionError("Project data access during import: " + str(p))
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (flags and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing and not p.is_relative_to(scratch):
                raise AssertionError("Write outside scratch: " + str(p))
    if event in {"os.remove", "os.rename", "os.rmdir", "os.mkdir"}:
        for arg in args[:2] if event == "os.rename" else args[:1]:
            if isinstance(arg, (str, bytes, os.PathLike)) and not pathlib.Path(os.fsdecode(arg)).resolve().is_relative_to(scratch):
                raise AssertionError("Mutation outside scratch")
sys.addaudithook(audit)
def profile(frame, event, arg):
    if event == "call" and pathlib.Path(frame.f_code.co_filename).parent == code:
        if frame.f_code.co_name not in {"<module>", "get_project_root"} and not frame.f_code.co_name.startswith("<"):
            # Class bodies execute during import; ordinary functions must not.
            if not frame.f_code.co_flags & 0x02:
                return
            raise AssertionError("Project function called during import: " + frame.f_code.co_name)
sys.setprofile(profile)
result = {}
for name in json.loads(sys.argv[3]):
    module = importlib.import_module(name)
    result[name] = {k: str(v) for k, v in vars(module).items() if isinstance(v, pathlib.Path)}
sys.setprofile(None)
import project_paths
result["project_paths"] = {"ROOT": str(project_paths.ROOT), "explicit": str(project_paths.get_project_root(scratch / "explicit"))}
print(json.dumps(result))
'''


class ProjectPathsTests(unittest.TestCase):
    def probe(self, cwd, scratch, override=None):
        env = os.environ.copy()
        env.pop("CITIBIKE_PROJECT_ROOT", None)
        env.update(PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(scratch / "mpl"),
                   TEMP=str(scratch), TMP=str(scratch), PYTHONIOENCODING="utf-8")
        if override is not None:
            env["CITIBIKE_PROJECT_ROOT"] = str(override)
        run = subprocess.run([sys.executable, "-B", "-c", PROBE, str(CODE),
                              str(scratch), json.dumps(SCRIPTS)], cwd=cwd,
                             env=env, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(run.returncode, 0, run.stderr)
        return json.loads(run.stdout)

    def test_cwd_independence_and_import_safety(self):
        with tempfile.TemporaryDirectory() as temp:
            scratch = Path(temp)
            a = self.probe(ROOT, scratch)
            b = self.probe(scratch, scratch)
            self.assertEqual(a, b)
            self.assertEqual(Path(a["project_paths"]["ROOT"]), ROOT)
            self.assertEqual(Path(a["project_paths"]["explicit"]), scratch / "explicit")

    def test_explicit_root_and_import_bound_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            scratch = Path(temp)
            before = self.probe(ROOT, scratch)
            target = scratch / "alternate_project"
            after = self.probe(scratch, scratch, target)
            checked = 0
            for module, values in before.items():
                for key, value in values.items():
                    if Path(value).is_relative_to(ROOT):
                        self.assertEqual(Path(after[module][key]), target / Path(value).relative_to(ROOT), (module, key))
                        checked += 1
            self.assertGreater(checked, 40)

    def test_syntax_internal_imports_and_unchanged_path_defaults(self):
        self.assertEqual(len(SCRIPTS), 23)
        replacements = 0
        for name in SCRIPTS:
            path = CODE / (name + ".py")
            source = path.read_text(encoding="utf-8-sig")
            tree = ast.parse(source, filename=str(path))
            compile(tree, str(path), "exec")
            old = subprocess.check_output(["git", "show", CHECKPOINT + ":01_CODE/" + path.name], cwd=ROOT).decode("utf-8-sig")
            for assignment in ['ROOT = Path(r"D:\\BikeSharing_OR_Project")', 'ROOT = Path(__file__).resolve().parents[1]']:
                replacements += old.count(assignment)
                old = old.replace(assignment, "from project_paths import ROOT")
            # Path expressions, CLI definitions and module-level defaults remain
            # stable while separately tested date parameters may evolve.
            def configuration(tree):
                nodes = [n for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))]
                nodes += [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
                          ((isinstance(n.func, ast.Name) and n.func.id == "Path") or
                           (isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"))]
                return [ast.dump(n) for n in nodes]
            self.assertEqual(configuration(ast.parse(old)), configuration(tree), name)
            for node in ast.walk(tree):
                names = ([node.module] if isinstance(node, ast.ImportFrom) else
                         [a.name for a in node.names] if isinstance(node, ast.Import) else [])
                for imported in names:
                    self.assertFalse(imported and imported.startswith(tuple("day%d_" % n for n in range(1, 7))))
                    if imported in SCRIPTS or imported == "project_paths":
                        self.assertTrue((CODE / (imported + ".py")).is_file())
            self.assertNotIn('D:\\BikeSharing_OR_Project', source)
        self.assertEqual(replacements, 13)


if __name__ == "__main__":
    unittest.main()
