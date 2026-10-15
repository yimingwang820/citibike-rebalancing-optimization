"""Explicit names used by migration and historical regression tests."""
from pathlib import Path
import ast
import re

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "preprocessing": ["audit_trip_data", "normalize_station_ids", "select_study_area",
                      "build_inventory_scenarios", "build_weekday_profiles", "extract_trip_events"],
    "simulation": ["event_baseline", "timeout_simulation"],
    "optimization": ["single_period_lp", "fixed_charge_milp", "hourly_balance_milp",
                     "strict_order_milp", "weekday_preposition", "weekend_preposition"],
    "evaluation": ["audit_hourly_feasibility", "service_metrics", "wait_duration_diagnostics",
                   "development_day_effects", "holdout_validation", "weekend_validation"],
    "experiments": ["compare_lp_milp", "fixed_cost_sensitivity"],
    "examples": ["minimal_event_simulation"],
}
MODULES = {name: f"citibike.{group}.{name}" for group, names in GROUPS.items() for name in names}
MODULES["project_paths"] = "citibike.paths"
SCRIPTS = sorted(set(MODULES) - {"project_paths"})

# Exact, reviewed helper-only substitutions accepted by historical AST tests.
IO_EXTRACTIONS = {
    "holdout_validation": ("file_sha256", "path", "from hashlib import sha256", "from citibike.file_io import file_sha256 as _file_sha256", "_file_sha256"),
    "weekend_validation": ("fingerprint", "path", "from hashlib import sha256", "from citibike.file_io import file_sha256 as _file_sha256", "_file_sha256"),
    "weekend_preposition": ("sha256", "path", "import hashlib", "from citibike.file_io import file_sha256 as _file_sha256", "_file_sha256"),
    "fixed_charge_milp": ("save_csv", "path, records", "import csv", "from citibike.file_io import write_records_csv", "write_records_csv"),
    "fixed_cost_sensitivity": ("save_csv", "filepath, rows", "import csv", "from citibike.file_io import write_records_csv", "write_records_csv"),
}


def adapt_shared_io(tree, name):
    """Allow only exact helper delegates, not arbitrary function exceptions."""
    if name not in IO_EXTRACTIONS:
        return tree
    function, args, old_import, new_import, target = IO_EXTRACTIONS[name]
    old_node = ast.parse(old_import).body[0]
    for index, node in enumerate(tree.body):
        if ast.dump(node) == ast.dump(old_node):
            tree.body[index] = ast.parse(new_import).body[0]
        elif isinstance(node, ast.FunctionDef) and node.name == function:
            tree.body[index] = ast.parse(f"def {function}({args}):\n    return {target}({args})").body[0]
    return tree


def module_name(name):
    return MODULES[name]


def module_path(name):
    target = module_name(name)
    return ROOT / (target.replace(".", "/") + ".py")


def rewrite_imports(source):
    """Adapt historical import lines only; leave its computation untouched."""
    for name in MODULES:
        target = module_name(name)
        source = re.sub(r"(?m)^(\s*from )" + name + r"( import )", r"\g<1>" + target + r"\2", source)
        source = re.sub(r"(?m)^(\s*import )" + name + r"( as )", r"\g<1>" + target + r"\2", source)
    return source
