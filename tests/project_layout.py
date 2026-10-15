"""Explicit names used by migration and historical regression tests."""
from pathlib import Path
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
