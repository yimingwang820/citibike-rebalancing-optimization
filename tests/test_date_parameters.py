"""Synthetic date regressions against the immutable stage 2A checkpoint.

No formal main, optimizer, real trip data or frozen results are used.
"""
import ast
import contextlib
import copy
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from project_layout import ROOT, SCRIPTS, module_name, module_path, rewrite_imports
CODE = ROOT
CHECKPOINT = "7d1a0d3af1f149039517b7600de5095c6ea9365c"


def original(name):
    return subprocess.check_output(["git", "show", CHECKPOINT + ":01_CODE/" + name + ".py"], cwd=ROOT).decode("utf-8-sig")


class DateParametersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.scratch = Path(cls.temp.name)
        cls.env = patch.dict(os.environ, {"CITIBIKE_PROJECT_ROOT": str(cls.scratch),
                                         "MPLCONFIGDIR": str(cls.scratch / "mpl")})
        cls.env.start()
        sys.path.insert(0, str(CODE))
        # Reject any attempt to read or write real data/results in this process.
        def guard(event, args):
            if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
                p = Path(os.fsdecode(args[0])).resolve()
                # Formal Markdown documentation is read by separate link tests;
                # raw data and frozen result directories remain inaccessible.
                if (any(p.is_relative_to(ROOT / folder) for folder in ("00_DATA", "02_RESULTS"))
                        or (p.is_relative_to(ROOT / "03_SHOWCASE") and p.suffix != ".md")):
                    raise AssertionError("Test accessed protected project files: " + str(p))
        sys.addaudithook(guard)
        import pandas as pd
        cls.pd = pd
        cls.modules = {n: importlib.import_module(module_name(n)) for n in
                       ("event_baseline", "timeout_simulation", "wait_duration_diagnostics",
                        "holdout_validation", "weekend_validation", "weekend_preposition")}
        cls.legacy = {}
        for name in ("event_baseline", "timeout_simulation", "wait_duration_diagnostics", "holdout_validation", "weekend_validation"):
            source = original(name).replace("import event_baseline as", "import legacy_event_baseline as")
            source = source.replace("import timeout_simulation as", "import legacy_timeout_simulation as")
            source = source.replace("import holdout_validation as", "import legacy_holdout_validation as")
            source = rewrite_imports(source)
            alias = "legacy_" + name
            module = types.ModuleType(alias)
            module.__file__ = str(cls.scratch / (alias + ".py"))
            sys.modules[alias] = module
            exec(compile(source, module.__file__, "exec"), module.__dict__)
            cls.legacy[name] = module

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        sys.path.remove(str(CODE))
        cls.temp.cleanup()

    def trips(self, date, rows):
        """Rows: ride id, start minute, end minute, origin, destination."""
        pd = self.pd
        result = []
        for ride, start, end, origin, destination in rows:
            st = pd.Timestamp(date) + pd.Timedelta(minutes=start)
            en = pd.Timestamp(date) + pd.Timedelta(minutes=end)
            result.append(dict(ride_id=ride, started_at=st, ended_at=en,
                               start_date=str(st.date()), end_date=str(en.date()),
                               start_hour=st.hour, end_hour=en.hour,
                               start_station_id=origin, end_station_id=destination,
                               start_lat=40., start_lng=-73., end_lat=40., end_lng=-73.))
        return pd.DataFrame(result)

    def same(self, a, b):
        self.assertEqual(json.dumps(a, sort_keys=True, default=str), json.dumps(b, sort_keys=True, default=str))

    def compare(self, date, rows, initial=None):
        initial = {"A": 1, "B": 0} if initial is None else initial
        frame = self.trips(date, rows)
        args = (["A", "B"], {"A": 1, "B": 1}, initial, frame, "synthetic")
        self.legacy["event_baseline"].DATE = date  # Oracle only, never production state.
        outputs = {}
        for module, function in (("event_baseline", "simulate"), ("timeout_simulation", "replay"),
                                 ("wait_duration_diagnostics", "replay_waits")):
            expected = getattr(self.legacy[module], function)(*args)
            actual = getattr(self.modules[module], function)(*args, date=date)
            self.same(expected, actual)
            outputs[module] = actual
        self.assertEqual(self.modules["event_baseline"].DATE, "2026-04-21")
        return outputs

    def test_alternating_dates_and_repeatability(self):
        rows = [("internal", 60, 90, "A", "B"), ("outbound", 120, 160, "B", "X")]
        first = self.compare("2026-04-21", rows)
        self.compare("2026-04-25", rows)
        self.compare("2026-04-26", rows)
        self.same(first, self.compare("2026-04-21", rows))

    def test_midnight_carry_in_and_carry_out(self):
        outputs = self.compare("2026-04-25", [
            ("carry_in", -20, 10, "A", "B"),
            ("carry_out", 1430, 1450, "A", "B"),
            ("next_midnight", 1440, 1460, "B", "X"),
            ("arrive_midnight", -10, 0, "X", "A")])
        self.assertEqual(outputs["timeout_simulation"][0]["requests"], 1)
        self.assertEqual(outputs["timeout_simulation"][0]["internal_in_transit_at_midnight"], 1)

    def test_same_time_departure_before_arrival_and_cancelled_return(self):
        outputs = self.compare("2026-04-26", [
            ("return", 530, 540, "X", "A"),
            ("rejected", 540, 560, "A", "B")], {"A": 0, "B": 0})
        summary = outputs["timeout_simulation"][0]
        self.assertEqual(summary["borrow_rejected"], 1)
        self.assertEqual(summary["cancelled_internal_arrivals_today"], 1)
        self.assertEqual(summary["immediate_returns"], 1)

    def test_fifo_and_exact_timeout_boundary(self):
        outputs = self.compare("2026-04-25", [
            ("wait_first", 470, 480, "X", "A"),
            ("wait_second", 471, 481, "X", "A"),
            ("depart_first", 510, 520, "A", "X"),
            ("depart_second", 511, 521, "A", "X")])
        summary, waits, _ = outputs["timeout_simulation"]
        self.assertEqual(summary["wait_resolved"], 2)
        self.assertEqual(summary["timeout_external_exit"], 0)
        self.assertEqual([w["observed_wait_minutes"] for w in waits], [30., 30.])
        self.assertEqual([w["ride_id"] for w in waits], ["wait_first", "wait_second"])

    def test_after_timeout_and_midnight_censoring(self):
        outputs = self.compare("2026-04-26", [
            ("timeout", 470, 480, "X", "A"),
            ("late_departure", 510 + 1/60, 520, "A", "X"),
            ("refill", 1400, 1410, "X", "A"),
            ("censor", 1401, 1410, "X", "A")])
        summary, waits, _ = outputs["timeout_simulation"]
        self.assertEqual(summary["timeout_external_exit"], 1)
        self.assertEqual(summary["pending_at_midnight"], 1)
        self.assertEqual(waits[-1]["status"], "censored_at_midnight")
        self.assertEqual(waits[-1]["observed_wait_minutes"], 30.)

    def test_equal_time_arrivals_keep_input_tie_order(self):
        outputs = self.compare("2026-04-25", [
            ("first", 470, 480, "X", "A"),
            ("second", 470, 480, "X", "A"),
            ("leave1", 490, 500, "A", "X"),
            ("leave2", 491, 501, "A", "X")])
        waits = outputs["timeout_simulation"][1]
        self.assertEqual([w["ride_id"] for w in waits], ["first", "second"])
        self.assertEqual([w["observed_wait_minutes"] for w in waits], [10., 11.])

    def test_weekend_forwarding_oracle_and_exception_isolation(self):
        wk = self.modules["weekend_validation"]
        hold = self.modules["holdout_validation"]
        legacy_wk = self.legacy["weekend_validation"]
        legacy_hold = self.legacy["holdout_validation"]
        dates = ["2026-04-25", "2026-04-26"]
        trips = self.pd.concat([self.trips(date, [(date, 60, 90, "A", "B")]) for date in dates], ignore_index=True)
        configurations = {"no_rebalancing": {"A": 1, "B": 0}, "weekday_80": {"A": 0, "B": 1}}
        reference = []
        for date in dates:
            self.legacy["event_baseline"].DATE = date
            row = {"date": date}
            for prefix, stock in zip(("base", "plan"), configurations.values()):
                summary = self.legacy["timeout_simulation"].replay(
                    ["A", "B"], {"A": 1, "B": 1}, stock, trips, prefix)[0]
                row[prefix + "_rejected"] = summary["borrow_rejected"]
                row[prefix + "_timeouts"] = summary["timeout_external_exit"]
            reference.append(row)
        frozen = wk.RESULTS / "DAY6_HOLDOUT" / "holdout_daily_comparison.csv"
        self.assertTrue(frozen.is_relative_to(self.scratch))
        frozen.parent.mkdir(parents=True, exist_ok=True)
        self.pd.DataFrame(reference).to_csv(frozen, index=False)
        empty = self.pd.DataFrame(columns=["date", "station_id"])
        args = (["A", "B"], {"A": 1, "B": 1}, configurations)
        snapshot = (self.modules["event_baseline"].DATE, list(hold.DATES), set(hold.DATE_SET))
        with contextlib.redirect_stdout(io.StringIO()):
            with patch.object(legacy_hold, "extract_holdout", return_value=(trips, {})), patch.object(legacy_hold, "reconcile", return_value=(empty, empty)):
                expected = legacy_wk.event_evaluation(*args)
            with patch.object(hold, "extract_holdout", return_value=(trips, {})) as extract, patch.object(hold, "reconcile", return_value=(empty, empty)) as reconcile:
                actual = wk.event_evaluation(*args, dates=dates)
                self.same(expected, actual)
                self.assertEqual(extract.call_args.kwargs["dates"], dates)
                self.assertEqual(reconcile.call_args.kwargs["dates"], dates)
                reversed_result = wk.event_evaluation(*args, dates=list(reversed(dates)))
                self.assertEqual([r["date"] for r in reversed_result[0]], [dates[1]] * 2 + [dates[0]] * 2)
                self.same(actual, wk.event_evaluation(*args))
            with patch.object(hold, "extract_holdout", side_effect=RuntimeError("synthetic failure")):
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    wk.event_evaluation(*args, dates=["2026-04-22"])
            with patch.object(hold, "extract_holdout", return_value=(trips, {})), patch.object(hold, "reconcile", return_value=(empty, empty)), patch.object(wk.event_model, "replay", side_effect=RuntimeError("replay failure")):
                with self.assertRaisesRegex(RuntimeError, "replay failure"):
                    wk.event_evaluation(*args, dates=dates)
        self.assertEqual(snapshot, (self.modules["event_baseline"].DATE, hold.DATES, hold.DATE_SET))

    def test_default_dates_and_date_sets_unchanged(self):
        hold = self.modules["holdout_validation"]
        wk = self.modules["weekend_validation"]
        self.assertEqual(hold.DATES, [f"2026-04-{d}" for d in range(22, 31)])
        self.assertEqual(hold.DATE_SET, set(hold.DATES))
        self.assertEqual(wk.EVAL_DATES, ["2026-04-25", "2026-04-26"])
        self.assertEqual(self.modules["weekend_preposition"].TRAIN_DATES,
                         [f"2026-04-{d:02}" for d in (4, 5, 11, 12, 18, 19)])
        args = (["A", "B"], {"A": 1, "B": 1}, {"A": 1, "B": 0},
                self.trips("2026-04-21", [("one", 60, 90, "A", "B")]), "default")
        for module, fn in (("event_baseline", "simulate"), ("timeout_simulation", "replay"),
                           ("wait_duration_diagnostics", "replay_waits")):
            function = getattr(self.modules[module], fn)
            self.same(function(*args), function(*args, date="2026-04-21"))

    def test_extraction_and_reconciliation_date_order_membership(self):
        pd = self.pd
        hold = self.modules["holdout_validation"]
        legacy = self.legacy["holdout_validation"]
        trips = self.trips("2026-04-25", [("cross", 1430, 1450, "A", "B"),
                                         ("same", 60, 90, "A", "B"),
                                         ("prior", -10, 10, "X", "A")])
        mapping = self.scratch / "mapping.csv"
        pd.DataFrame({"raw_station_id": ["A", "B"], "canonical_station_id": ["A", "B"]}).to_csv(mapping, index=False)
        archive = self.scratch / "synthetic.zip"
        with ZipFile(archive, "w") as z:
            z.writestr("trips.csv", trips[hold.extract_module.COLUMNS].to_csv(index=False))
        flow = []
        for row in trips.itertuples():
            for prefix, kind in (("start", "start"), ("end", "end")):
                station = getattr(row, prefix + "_station_id")
                if station in {"A", "B"}:
                    flow.append(dict(date=getattr(row, prefix + "_date"), station_id=station,
                                     hour=getattr(row, prefix + "_hour"), type=kind, count=1))
        flowfile = self.scratch / "flow.csv"
        pd.DataFrame(flow).to_csv(flowfile, index=False)
        snapshot = (list(hold.DATES), set(hold.DATE_SET))
        with patch.multiple(hold.extract_module, MAPPING_FILE=mapping, ZIP_FILE=archive, FLOW_FILE=flowfile):
            for dates in (["2026-04-25"], ["2026-04-26"],
                          ["2026-04-26", "2026-04-25", "2026-04-25"], ["2026-04-25"]):
                legacy.DATES, legacy.DATE_SET = list(dates), set(dates)
                old_trips, old_stats = legacy.extract_holdout(["A", "B"])
                new_trips, new_stats = hold.extract_holdout(["A", "B"], dates=dates)
                pd.testing.assert_frame_equal(old_trips, new_trips)
                self.same(old_stats, new_stats)
                old = legacy.reconcile(old_trips, ["A", "B"])
                new = hold.reconcile(new_trips, ["A", "B"], dates=dates)
                for a, b in zip(old, new):
                    pd.testing.assert_frame_equal(a, b)
                self.assertTrue(new[1].empty)
                self.assertEqual(snapshot, (hold.DATES, hold.DATE_SET))

    def test_engine_ast_and_models_unchanged(self):
        allowed = {"event_baseline": {"read_inputs", "simulate"}, "timeout_simulation": {"replay"},
                   "wait_duration_diagnostics": {"replay_waits"},
                   "holdout_validation": {"extract_holdout", "reconcile", "main"},
                   "weekend_validation": {"event_evaluation"}}
        for script in SCRIPTS:
            path = module_path(script)
            if path.stem.startswith(("day7_", "lesson")) or path.stem == "project_paths":
                continue
            old = ast.parse(original(path.stem))
            new = ast.parse(path.read_text(encoding="utf-8-sig"))
            old_funcs = {n.name: n for n in old.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            new_funcs = {n.name: n for n in new.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            for name, node in old_funcs.items():
                if name not in allowed.get(path.stem, set()):
                    self.assertEqual(ast.dump(node), ast.dump(new_funcs[name]), (path.stem, name))
            if path.stem in {"event_baseline", "timeout_simulation", "wait_duration_diagnostics"}:
                for name in allowed[path.stem]:
                    node = copy.deepcopy(new_funcs[name])
                    node.args.kwonlyargs = []; node.args.kw_defaults = []
                    node.body.pop(0)  # Only the None -> original default adapter.
                    token = "DATE" if path.stem == "event_baseline" else "base.DATE"
                    class RestoreDate(ast.NodeTransformer):
                        def visit_Name(self, n):
                            return ast.parse(token, mode="eval").body if n.id == "date" else n
                    node = RestoreDate().visit(node)
                    self.assertEqual(ast.dump(node), ast.dump(old_funcs[name]), (path.stem, name))
            if path.stem == "holdout_validation":
                for name in ("extract_holdout", "reconcile"):
                    node = copy.deepcopy(new_funcs[name])
                    node.args.kwonlyargs = []; node.args.kw_defaults = []
                    node.body = node.body[2:]  # Per-call list and membership set.
                    class RestoreDates(ast.NodeTransformer):
                        def visit_Name(self, n):
                            if n.id in {"dates", "date_set"}:
                                n.id = {"dates": "DATES", "date_set": "DATE_SET"}[n.id]
                            return n
                    node = RestoreDates().visit(node)
                    self.assertEqual(ast.dump(node), ast.dump(old_funcs[name]), name)
            for n in ast.walk(new):
                if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store):
                    self.assertNotIn(n.attr, {"DATE", "DATES", "DATE_SET"})
        # Both validation callers must forward date explicitly at every replay.
        for name in ("holdout_validation", "weekend_validation"):
            tree = ast.parse(module_path(name).read_text(encoding="utf-8-sig"))
            for n in ast.walk(tree):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "replay":
                    self.assertIn("date", [k.arg for k in n.keywords])


if __name__ == "__main__":
    unittest.main()
