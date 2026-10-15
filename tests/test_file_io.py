"""Byte-level helper regressions against the stage 2C checkpoint."""
import ast
import copy
import csv
import hashlib
import importlib
import io
from pathlib import Path
import subprocess
import tempfile
import unittest

from project_layout import ROOT, MODULES, IO_EXTRACTIONS
from citibike.file_io import file_sha256, write_records_csv

CHECKPOINT = "8a0cd0113a69b35abfd21dcb3bfd3b8687f69c14"


def old_function(module, function):
    source = subprocess.check_output([
        "git", "show", CHECKPOINT + ":" + MODULES[module].replace(".", "/") + ".py"
    ], cwd=ROOT).decode("utf-8-sig")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == function)
    # Execute only this original helper, never import or execute an experiment.
    namespace = {"csv": csv, "hashlib": hashlib, "sha256": hashlib.sha256}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<checkpoint helper>", "exec"), namespace)
    return namespace[function]


class FileIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hashes = [old_function(m, IO_EXTRACTIONS[m][0]) for m in
                      ("holdout_validation", "weekend_preposition", "weekend_validation")]
        cls.writers = [old_function(m, "save_csv") for m in
                       ("fixed_charge_milp", "fixed_cost_sensitivity")]

    def test_sha256_empty_binary_and_chunk_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "中文 文件.bin"
            for size in (0, 1, 1024 * 1024 - 1, 1024 * 1024, 1024 * 1024 + 1, 3 * 1024 * 1024 + 7):
                payload = (bytes(range(256)) * (size // 256 + 1))[:size]
                path.write_bytes(payload)
                expected = hashlib.sha256(payload).hexdigest()
                for function in self.hashes + [file_sha256]:
                    self.assertEqual(function(path), expected)
                self.assertEqual(path.read_bytes(), payload)

    def test_sha256_read_mode_chunk_size_and_errors(self):
        class TracedFile(io.BytesIO):
            def read(self, size):
                calls.append(size)
                return super().read(size)
        class TracedPath:
            def open(self, mode):
                self.mode = mode
                return TracedFile(b"abc")
        for function in self.hashes + [file_sha256]:
            calls = []
            path = TracedPath()
            self.assertEqual(function(path), hashlib.sha256(b"abc").hexdigest())
            self.assertEqual(path.mode, "rb")
            self.assertEqual(calls, [1024 * 1024, 1024 * 1024])
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(FileNotFoundError):
                    function(Path(tmp) / "missing")
            with self.assertRaises(AttributeError):
                function("string paths were not supported by the old helpers")

    def test_csv_bytes_field_order_quoting_and_station_id_zeros(self):
        rows = [{"station_id": "5980.10", "note": '中文, "引号"\n换行', "stock": 0},
                {"stock": None, "note": "", "station_id": "001.00"},
                {"station_id": "0000", "stock": 2}]
        untouched = copy.deepcopy(rows)
        with tempfile.TemporaryDirectory() as tmp:
            expected = None
            for index, function in enumerate(self.writers + [write_records_csv]):
                path = Path(tmp) / f"{index}.csv"
                self.assertIsNone(function(path, rows))
                data = path.read_bytes()
                self.assertTrue(data.startswith(b"\xef\xbb\xbf"))
                if expected is None:
                    expected = data
                self.assertEqual(data, expected)
                with path.open(encoding="utf-8-sig", newline="") as file:
                    reader = csv.DictReader(file)
                    parsed = list(reader)
                    self.assertEqual(reader.fieldnames, ["station_id", "note", "stock"])
                self.assertEqual([r["station_id"] for r in parsed], ["5980.10", "001.00", "0000"])
            self.assertEqual(rows, untouched)

    def test_csv_empty_input_does_not_create_or_truncate(self):
        with tempfile.TemporaryDirectory() as tmp:
            for function in self.writers + [write_records_csv]:
                for empty in ([], (), None):
                    path = Path(tmp) / "not_created" / "empty.csv"
                    self.assertIsNone(function(path, empty))
                    self.assertFalse(path.parent.exists())
                    existing = Path(tmp) / "existing.csv"
                    existing.write_bytes(b"preserve me")
                    self.assertIsNone(function(existing, empty))
                    self.assertEqual(existing.read_bytes(), b"preserve me")

    def test_csv_errors_and_partial_write_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            for make_rows in (lambda: [{"id": "01"}, {"id": "02", "extra": 1}],
                              lambda: iter([{"id": "01"}])):
                outcomes = []
                for index, function in enumerate(self.writers + [write_records_csv]):
                    path = Path(tmp) / f"error{index}.csv"
                    path.write_bytes(b"previous")
                    try:
                        function(path, make_rows())
                    except (ValueError, TypeError) as error:
                        outcomes.append((type(error).__name__, str(error), path.read_bytes()))
                    else:
                        self.fail("Expected the original writer error")
                self.assertEqual(outcomes[0], outcomes[1])
                self.assertEqual(outcomes[0], outcomes[2])
            for function in self.writers + [write_records_csv]:
                with self.assertRaises(FileNotFoundError):
                    function(Path(tmp) / "missing" / "out.csv", [{"id": "01"}])

    def test_original_keyword_signatures_and_delegates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.bin"
            path.write_bytes(b"original keyword compatibility")
            for module in ("holdout_validation", "weekend_preposition", "weekend_validation"):
                function = getattr(importlib.import_module(MODULES[module]), IO_EXTRACTIONS[module][0])
                self.assertEqual(function(path=path), file_sha256(path))
            fixed = importlib.import_module(MODULES["fixed_charge_milp"])
            sensitivity = importlib.import_module(MODULES["fixed_cost_sensitivity"])
            a, b = Path(tmp) / "a.csv", Path(tmp) / "b.csv"
            self.assertIsNone(fixed.save_csv(path=a, records=[{"id": "001.00"}]))
            self.assertIsNone(sensitivity.save_csv(filepath=str(b), rows=[{"id": "001.00"}]))
            self.assertEqual(a.read_bytes(), b.read_bytes())


if __name__ == "__main__":
    unittest.main()
