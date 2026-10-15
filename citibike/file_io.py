"""Small file helpers with the existing experiment I/O contracts."""
import csv
import hashlib


def file_sha256(path):
    """Hash a path.open-compatible file in 1 MiB blocks without modifying it."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_records_csv(path, records):
    """Write UTF-8 BOM CSV in first-record key order; empty input is a no-op.

    Preserve csv.DictWriter defaults and errors. Do not create parent folders.
    """
    if not records:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
