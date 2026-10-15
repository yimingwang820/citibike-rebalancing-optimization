"""Shared project root, resolved once before dependent modules bind their paths.

Set CITIBIKE_PROJECT_ROOT before starting Python to override the code-based root.
CLI arguments that explicitly accept relative paths retain their CWD semantics.
This module does not create directories or read project data.
"""

import os
from pathlib import Path


def get_project_root(root=None):
    """Resolve an explicit root, the environment override, or the code location.

    Explicit roots take precedence. Relative overrides resolve against the current
    working directory. This helper does not mutate already imported modules.
    """
    value = root if root is not None else os.environ.get("CITIBIKE_PROJECT_ROOT")
    if value is None:
        return Path(__file__).resolve().parents[1]
    if not str(value).strip():
        raise ValueError("Project root must not be empty")
    return Path(value).expanduser().resolve()


# Set the override before importing ANY project module; no runtime root setter.
ROOT = get_project_root()
