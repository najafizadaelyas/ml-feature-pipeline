"""Patch Unix-only modules that Airflow imports unconditionally on Windows."""

from __future__ import annotations

import sys
import types

# fcntl and termios are Linux-only; mock them before any Airflow import touches them
for _mod in ("fcntl", "termios", "grp", "pwd"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)
