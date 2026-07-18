"""Lancia tutti i test del progetto senza pytest.

Uso:
    python run_tests.py

Trova ogni tests/test_*.py, ne esegue le funzioni test_*, e riporta il totale.
Esce con codice != 0 se qualcosa fallisce (utile per CI / hook).
"""
from __future__ import annotations

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.join(ROOT, "tests")
sys.path.insert(0, ROOT)
sys.path.insert(0, TESTS_DIR)  # per l'import di _runner dai file di test

from _runner import run  # noqa: E402


def main() -> int:
    files = sorted(
        f for f in os.listdir(TESTS_DIR)
        if f.startswith("test_") and f.endswith(".py")
    )
    total_failed = 0
    for fname in files:
        print(f"\n=== {fname} ===")
        path = os.path.join(TESTS_DIR, fname)
        spec = importlib.util.spec_from_file_location(fname[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        total_failed += run(vars(mod))
    print(f"\n===== TOTALE: {'TUTTO VERDE' if total_failed == 0 else str(total_failed) + ' FALLITI'} =====")
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
