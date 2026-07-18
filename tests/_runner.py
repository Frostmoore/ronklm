"""Runner di test minimale, senza dipendenze (pytest non e' installato).

Esegue tutte le funzioni test_* del modulo chiamante. Ogni file di test finisce con:

    if __name__ == "__main__":
        from _runner import run
        run(globals())

E c'e' run_tests.py alla radice che lancia tutti i file di test insieme.
"""
from __future__ import annotations

import traceback


def run(ns: dict) -> int:
    """Esegue le funzioni test_* nel namespace `ns`. Ritorna il numero di fallimenti."""
    tests = sorted((k, v) for k, v in ns.items() if k.startswith("test_") and callable(v))
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}")
            print("        " + "\n        ".join(traceback.format_exc().splitlines()))
        else:
            passed += 1
            print(f"  ok    {name}")
    print(f"  -> {passed} passati, {failed} falliti")
    return failed


if __name__ == "__main__":
    raise SystemExit("Questo modulo va importato dai file di test, non eseguito.")
