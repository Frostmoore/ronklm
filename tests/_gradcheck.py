"""Helper condiviso: gradient check numerico vs analitico (usato dai test di autograd
e del blocco). Confronta il gradiente del nostro backward con la stima a differenze
finite centrali, che usa solo il forward.
"""
from __future__ import annotations

import numpy as np


def grad_check(build, leaves, h: float = 1e-6, tol: float = 1e-5) -> float:
    """`build`: funzione senza argomenti che ricostruisce il grafo dai `leaves` e
    ritorna un Tensor SCALARE. `leaves`: i tensori-foglia da verificare. Ritorna
    l'errore relativo massimo (e asserisce che sia < tol)."""
    for L in leaves:
        L.zero_grad()
    out = build()
    assert out.data.size == 1, "grad_check richiede un output scalare"
    out.backward()
    analytic = [L.grad.copy() for L in leaves]

    max_rel = 0.0
    for k, L in enumerate(leaves):
        num = np.zeros_like(L.data)
        it = np.nditer(L.data, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = L.data[idx]
            L.data[idx] = orig + h
            lp = float(build().data)
            L.data[idx] = orig - h
            lm = float(build().data)
            L.data[idx] = orig
            num[idx] = (lp - lm) / (2 * h)
            it.iternext()
        denom = np.maximum(1e-8, np.abs(num) + np.abs(analytic[k]))
        rel = np.abs(num - analytic[k]) / denom
        max_rel = max(max_rel, float(rel.max()))
    assert max_rel < tol, f"gradient check fallito: err rel max {max_rel:.2e}"
    return max_rel
