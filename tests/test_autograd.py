"""Test di ronkgrad (Fase 3) — il gradient check su OGNI operazione.

Metodo: per ogni espressione, confrontiamo il gradiente ANALITICO (dal nostro
backward) con quello NUMERICO (differenze finite centrali, che usano solo il forward).
Due strade indipendenti: se coincidono, il backward e' giusto. Se un giorno una di
queste operazioni si romperà, questo file lo grida subito — ed e' cio' che permette a
tutte le fasi 4-8 di fidarsi ciecamente del motore.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ronklm.autograd import Tensor, cross_entropy  # noqa: E402


def grad_check(build, leaves, h: float = 1e-6, tol: float = 1e-5) -> float:
    """Confronta gradiente analitico e numerico per un'espressione scalare.

    `build` e' una funzione senza argomenti che ricostruisce il grafo dai `leaves`
    (i cui .data possono essere mutati) e ritorna un Tensor SCALARE. `leaves` sono i
    tensori-foglia di cui verifichiamo il gradiente. Ritorna l'errore relativo massimo.
    """
    # --- gradiente analitico: un forward + un backward ---
    for L in leaves:
        L.zero_grad()
    out = build()
    assert out.data.size == 1, "grad_check richiede un output scalare"
    out.backward()
    analytic = [L.grad.copy() for L in leaves]

    # --- gradiente numerico: due forward per ogni componente di ogni foglia ---
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
            num[idx] = (lp - lm) / (2 * h)  # differenza centrale (errore ~ h^2)
            it.iternext()
        denom = np.maximum(1e-8, np.abs(num) + np.abs(analytic[k]))
        rel = np.abs(num - analytic[k]) / denom
        max_rel = max(max_rel, float(rel.max()))
    assert max_rel < tol, f"gradient check fallito: err rel max {max_rel:.2e}"
    return max_rel


def _rng():
    return np.random.default_rng(0)


# --- operazioni base --------------------------------------------------------
def test_add_broadcasting():
    r = _rng()
    a = Tensor(r.normal(size=(4, 5)))
    b = Tensor(r.normal(size=(5,)))  # broadcast (5,) su (4,5)
    grad_check(lambda: (a + b).sum(), [a, b])


def test_sub():
    r = _rng()
    a = Tensor(r.normal(size=(3, 3)))
    b = Tensor(r.normal(size=(3, 3)))
    grad_check(lambda: (a - b).sum(), [a, b])


def test_mul_broadcasting():
    r = _rng()
    a = Tensor(r.normal(size=(4, 5)))
    b = Tensor(r.normal(size=(1, 5)))  # broadcast riga su (4,5)
    grad_check(lambda: (a * b).sum(), [a, b])


def test_div():
    r = _rng()
    a = Tensor(r.normal(size=(3, 4)))
    b = Tensor(r.normal(size=(3, 4)) ** 2 + 1.0)  # denominatore positivo
    grad_check(lambda: (a / b).sum(), [a, b])


def test_pow():
    r = _rng()
    a = Tensor(np.abs(r.normal(size=(3, 3))) + 0.5)
    grad_check(lambda: (a ** 3).sum(), [a])


def test_matmul_2d():
    r = _rng()
    a = Tensor(r.normal(size=(4, 6)))
    b = Tensor(r.normal(size=(6, 5)))
    grad_check(lambda: (a @ b).sum(), [a, b])


def test_matmul_batched():
    # matmul "a batch" (3D): serve all'attention di Fase 5.
    r = _rng()
    a = Tensor(r.normal(size=(2, 4, 6)))
    b = Tensor(r.normal(size=(2, 6, 5)))
    grad_check(lambda: (a @ b).sum(), [a, b])


def test_sum_axis():
    r = _rng()
    a = Tensor(r.normal(size=(4, 5)))
    coeff = r.normal(size=(4,))
    grad_check(lambda: (a.sum(axis=1) * Tensor(coeff)).sum(), [a])


def test_mean():
    r = _rng()
    a = Tensor(r.normal(size=(4, 5)))
    grad_check(lambda: a.mean(), [a])


# --- non-linearita' ---------------------------------------------------------
def test_relu():
    r = _rng()
    a = Tensor(r.normal(size=(5, 5)))
    grad_check(lambda: a.relu().sum(), [a])


def test_tanh():
    r = _rng()
    a = Tensor(r.normal(size=(5, 5)))
    grad_check(lambda: a.tanh().sum(), [a])


def test_exp():
    r = _rng()
    a = Tensor(r.normal(size=(4, 4)) * 0.5)
    grad_check(lambda: a.exp().sum(), [a])


def test_log():
    r = _rng()
    a = Tensor(np.abs(r.normal(size=(4, 4))) + 0.5)  # argomento positivo
    grad_check(lambda: a.log().sum(), [a])


def test_gelu_composite():
    r = _rng()
    a = Tensor(r.normal(size=(4, 4)))
    grad_check(lambda: a.gelu().sum(), [a])


def test_softmax():
    r = _rng()
    a = Tensor(r.normal(size=(3, 6)))
    w = r.normal(size=(3, 6))  # pesi fissi per ridurre a scalare
    grad_check(lambda: (a.softmax(axis=-1) * Tensor(w)).sum(), [a])


def test_cross_entropy():
    r = _rng()
    logits = Tensor(r.normal(size=(8, 5)))
    targets = r.integers(0, 5, size=8)
    grad_check(lambda: cross_entropy(logits, targets), [logits])


# --- casi che stressano l'accumulo ------------------------------------------
def test_reused_tensor_accumulates():
    # a usato piu' volte: il gradiente deve SOMMARE i contributi (a*a + a).
    r = _rng()
    a = Tensor(r.normal(size=(4,)))
    grad_check(lambda: (a * a + a).sum(), [a])


def test_small_mlp_composite():
    # una mini-rete: (x @ W1).tanh() @ W2 -> cross_entropy. Verifica che i gradienti
    # fluiscano attraverso una composizione realistica.
    r = _rng()
    x = Tensor(r.normal(size=(6, 4)))
    W1 = Tensor(r.normal(size=(4, 8)) * 0.5)
    W2 = Tensor(r.normal(size=(8, 5)) * 0.5)
    targets = r.integers(0, 5, size=6)
    grad_check(lambda: cross_entropy((x @ W1).tanh() @ W2, targets), [x, W1, W2])


if __name__ == "__main__":
    from _runner import run

    run(globals())
