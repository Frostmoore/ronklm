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

from ronklm.autograd import Tensor, cat, cross_entropy  # noqa: E402

from _gradcheck import grad_check  # noqa: E402


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


def test_reshape():
    r = _rng()
    a = Tensor(r.normal(size=(2, 3, 4)))
    w = r.normal(size=(2, 12))
    grad_check(lambda: (a.reshape(2, 12) * Tensor(w)).sum(), [a])


def test_transpose():
    r = _rng()
    a = Tensor(r.normal(size=(2, 3, 4)))
    w = r.normal(size=(2, 4, 3))
    grad_check(lambda: (a.transpose(1, 2) * Tensor(w)).sum(), [a])


def test_gather_rows():
    # embedding: seleziona righe di una tabella (V, C) con indici (B, T)
    r = _rng()
    table = Tensor(r.normal(size=(5, 3)))  # (V=5, C=3)
    idx = np.array([[0, 2, 2], [1, 4, 0]])  # (B=2, T=3), con indici ripetuti
    w = r.normal(size=(2, 3, 3))
    grad_check(lambda: (table.gather_rows(idx) * Tensor(w)).sum(), [table])


def test_masked_fill():
    r = _rng()
    a = Tensor(r.normal(size=(3, 3)))
    mask = np.triu(np.ones((3, 3), dtype=bool), k=1)  # sopra la diagonale
    w = r.normal(size=(3, 3))
    # softmax dopo il mask, per un caso realistico di attention
    grad_check(lambda: (a.masked_fill(mask, -1e9).softmax(-1) * Tensor(w)).sum(), [a])


def test_var():
    r = _rng()
    a = Tensor(r.normal(size=(4, 6)))
    grad_check(lambda: a.var(axis=-1).sum(), [a])


def test_cat():
    # concatenazione (multi-head): il gradiente si ri-spezza sui pezzi originali
    r = _rng()
    a = Tensor(r.normal(size=(2, 3)))
    b = Tensor(r.normal(size=(2, 4)))
    w = r.normal(size=(2, 7))
    grad_check(lambda: (cat([a, b], axis=-1) * Tensor(w)).sum(), [a, b])


if __name__ == "__main__":
    from _runner import run

    run(globals())
