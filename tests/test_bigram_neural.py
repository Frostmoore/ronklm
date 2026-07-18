"""Test del BigramNeural (Fase 2).

Cosa dimostrano:
    - sanity di init: la loss iniziale e' ~log(V) (il modello parte ~uniforme);
    - il training FA SCENDERE la loss;
    - convergenza: la NLL appresa raggiunge quella del bigram a CONTEGGIO (stessa
      soluzione, strada diversa) -> il meccanismo forward/loss/backward/update funziona;
    - le righe apprese hanno gli stessi picchi dei conteggi (es. dopo 'q' -> 'u');
    - GRADIENT CHECK: il gradiente analitico dW combacia con quello numerico
      (differenze finite centrali). E' il test piu' importante della fase.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ronklm.dataset import Dataset  # noqa: E402
from ronklm.models.bigram_count import BigramCount  # noqa: E402
from ronklm.models.bigram_neural import BigramNeural  # noqa: E402

CORPUS = os.path.join(os.path.dirname(__file__), "..", "data", "input.txt")


def test_initial_loss_is_about_log_vocab():
    ds = Dataset.from_file(CORPUS)
    V = ds.tokenizer.vocab_size
    m = BigramNeural(V, np.random.default_rng(0))
    init = m.nll(ds.train)
    assert abs(init - np.log(V)) < 0.05  # ~uniforme all'avvio


def test_minibatch_training_decreases_loss():
    # Il ciclo stocastico "onesto" su minibatch: economico, deve far scendere la loss.
    ds = Dataset.from_file(CORPUS)
    V = ds.tokenizer.vocab_size
    m = BigramNeural(V, np.random.default_rng(0))
    hist = m.train(ds.train, steps=40, lr=5.0, rng=np.random.default_rng(1), batch_size=8192)
    assert hist[-1] < hist[0]


def test_converges_to_count_bigram():
    # La soluzione a minima cross-entropy E' la distribuzione empirica: il modello
    # neurale, partito da rumore, deve riscoprirla per pura discesa del gradiente.
    # Usiamo la via full-batch esatta dai conteggi (veloce): stessa soluzione.
    ds = Dataset.from_file(CORPUS)
    V = ds.tokenizer.vocab_size
    counts = BigramCount(V).fit(ds.train, smoothing=1)
    neural = BigramNeural(V, np.random.default_rng(0))
    neural.train_from_counts(counts.N, steps=500, lr=20.0)
    # le due NLL di train devono quasi coincidere (il neurale, senza smoothing,
    # sta un filo piu' basso: fitta meglio il train)
    assert abs(neural.nll(ds.train) - counts.nll(ds.train)) < 0.05
    assert neural.nll(ds.train) < np.log(V)


def test_learned_rows_match_counts_argmax():
    ds = Dataset.from_file(CORPUS)
    tok = ds.tokenizer
    V = tok.vocab_size
    counts = BigramCount(V).fit(ds.train, smoothing=1)
    neural = BigramNeural(V, np.random.default_rng(0))
    neural.train_from_counts(counts.N, steps=500, lr=20.0)
    P_neural = neural.probabilities()
    # per un carattere molto informativo come 'q', il piu' probabile deve coincidere
    qi = tok.stoi["q"]
    assert int(P_neural[qi].argmax()) == int(counts.P[qi].argmax())


def test_gradient_check_numerical_vs_analytic():
    # Verifica il backward con SOLO il forward: due strade indipendenti, stesso numero.
    rng = np.random.default_rng(1)
    V = 8
    m = BigramNeural(V, rng, init_std=0.5)
    x = rng.integers(0, V, size=32)
    y = rng.integers(0, V, size=32)

    _, dW = m.loss_and_grad(x, y)  # gradiente analitico

    h = 1e-5
    # controlliamo un campione di celle di W (non tutte: sarebbe lento)
    coords = [(int(rng.integers(0, V)), int(rng.integers(0, V))) for _ in range(20)]
    max_rel_err = 0.0
    for (i, j) in coords:
        orig = m.W[i, j]
        m.W[i, j] = orig + h
        lp = m.loss(x, y)
        m.W[i, j] = orig - h
        lm = m.loss(x, y)
        m.W[i, j] = orig
        num = (lp - lm) / (2 * h)  # differenza finita centrale (errore ~ h^2)
        ana = dW[i, j]
        denom = max(1e-8, abs(num) + abs(ana))
        max_rel_err = max(max_rel_err, abs(num - ana) / denom)
    assert max_rel_err < 1e-4, f"gradient check fallito: err rel max {max_rel_err:.2e}"


if __name__ == "__main__":
    from _runner import run

    run(globals())
