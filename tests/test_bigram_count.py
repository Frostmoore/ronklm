"""Test del BigramCount (Fase 1).

Cosa dimostrano:
    - i conteggi sono corretti su un corpus giocattolo noto;
    - ogni riga di P e' una distribuzione di probabilita' (somma 1);
    - lo smoothing evita zeri (nessuna probabilita' nulla);
    - la NLL del bigram batte quella del modello uniforme (impara qualcosa);
    - la NLL su val e' finita (grazie allo smoothing);
    - la generazione produce indici validi ed e' riproducibile col seed.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ronklm.dataset import Dataset  # noqa: E402
from ronklm.models.bigram_count import BigramCount  # noqa: E402

CORPUS = os.path.join(os.path.dirname(__file__), "..", "data", "input.txt")


def test_counts_on_toy_corpus():
    # "abab" -> coppie: a->b, b->a, a->b. Quindi N[a,b]=2, N[b,a]=1.
    data = np.array([0, 1, 0, 1], dtype=np.int64)  # a=0, b=1
    m = BigramCount(vocab_size=2).fit(data, smoothing=0)
    assert m.N[0, 1] == 2
    assert m.N[1, 0] == 1
    assert m.N[0, 0] == 0 and m.N[1, 1] == 0


def test_rows_are_probability_distributions():
    ds = Dataset.from_file(CORPUS)
    m = BigramCount(ds.tokenizer.vocab_size).fit(ds.train)
    row_sums = m.P.sum(axis=1)
    assert np.allclose(row_sums, 1.0)


def test_smoothing_removes_zeros():
    ds = Dataset.from_file(CORPUS)
    m = BigramCount(ds.tokenizer.vocab_size).fit(ds.train, smoothing=1)
    assert (m.P > 0).all()  # nessuna probabilita' nulla


def test_beats_uniform_on_train_and_val():
    ds = Dataset.from_file(CORPUS)
    V = ds.tokenizer.vocab_size
    m = BigramCount(V).fit(ds.train)
    uniform = BigramCount.uniform_nll(V)
    assert m.nll(ds.train) < uniform
    assert m.nll(ds.val) < uniform


def test_val_nll_is_finite():
    ds = Dataset.from_file(CORPUS)
    m = BigramCount(ds.tokenizer.vocab_size).fit(ds.train, smoothing=1)
    assert np.isfinite(m.nll(ds.val))


def test_generation_is_valid_and_reproducible():
    ds = Dataset.from_file(CORPUS)
    V = ds.tokenizer.vocab_size
    m = BigramCount(V).fit(ds.train)
    a = m.generate(np.random.default_rng(0), n=200, start=0)
    b = m.generate(np.random.default_rng(0), n=200, start=0)
    assert len(a) == 200
    assert all(0 <= i < V for i in a)
    assert a == b  # stesso seed -> stessa generazione


if __name__ == "__main__":
    from _runner import run

    run(globals())
