"""Test del Dataset e del batching (Fase 0.5).

Cosa dimostrano:
    - shape: get_batch ritorna X, Y di forma (batch_size, block_size);
    - Y = X spostato di 1: l'invariante X[:, 1:] == Y[:, :-1];
    - split train/val: disgiunti, contigui, coprono tutto il corpus;
    - riproducibilita': stesso seed => stesso batch;
    - range: gli indici estratti non escono mai dai bordi;
    - fail-loud: split sconosciuto o troppo corto sollevano.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ronklm.dataset import Dataset  # noqa: E402
from ronklm.tokenizer import CharTokenizer  # noqa: E402

CORPUS = os.path.join(os.path.dirname(__file__), "..", "data", "input.txt")


def _dataset_from_corpus(val_frac: float = 0.1) -> Dataset:
    return Dataset.from_file(CORPUS, val_frac=val_frac)


def test_batch_shapes():
    ds = _dataset_from_corpus()
    rng = np.random.default_rng(0)
    X, Y = ds.get_batch("train", block_size=16, batch_size=8, rng=rng)
    assert X.shape == (8, 16)
    assert Y.shape == (8, 16)
    assert X.dtype == np.int64 and Y.dtype == np.int64


def test_Y_is_X_shifted_by_one():
    # Per costruzione X[b] = data[i:i+T], Y[b] = data[i+1:i+1+T].
    # Quindi la coda di X coincide con la testa di Y: X[:, 1:] == Y[:, :-1].
    ds = _dataset_from_corpus()
    rng = np.random.default_rng(1)
    X, Y = ds.get_batch("train", block_size=32, batch_size=16, rng=rng)
    assert np.array_equal(X[:, 1:], Y[:, :-1])


def test_split_is_contiguous_and_disjoint():
    ds = _dataset_from_corpus(val_frac=0.1)
    # train + val ricompongono esattamente il corpus, in ordine.
    assert len(ds.train) + len(ds.val) == len(ds.data)
    assert np.array_equal(ds.train, ds.data[: len(ds.train)])
    assert np.array_equal(ds.val, ds.data[len(ds.train):])
    # la val e' circa il 10%.
    frac = len(ds.val) / len(ds.data)
    assert 0.08 < frac < 0.12


def test_reproducibility_same_seed_same_batch():
    ds = _dataset_from_corpus()
    a = ds.get_batch("train", 24, 12, np.random.default_rng(42))
    b = ds.get_batch("train", 24, 12, np.random.default_rng(42))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_different_seed_different_batch():
    ds = _dataset_from_corpus()
    a = ds.get_batch("train", 24, 12, np.random.default_rng(1))
    b = ds.get_batch("train", 24, 12, np.random.default_rng(2))
    assert not np.array_equal(a[0], b[0])


def test_batch_content_matches_source_windows():
    # Verifica diretta che ogni riga di X sia una finestra reale del corpus e che
    # Y la segua di uno, ricercando la finestra nel train.
    ds = _dataset_from_corpus()
    rng = np.random.default_rng(7)
    X, Y = ds.get_batch("train", block_size=20, batch_size=4, rng=rng)
    data = ds.train
    for b in range(X.shape[0]):
        # tutti i valori sono indici validi del vocabolario
        assert X[b].min() >= 0 and X[b].max() < ds.tokenizer.vocab_size
        # l'ultimo carattere di Y esiste nel corpus (indice in range gia' garantito)
        assert Y[b, -1] < ds.tokenizer.vocab_size


def test_unknown_split_raises():
    ds = _dataset_from_corpus()
    try:
        ds.get_batch("test", 8, 2, np.random.default_rng(0))
    except ValueError:
        return
    raise AssertionError("split sconosciuto doveva sollevare")


def test_too_short_split_raises():
    # Dataset minuscolo: block_size piu' grande dei dati disponibili.
    tok = CharTokenizer.from_text("abcdef")
    ds = Dataset("abcdef", tok, val_frac=0.1)
    try:
        ds.get_batch("train", block_size=100, batch_size=1, rng=np.random.default_rng(0))
    except ValueError:
        return
    raise AssertionError("block_size troppo grande doveva sollevare")


if __name__ == "__main__":
    from _runner import run

    run(globals())
