"""Test della self-attention causale (Fase 5).

Cosa dimostrano:
    - shape: Head trasforma (B, T, C) -> (B, T, head_size);
    - CAUSALITA': la matrice di attenzione e' triangolare inferiore (una posizione non
      guarda mai il futuro) e le sue righe sommano a 1;
    - i gradienti fluiscono su tutti i parametri (Q, K, V collegati al grafo);
    - l'AttentionLM si addestra: la loss su tutte le posizioni scende.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ronklm.autograd import Tensor  # noqa: E402
from ronklm.dataset import Dataset  # noqa: E402
from ronklm.models.attention import AttentionLM, Head  # noqa: E402
from ronklm.optim import AdamW  # noqa: E402

CORPUS = os.path.join(os.path.dirname(__file__), "..", "data", "input.txt")


def test_head_output_shape():
    rng = np.random.default_rng(0)
    B, T, C, H = 2, 5, 8, 4
    head = Head(C, H, block_size=T, rng=rng)
    x = Tensor(rng.normal(size=(B, T, C)))
    out = head(x)
    assert out.data.shape == (B, T, H)


def test_attention_is_causal_and_normalized():
    rng = np.random.default_rng(1)
    B, T, C, H = 2, 6, 8, 4
    head = Head(C, H, block_size=T, rng=rng)
    x = Tensor(rng.normal(size=(B, T, C)))
    head(x)
    att = head.last_att  # (B, T, T)
    # causalita': tutto cio' che sta STRETTAMENTE sopra la diagonale deve essere ~0
    upper = np.triu(np.ones((T, T), dtype=bool), k=1)
    assert np.allclose(att[:, upper], 0.0, atol=1e-8)
    # ogni riga (distribuzione sulle posizioni <= t) somma a 1
    assert np.allclose(att.sum(axis=-1), 1.0)


def test_attention_gradients_flow():
    rng = np.random.default_rng(2)
    ds = Dataset.from_file(CORPUS)
    V = ds.tokenizer.vocab_size
    m = AttentionLM(V, block_size=8, n_embd=16, rng=rng)
    X, Y = ds.get_batch("train", 8, 4, np.random.default_rng(0))
    m.zero_grad()
    m.loss(X, Y).backward()
    for p in m.parameters():
        assert np.abs(p.grad).sum() > 0


def test_attention_lm_trains():
    rng = np.random.default_rng(3)
    ds = Dataset.from_file(CORPUS)
    V = ds.tokenizer.vocab_size
    m = AttentionLM(V, block_size=8, n_embd=24, rng=rng)
    opt = AdamW(m.parameters(), lr=3e-3)
    brng = np.random.default_rng(0)
    first = last = None
    losses = []
    for step in range(120):
        X, Y = ds.get_batch("train", 8, 32, brng)
        m.zero_grad()
        loss = m.loss(X, Y)
        loss.backward()
        opt.step()
        losses.append(loss.data.item())
    assert np.mean(losses[-15:]) < np.mean(losses[:15])
    assert np.mean(losses[-15:]) < np.log(V)  # meglio dell'uniforme


if __name__ == "__main__":
    from _runner import run

    run(globals())
