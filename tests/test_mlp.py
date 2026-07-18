"""Test dell'MLP e dell'infrastruttura nn/optim (Fase 4).

Cosa dimostrano:
    - Module.parameters() raccoglie TUTTI i parametri (embedding + 2 Linear con bias);
    - i logits hanno forma (B, vocab); la loss e' scalare;
    - dopo backward, OGNI parametro ha gradiente non nullo (nessun peso scollegato);
    - il training con AdamW fa scendere la loss;
    - MLP BATTE il bigram sulla validation (milestone M2): guardare piu' contesto paga.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ronklm.dataset import Dataset  # noqa: E402
from ronklm.models.mlp import MLP  # noqa: E402
from ronklm.optim import AdamW, SGD  # noqa: E402

CORPUS = os.path.join(os.path.dirname(__file__), "..", "data", "input.txt")


def _make(block_size=8, n_embd=16, n_hidden=64, seed=0):
    ds = Dataset.from_file(CORPUS)
    V = ds.tokenizer.vocab_size
    m = MLP(V, block_size, n_embd, n_hidden, np.random.default_rng(seed))
    return ds, V, m


def test_parameters_collected():
    _, _, m = _make()
    # Embedding.weight + h.(W,b) + head.(W,b) = 5 tensori parametro
    assert len(m.parameters()) == 5


def test_logits_shape_and_scalar_loss():
    ds, V, m = _make()
    X, Y = ds.get_batch("train", m.block_size, 12, np.random.default_rng(0))
    logits = m.logits(X)
    assert logits.data.shape == (12, V)
    loss = m.loss(X, MLP.targets_from_batch(Y))
    assert loss.data.size == 1


def test_all_params_get_gradient():
    ds, V, m = _make()
    X, Y = ds.get_batch("train", m.block_size, 16, np.random.default_rng(0))
    m.zero_grad()
    m.loss(X, MLP.targets_from_batch(Y)).backward()
    for p in m.parameters():
        assert np.abs(p.grad).sum() > 0, "un parametro non riceve gradiente (scollegato dal grafo)"


def test_training_decreases_loss():
    ds, V, m = _make()
    opt = SGD(m.parameters(), lr=0.5)
    rng = np.random.default_rng(1)
    losses = []
    for _ in range(60):
        X, Y = ds.get_batch("train", m.block_size, 64, rng)
        m.zero_grad()
        loss = m.loss(X, MLP.targets_from_batch(Y))
        loss.backward()
        opt.step()
        losses.append(loss.data.item())
    assert np.mean(losses[-10:]) < np.mean(losses[:10])


def test_mlp_beats_bigram_on_val():
    # Milestone M2: guardare piu' contesto abbassa la NLL sotto quella del bigram (2.35).
    ds, V, m = _make(block_size=8, n_embd=24, n_hidden=128)
    opt = AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    for _ in range(800):
        X, Y = ds.get_batch("train", m.block_size, 64, rng)
        m.zero_grad()
        m.loss(X, MLP.targets_from_batch(Y)).backward()
        opt.step()
    val_nll = m.nll(ds.val, m.block_size, np.random.default_rng(99))
    assert val_nll < 2.2, f"NLL val {val_nll:.3f} non batte il bigram (2.35)"


if __name__ == "__main__":
    from _runner import run

    run(globals())
