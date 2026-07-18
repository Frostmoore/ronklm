"""Test del blocco Transformer e dei suoi componenti (Fase 6).

Cosa dimostrano:
    - LayerNorm normalizza (media ~0, varianza ~1 sull'ultima dimensione all'init);
    - LayerNorm ha gradiente corretto (gradient check);
    - MultiHeadAttention e FeedForward preservano la forma (B, T, n_embd);
    - il Block preserva la forma (e' cio' che lo rende impilabile);
    - dopo backward OGNI parametro del blocco riceve gradiente (nessuno scollegato);
    - un mini-modello con un Block si addestra (la loss scende).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ronklm.autograd import Tensor, cross_entropy  # noqa: E402
from ronklm.dataset import Dataset  # noqa: E402
from ronklm.models.block import Block, FeedForward, MultiHeadAttention  # noqa: E402
from ronklm.nn import Embedding, LayerNorm, Linear  # noqa: E402
from ronklm.optim import AdamW  # noqa: E402

CORPUS = os.path.join(os.path.dirname(__file__), "..", "data", "input.txt")


def test_layernorm_normalizes():
    rng = np.random.default_rng(0)
    ln = LayerNorm(8)
    x = Tensor(rng.normal(loc=5.0, scale=3.0, size=(4, 8)))  # media/scala lontane da 0/1
    out = ln(x).data
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-6)
    assert np.allclose(out.std(axis=-1), 1.0, atol=1e-3)


def test_layernorm_gradcheck():
    from _gradcheck import grad_check

    rng = np.random.default_rng(1)
    ln = LayerNorm(6)
    x = Tensor(rng.normal(size=(3, 6)))
    w = rng.normal(size=(3, 6))
    grad_check(lambda: (ln(x) * Tensor(w)).sum(), [x, ln.gamma, ln.beta])


def test_multihead_and_ffn_preserve_shape():
    rng = np.random.default_rng(2)
    B, T, C = 2, 5, 12
    mha = MultiHeadAttention(C, n_head=3, block_size=T, rng=rng)
    ffn = FeedForward(C, rng)
    x = Tensor(rng.normal(size=(B, T, C)))
    assert mha(x).data.shape == (B, T, C)
    assert ffn(x).data.shape == (B, T, C)


def test_block_preserves_shape():
    rng = np.random.default_rng(3)
    B, T, C = 2, 6, 16
    blk = Block(C, n_head=4, block_size=T, rng=rng)
    x = Tensor(rng.normal(size=(B, T, C)))
    out = blk(x)
    assert out.data.shape == (B, T, C)
    assert np.isfinite(out.data).all()


def test_block_all_params_get_gradient():
    rng = np.random.default_rng(4)
    B, T, C = 2, 6, 16
    blk = Block(C, n_head=4, block_size=T, rng=rng)
    x = Tensor(rng.normal(size=(B, T, C)))
    blk.zero_grad()
    blk(x).sum().backward()
    for p in blk.parameters():
        assert np.abs(p.grad).sum() > 0, "un parametro del blocco non riceve gradiente"


def test_block_lm_trains():
    # mini-LM: embedding -> Block -> testa. Deve addestrarsi (loss in calo).
    rng = np.random.default_rng(5)
    ds = Dataset.from_file(CORPUS)
    V = ds.tokenizer.vocab_size
    T, C = 8, 32
    tok = Embedding(V, C, rng)
    blk = Block(C, n_head=4, block_size=T, rng=rng)
    head = Linear(C, V, rng)
    params = tok.parameters() + blk.parameters() + head.parameters()
    opt = AdamW(params, lr=2e-3)
    brng = np.random.default_rng(0)
    losses = []
    for _ in range(80):
        X, Y = ds.get_batch("train", T, 16, brng)
        for p in params:
            p.zero_grad()
        logits = head(blk(tok(X)))              # (B, T, V)
        loss = cross_entropy(logits.reshape(X.shape[0] * T, V), Y.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.data.item())
    assert np.mean(losses[-10:]) < np.mean(losses[:10])
    assert np.mean(losses[-10:]) < np.log(V)


if __name__ == "__main__":
    from _runner import run

    run(globals())
