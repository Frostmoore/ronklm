"""Test del GPT completo e della generazione (Fase 7).

Cosa dimostrano:
    - forma dei logits (B, T, vocab) e loss scalare;
    - loss iniziale nell'ordine di log(vocab) (sanity dell'inizializzazione);
    - ogni parametro riceve gradiente (nessun componente scollegato);
    - la generazione produce indici validi, rispetta la lunghezza, ed e' riproducibile;
    - top-k limita il supporto; temperature->0 tende al greedy (deterministico);
    - save/load ricostruisce un modello IDENTICO (stessi logit) + il vocabolario;
    - un breve training fa scendere la loss.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ronklm.dataset import Dataset  # noqa: E402
from ronklm.generate import generate  # noqa: E402
from ronklm.models.gpt import GPT, GPTConfig  # noqa: E402
from ronklm.optim import AdamW  # noqa: E402

CORPUS = os.path.join(os.path.dirname(__file__), "..", "data", "input.txt")


def _small(seed=0):
    ds = Dataset.from_file(CORPUS)
    V = ds.tokenizer.vocab_size
    cfg = GPTConfig(vocab_size=V, block_size=16, n_layer=2, n_head=2, n_embd=32)
    return ds, cfg, GPT(cfg, np.random.default_rng(seed))


def test_logits_shape_and_scalar_loss():
    ds, cfg, m = _small()
    X, Y = ds.get_batch("train", cfg.block_size, 4, np.random.default_rng(0))
    assert m.logits(X).data.shape == (4, cfg.block_size, cfg.vocab_size)
    assert m.loss(X, Y).data.size == 1


def test_initial_loss_order_of_log_vocab():
    ds, cfg, m = _small()
    X, Y = ds.get_batch("train", cfg.block_size, 32, np.random.default_rng(0))
    init = m.loss(X, Y).data.item()
    # non chiediamo esattamente log(V), ma lo stesso ordine di grandezza
    assert np.log(cfg.vocab_size) - 1.0 < init < np.log(cfg.vocab_size) + 1.0


def test_all_params_get_gradient():
    ds, cfg, m = _small()
    X, Y = ds.get_batch("train", cfg.block_size, 8, np.random.default_rng(0))
    m.zero_grad()
    m.loss(X, Y).backward()
    for p in m.parameters():
        assert np.abs(p.grad).sum() > 0


def test_generation_valid_and_reproducible():
    ds, cfg, m = _small()
    a = generate(m, [0, 1, 2], 30, np.random.default_rng(7))
    b = generate(m, [0, 1, 2], 30, np.random.default_rng(7))
    assert len(a) == 30
    assert all(0 <= i < cfg.vocab_size for i in a)
    assert a == b  # stesso seed -> stessa generazione


def test_top_k_restricts_support():
    ds, cfg, m = _small()
    # con top_k=1 la scelta e' deterministica (solo il logit massimo sopravvive)
    a = generate(m, [3, 4], 20, np.random.default_rng(1), top_k=1)
    b = generate(m, [3, 4], 20, np.random.default_rng(2), top_k=1)
    assert a == b  # indipendente dal seed: top_k=1 == greedy


def test_low_temperature_is_greedy():
    ds, cfg, m = _small()
    a = generate(m, [5], 20, np.random.default_rng(1), temperature=1e-6)
    b = generate(m, [5], 20, np.random.default_rng(9), temperature=1e-6)
    assert a == b


def test_save_load_roundtrip():
    ds, cfg, m = _small(seed=3)
    X, _ = ds.get_batch("train", cfg.block_size, 4, np.random.default_rng(0))
    before = m.logits(X).data.copy()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.npz")
        m.save(path, ds.tokenizer)
        m2, tok2 = GPT.load(path, np.random.default_rng(123))
    after = m2.logits(X).data
    assert np.allclose(before, after)              # stessi logit -> pesi identici
    assert tok2.chars == ds.tokenizer.chars         # vocabolario preservato


def test_short_training_decreases_loss():
    ds, cfg, m = _small()
    opt = AdamW(m.parameters(), lr=3e-3)
    rng = np.random.default_rng(0)
    losses = []
    for _ in range(40):
        X, Y = ds.get_batch("train", cfg.block_size, 16, rng)
        m.zero_grad()
        loss = m.loss(X, Y)
        loss.backward()
        opt.step()
        losses.append(loss.data.item())
    assert np.mean(losses[-8:]) < np.mean(losses[:8])


if __name__ == "__main__":
    from _runner import run

    run(globals())
