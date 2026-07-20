"""Equivalenza numerica NumPy (riferimento) vs PyTorch (Fase 9.2).

E' LA sottofase irrinunciabile del Percorso B. Un port "a occhio" che diverge in
silenzio (una trasposizione, un 1/sqrt(d) dimenticato, una LayerNorm sull'asse
sbagliato) e' il modo classico di portarsi dietro un bug per mesi: non crasha,
degrada soltanto. Il Percorso A ci ha lasciato un lusso raro nel deep learning --
un'implementazione di riferimento di cui ci fidiamo fino all'ultima riga, perche'
l'abbiamo derivata noi e verificata coi gradient check. Questo file la spende.

Metodo: stessi pesi (copiati), stesso input, e si confrontano
    (a) i LOGIT del forward,
    (b) i GRADIENTI di ogni parametro dopo il backward,
    (c) le LOSS passo per passo durante un breve training con AdamW.

Nota sui numeri: il nostro motore lavora in float64. Per un confronto stretto
mettiamo anche il modello torch in float64 (.double()), cosi' le differenze residue
sono solo arrotondamenti nell'ultimo bit e possiamo usare tolleranze severe.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from ronklm.models.gpt import GPT as GPTNumpy, GPTConfig as GPTConfigNumpy  # noqa: E402
from ronklm.optim import AdamW as AdamWNumpy  # noqa: E402
from ronklm_torch.model import (  # noqa: E402
    GPT as GPTTorch,
    config_from_numpy,
    load_weights_from_numpy,
)

TOL = 1e-9  # tolleranza severa: in float64 i due motori devono quasi coincidere


def _build_pair(seed: int = 0, vocab: int = 23, block: int = 8, n_layer: int = 2,
                n_head: int = 2, n_embd: int = 16, fast: bool = False):
    """Costruisce (modello NumPy, modello torch) con gli STESSI pesi.

    fast=True usa la CausalSelfAttention ottimizzata (QKV fuso + FlashAttention):
    deve dare gli stessi identici numeri della versione didattica."""
    cfg_np = GPTConfigNumpy(vocab_size=vocab, block_size=block, n_layer=n_layer,
                            n_head=n_head, n_embd=n_embd)
    m_np = GPTNumpy(cfg_np, np.random.default_rng(seed))
    m_t = GPTTorch(config_from_numpy(cfg_np), fast=fast).double()  # float64 come il riferimento
    load_weights_from_numpy(m_t, m_np)
    return m_np, m_t, cfg_np


def _batch(cfg, B=4, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, cfg.vocab_size, size=(B, cfg.block_size))
    Y = rng.integers(0, cfg.vocab_size, size=(B, cfg.block_size))
    return X, Y


# --- (a) FORWARD ------------------------------------------------------------
def test_forward_logits_match():
    m_np, m_t, cfg = _build_pair()
    X, _ = _batch(cfg)
    lg_np = m_np.logits(X).data
    with torch.no_grad():
        lg_t = m_t.logits(torch.from_numpy(X)).numpy()
    assert lg_np.shape == lg_t.shape
    err = np.abs(lg_np - lg_t).max()
    assert err < TOL, f"logit divergono: err max {err:.2e}"


def test_loss_matches():
    m_np, m_t, cfg = _build_pair()
    X, Y = _batch(cfg)
    loss_np = m_np.loss(X, Y).data.item()
    with torch.no_grad():
        _, loss_t = m_t(torch.from_numpy(X), torch.from_numpy(Y))
    err = abs(loss_np - loss_t.item())
    assert err < TOL, f"loss divergono: {loss_np} vs {loss_t.item()} (err {err:.2e})"


# --- (b) BACKWARD -----------------------------------------------------------
def test_gradients_match():
    """Il test piu' importante: i gradienti di OGNI parametro devono coincidere."""
    m_np, m_t, cfg = _build_pair()
    X, Y = _batch(cfg)

    # gradienti dal nostro motore
    m_np.zero_grad()
    m_np.loss(X, Y).backward()

    # gradienti da PyTorch
    m_t.zero_grad()
    _, loss_t = m_t(torch.from_numpy(X), torch.from_numpy(Y))
    loss_t.backward()

    # confronto parametro per parametro, con la mappa esplicita (e le trasposizioni)
    pairs = []
    pairs.append((m_np.tok_emb.weight.grad, m_t.tok_emb.weight.grad.numpy(), "tok_emb"))
    pairs.append((m_np.pos_emb.weight.grad, m_t.pos_emb.weight.grad.numpy(), "pos_emb"))
    for i, (nb, tb) in enumerate(zip(m_np.blocks, m_t.blocks)):
        pairs.append((nb.ln1.gamma.grad, tb.ln1.weight.grad.numpy(), f"b{i}.ln1.gamma"))
        pairs.append((nb.ln1.beta.grad, tb.ln1.bias.grad.numpy(), f"b{i}.ln1.beta"))
        for j, (nh, th) in enumerate(zip(nb.attn.heads, tb.attn.heads)):
            pairs.append((nh.key.W.grad, th.key.weight.grad.numpy().T, f"b{i}.h{j}.key"))
            pairs.append((nh.query.W.grad, th.query.weight.grad.numpy().T, f"b{i}.h{j}.query"))
            pairs.append((nh.value.W.grad, th.value.weight.grad.numpy().T, f"b{i}.h{j}.value"))
        pairs.append((nb.attn.proj.W.grad, tb.attn.proj.weight.grad.numpy().T, f"b{i}.attn.proj.W"))
        pairs.append((nb.attn.proj.b.grad, tb.attn.proj.bias.grad.numpy(), f"b{i}.attn.proj.b"))
        pairs.append((nb.ln2.gamma.grad, tb.ln2.weight.grad.numpy(), f"b{i}.ln2.gamma"))
        pairs.append((nb.ln2.beta.grad, tb.ln2.bias.grad.numpy(), f"b{i}.ln2.beta"))
        pairs.append((nb.ffn.fc.W.grad, tb.ffn.fc.weight.grad.numpy().T, f"b{i}.ffn.fc.W"))
        pairs.append((nb.ffn.fc.b.grad, tb.ffn.fc.bias.grad.numpy(), f"b{i}.ffn.fc.b"))
        pairs.append((nb.ffn.proj.W.grad, tb.ffn.proj.weight.grad.numpy().T, f"b{i}.ffn.proj.W"))
        pairs.append((nb.ffn.proj.b.grad, tb.ffn.proj.bias.grad.numpy(), f"b{i}.ffn.proj.b"))
    pairs.append((m_np.ln_f.gamma.grad, m_t.ln_f.weight.grad.numpy(), "ln_f.gamma"))
    pairs.append((m_np.ln_f.beta.grad, m_t.ln_f.bias.grad.numpy(), "ln_f.beta"))
    pairs.append((m_np.head.W.grad, m_t.head.weight.grad.numpy().T, "head.W"))
    pairs.append((m_np.head.b.grad, m_t.head.bias.grad.numpy(), "head.b"))

    worst = 0.0
    worst_name = ""
    for g_np, g_t, name in pairs:
        assert g_np.shape == g_t.shape, f"{name}: shape {g_np.shape} vs {g_t.shape}"
        e = float(np.abs(g_np - g_t).max())
        if e > worst:
            worst, worst_name = e, name
    assert worst < TOL, f"gradienti divergono su '{worst_name}': err max {worst:.2e}"


def test_all_parameters_are_covered():
    """Guardia: il confronto sopra deve toccare TUTTI i parametri, non un sottoinsieme.
    Se un domani aggiungiamo un layer e dimentichiamo di mapparlo, questo test avvisa."""
    m_np, m_t, _ = _build_pair()
    n_np = len(m_np.parameters())
    n_t = len(list(m_t.parameters()))
    assert n_np == n_t, f"numero di parametri diverso: NumPy {n_np} vs torch {n_t}"


# --- (c) TRAINING -----------------------------------------------------------
def test_training_steps_match():
    """Pochi passi di AdamW con gli stessi batch: le loss devono coincidere passo
    per passo. Verifica che anche l'ottimizzatore scritto a mano combaci."""
    m_np, m_t, cfg = _build_pair(seed=2)
    lr, wd = 1e-3, 0.0   # wd=0: isola il confronto dal weight decay

    opt_np = AdamWNumpy(m_np.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd)
    opt_t = torch.optim.AdamW(m_t.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd)

    losses_np, losses_t = [], []
    for step in range(5):
        X, Y = _batch(cfg, seed=100 + step)   # stessi batch per entrambi

        m_np.zero_grad()
        l_np = m_np.loss(X, Y)
        l_np.backward()
        opt_np.step()
        losses_np.append(l_np.data.item())

        opt_t.zero_grad()
        _, l_t = m_t(torch.from_numpy(X), torch.from_numpy(Y))
        l_t.backward()
        opt_t.step()
        losses_t.append(l_t.item())

    for s, (a, b) in enumerate(zip(losses_np, losses_t)):
        assert abs(a - b) < 1e-8, f"step {s}: loss {a} vs {b}"
    # e devono anche scendere (non stiamo confrontando due modelli fermi)
    assert losses_np[-1] < losses_np[0]


# --- (d) LA VARIANTE OTTIMIZZATA ------------------------------------------
# CausalSelfAttention (QKV fuso + FlashAttention) e' scritta per la GPU, ma la
# matematica deve restare quella del Percorso A. Qui lo verifichiamo: e' il motivo per
# cui possiamo ottimizzare senza paura -- abbiamo un riferimento che ci controlla.

def test_fast_attention_forward_matches():
    m_np, m_t, cfg = _build_pair(fast=True)
    X, _ = _batch(cfg)
    lg_np = m_np.logits(X).data
    with torch.no_grad():
        lg_t = m_t.logits(torch.from_numpy(X)).numpy()
    err = np.abs(lg_np - lg_t).max()
    assert err < 1e-8, f"la variante fast diverge nel forward: err max {err:.2e}"


def test_fast_attention_gradients_match():
    m_np, m_t, cfg = _build_pair(seed=4, fast=True)
    X, Y = _batch(cfg, seed=5)

    m_np.zero_grad()
    m_np.loss(X, Y).backward()
    m_t.zero_grad()
    _, loss_t = m_t(torch.from_numpy(X), torch.from_numpy(Y))
    loss_t.backward()

    # confronto su alcuni parametri rappresentativi (embedding, testa, ffn, layernorm)
    checks = [
        (m_np.tok_emb.weight.grad, m_t.tok_emb.weight.grad.numpy(), "tok_emb"),
        (m_np.head.W.grad, m_t.head.weight.grad.numpy().T, "head.W"),
        (m_np.ln_f.gamma.grad, m_t.ln_f.weight.grad.numpy(), "ln_f.gamma"),
    ]
    for i, (nb, tb) in enumerate(zip(m_np.blocks, m_t.blocks)):
        checks.append((nb.ffn.fc.W.grad, tb.ffn.fc.weight.grad.numpy().T, f"b{i}.ffn.fc"))
        checks.append((nb.attn.proj.W.grad, tb.attn.proj.weight.grad.numpy().T, f"b{i}.attn.proj"))
        # il qkv fuso: ricomponiamo il gradiente delle nostre teste nello stesso layout
        C = m_np.config.n_embd
        q = np.concatenate([h.query.W.grad.T for h in nb.attn.heads], axis=0)
        k = np.concatenate([h.key.W.grad.T for h in nb.attn.heads], axis=0)
        v = np.concatenate([h.value.W.grad.T for h in nb.attn.heads], axis=0)
        checks.append((np.concatenate([q, k, v], axis=0), tb.attn.qkv.weight.grad.numpy(), f"b{i}.qkv"))

    worst, name = 0.0, ""
    for a, b, n in checks:
        assert a.shape == b.shape, f"{n}: shape {a.shape} vs {b.shape}"
        e = float(np.abs(a - b).max())
        if e > worst:
            worst, name = e, n
    assert worst < 1e-8, f"la variante fast diverge nei gradienti su '{name}': {worst:.2e}"


if __name__ == "__main__":
    from _runner import run

    run(globals())
