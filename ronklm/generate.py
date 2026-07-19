"""Generazione autoregressiva di testo da un GPT (Fase 7).

Un carattere alla volta: forward sugli ultimi block_size token, si prende la
distribuzione dell'ULTIMA posizione, si applica temperature e top-k, si campiona, si
appende, si ripete. E' esattamente il modo in cui i LLM "scrivono".
"""
from __future__ import annotations

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def generate(
    model,
    context: list[int],
    max_new_tokens: int,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> list[int]:
    """Genera `max_new_tokens` indici a partire da `context` (lista di indici).

    temperature: divide i logit prima della softmax. <1 = piu' conservativo/ripetitivo,
        >1 = piu' vario/sgangherato, ->0 = greedy (argmax). Non cambia l'ordine delle
        preferenze, solo quanto ci si azzarda a deviare dalla prima scelta.
    top_k: se dato, tiene solo i k logit migliori (azzera gli altri a -inf) prima di
        campionare. Taglia la coda di caratteri assurdi che, se pescati, farebbero
        deragliare tutto il seguito (errore composto: testo fuori distribuzione).
    """
    block = model.config.block_size
    ctx = list(context)
    out: list[int] = []
    for _ in range(max_new_tokens):
        # si tronca agli ultimi block_size token: oltre, il modello non e' definito
        # (la tabella posizionale ha block_size righe). E' il "limite di contesto".
        window = np.array(ctx[-block:], dtype=np.int64)[None, :]      # (1, t)
        logits = model.logits(window).data[0, -1]                    # (vocab,): ultima posizione
        logits = logits / max(temperature, 1e-8)

        if top_k is not None:
            k = min(top_k, logits.shape[0])
            kth = np.sort(logits)[-k]                                 # k-esimo valore piu' grande
            logits = np.where(logits < kth, -np.inf, logits)         # azzera la coda

        probs = _softmax(logits)
        nxt = int(rng.choice(len(probs), p=probs))
        ctx.append(nxt)
        out.append(nxt)
    return out
