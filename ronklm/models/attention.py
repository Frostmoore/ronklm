"""Self-attention causale, una testa, da zero (Fase 5).

Il cuore del transformer. L'idea in una frase: invece di un contesto rigido e
concatenato (MLP), OGNI posizione della sequenza decide da sola, dinamicamente, a
quali posizioni precedenti prestare attenzione e quanto -- con pesi di attenzione
CALCOLATI dai dati stessi a ogni forward.

Meccanica (spiegata a fondo in explain.md):
    ogni posizione emette query (che cerco), key (come mi faccio trovare),
    value (cosa consegno). L'affinita' query-key (prodotto scalare) decide quanto
    una posizione attinge dai value delle altre. La maschera causale impedisce di
    guardare il futuro; la softmax normalizza; l'output e' una media pesata dei value.
"""
from __future__ import annotations

import numpy as np

from ronklm.autograd import Tensor, cross_entropy
from ronklm.nn import Embedding, Linear, Module


class Head(Module):
    """Una testa di self-attention causale.

    Input:  x (B, T, n_embd)      -- una sequenza di T vettori per ognuno dei B esempi
    Output: (B, T, head_size)     -- per ogni posizione, una miscela pesata del passato

    Attributi:
        key, query, value (Linear) : le tre proiezioni (senza bias, come da prassi).
        mask (np.ndarray[bool])    : maschera causale (T, T), True sopra la diagonale.
        scale (float)              : 1/sqrt(head_size), lo scaling dei punteggi.
        last_att (np.ndarray|None) : ultima matrice di attenzione (per visualizzarla).
    """

    def __init__(self, n_embd: int, head_size: int, block_size: int, rng: np.random.Generator) -> None:
        self.head_size = head_size
        self.key = Linear(n_embd, head_size, rng, bias=False)
        self.query = Linear(n_embd, head_size, rng, bias=False)
        self.value = Linear(n_embd, head_size, rng, bias=False)
        # maschera causale: True nelle celle (t, s) con s > t (il "futuro"), da azzerare.
        self.mask = np.triu(np.ones((block_size, block_size), dtype=bool), k=1)
        self.scale = 1.0 / np.sqrt(head_size)
        self.last_att: np.ndarray | None = None

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.data.shape
        k = self.key(x)      # (B, T, head_size)
        q = self.query(x)    # (B, T, head_size)
        v = self.value(x)    # (B, T, head_size)

        # punteggi di affinita' q . k, scalati. q @ k^T: (B,T,H) @ (B,H,T) = (B,T,T).
        # PERCHE' lo scaling 1/sqrt(H): il prodotto scalare di vettori di dim H ha
        # varianza ~H; senza scaling i punteggi crescono con H e saturano la softmax
        # (gradiente ~0 -> Q,K non imparano). Dividere riporta la varianza a ~1.
        scores = (q @ k.transpose(1, 2)) * self.scale       # (B, T, T)

        # maschera causale: -inf sopra la diagonale, PRIMA della softmax, cosi' dopo
        # exp() quelle celle valgono 0 e la normalizzazione si redistribuisce da sola
        # sulle sole posizioni <= t. Impone la freccia del tempo: t non vede il futuro.
        scores = scores.masked_fill(self.mask[:T, :T], -1e9)

        att = scores.softmax(axis=-1)                       # (B, T, T), righe sommano a 1
        self.last_att = att.data                            # salvata per l'ispezione
        out = att @ v                                       # (B, T, head_size): media pesata dei value
        return out


class AttentionLM(Module):
    """Mini language model con UNA testa di attention, per testare la Fase 5 end-to-end.

    token embedding -> Head -> Linear verso il vocab. Predice il prossimo carattere a
    OGNI posizione (loss su tutte le T posizioni, grazie alla maschera causale).
    NB: senza positional embedding (arriva in Fase 7) l'attention e' "cieca all'ordine":
    va bene per verificare il meccanismo, non per battere l'MLP.
    """

    def __init__(self, vocab_size: int, block_size: int, n_embd: int, rng: np.random.Generator) -> None:
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.tok = Embedding(vocab_size, n_embd, rng)
        self.head = Head(n_embd, n_embd, block_size, rng)
        self.lm_head = Linear(n_embd, vocab_size, rng)

    def logits(self, x_idx: np.ndarray) -> Tensor:
        x = self.tok(x_idx)             # (B, T, n_embd)
        h = self.head(x)                # (B, T, n_embd)
        return self.lm_head(h)          # (B, T, vocab)

    def loss(self, x_idx: np.ndarray, y_idx: np.ndarray) -> Tensor:
        B, T = x_idx.shape
        logits = self.logits(x_idx)                          # (B, T, vocab)
        flat = logits.reshape(B * T, self.vocab_size)        # (B*T, vocab)
        return cross_entropy(flat, y_idx.reshape(B * T))     # media su tutte le posizioni

    def __repr__(self) -> str:
        return f"AttentionLM(vocab={self.vocab_size}, block={self.block_size})"
