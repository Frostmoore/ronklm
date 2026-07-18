"""MLP — language model a contesto multi-carattere (Fase 4).

Supera il limite del bigram: invece di guardare un solo carattere, guarda gli ultimi
`block_size`. Architettura stile Bengio et al. 2003 (il primo LM neurale):

    contesto (B, T) di indici
      -> Embedding: ogni carattere -> vettore (B, T, n_embd)
      -> concatena i T vettori -> (B, T*n_embd)
      -> Linear + tanh (strato nascosto)      -> (B, n_hidden)
      -> Linear (testa)                        -> (B, vocab)  = logits del prossimo char

Concetti: embedding (compressione della "tabella impossibile" di Fase 1), strato
nascosto (rileva COMBINAZIONI di caratteri, non solo contributi indipendenti).
"""
from __future__ import annotations

import numpy as np

from ronklm.autograd import Tensor, cross_entropy
from ronklm.nn import Embedding, Linear, Module


class MLP(Module):
    """MLP a contesto fisso che predice il prossimo carattere.

    Attributi:
        block_size (int) : quanti caratteri di contesto.
        emb (Embedding)  : tabella (vocab, n_embd).
        h (Linear)       : strato nascosto (T*n_embd -> n_hidden).
        head (Linear)    : proiezione ai logits (n_hidden -> vocab).
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_hidden: int,
        rng: np.random.Generator,
    ) -> None:
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.emb = Embedding(vocab_size, n_embd, rng)
        self.h = Linear(block_size * n_embd, n_hidden, rng)
        self.head = Linear(n_hidden, vocab_size, rng)

    def logits(self, x_idx: np.ndarray) -> Tensor:
        """Contesto (B, T) di indici -> logits (B, vocab) sul prossimo carattere."""
        B, T = x_idx.shape
        e = self.emb(x_idx)                     # (B, T, n_embd)
        flat = e.reshape(B, T * self.n_embd)    # concatena i T embedding
        hidden = self.h(flat).tanh()            # (B, n_hidden), non-linearita'
        return self.head(hidden)                # (B, vocab)

    def loss(self, x_idx: np.ndarray, y_idx: np.ndarray) -> Tensor:
        """Cross-entropy tra i logits e i target (B,). Ritorna un Tensor (per backward)."""
        return cross_entropy(self.logits(x_idx), y_idx)

    @staticmethod
    def targets_from_batch(Y: np.ndarray) -> np.ndarray:
        """Dal batch (B, T) del Dataset, il target dell'MLP e' l'ultimo carattere di Y,
        cioe' il carattere che segue l'intero contesto X: Y[:, -1]."""
        return Y[:, -1]

    def nll(self, data: np.ndarray, block_size: int, rng: np.random.Generator, n_batches: int = 20, batch_size: int = 256) -> float:
        """Stima la NLL media su `data` mediando su piu' batch casuali (stabile)."""
        from ronklm.dataset import Dataset  # import locale per evitare cicli

        # costruiamo finestre direttamente dai dati (senza dipendere da Dataset.split)
        total = 0.0
        for _ in range(n_batches):
            ix = rng.integers(0, len(data) - block_size, size=batch_size)
            X = np.stack([data[i : i + block_size] for i in ix])
            Y = np.stack([data[i + 1 : i + 1 + block_size] for i in ix])
            total += self.loss(X, self.targets_from_batch(Y)).data.item()
        return total / n_batches

    def generate(self, rng: np.random.Generator, n: int, tokenizer=None, seed_ctx: np.ndarray | None = None) -> list[int]:
        """Genera n caratteri. Il contesto scorre: si tiene sempre gli ultimi block_size."""
        if seed_ctx is None:
            ctx = [0] * self.block_size  # contesto iniziale neutro (indice 0)
        else:
            ctx = list(seed_ctx[-self.block_size :])
        out: list[int] = []
        for _ in range(n):
            x = np.array(ctx[-self.block_size :], dtype=np.int64)[None, :]  # (1, T)
            probs = self.logits(x).softmax(axis=-1).data[0]                 # (vocab,)
            nxt = int(rng.choice(self.vocab_size, p=probs))
            out.append(nxt)
            ctx.append(nxt)
        return out

    def __repr__(self) -> str:
        return f"MLP(vocab={self.vocab_size}, block={self.block_size}, embd={self.n_embd})"
