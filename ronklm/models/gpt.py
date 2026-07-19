"""RonkLM — il GPT completo (Fase 7).

Assembla tutto: token embedding + positional embedding -> stack di Block ->
LayerNorm finale -> proiezione al vocabolario. E' un transformer decoder-only, lo
stesso schema (in piccolo) di GPT-2.

    indici (B, T)
      -> token embedding (B,T,C)  +  positional embedding (T,C)      [somma]
      -> Block x n_layer                                             (B,T,C)
      -> LayerNorm finale                                            (B,T,C)
      -> Linear verso il vocab                                       (B,T,vocab) = logits

Concetto nuovo: il positional embedding. L'attention e' cieca all'ordine (Fase 5);
una seconda tabella indicizzata dalla POSIZIONE, sommata al token embedding, da' a
ogni posizione una firma appresa, cosi' il modello sa DOVE sono i token.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ronklm.autograd import Tensor, cross_entropy
from ronklm.models.block import Block
from ronklm.nn import Embedding, LayerNorm, Linear, Module
from ronklm.tokenizer import CharTokenizer


@dataclass
class GPTConfig:
    """Iperparametri architetturali del modello (salvati nel checkpoint)."""
    vocab_size: int
    block_size: int = 32
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64


class GPT(Module):
    """Transformer decoder-only char-level."""

    def __init__(self, config: GPTConfig, rng: np.random.Generator) -> None:
        self.config = config
        C = config.n_embd
        self.tok_emb = Embedding(config.vocab_size, C, rng)         # significato dei token
        self.pos_emb = Embedding(config.block_size, C, rng)         # posizione nella sequenza
        self.blocks = [
            Block(C, config.n_head, config.block_size, rng) for _ in range(config.n_layer)
        ]
        self.ln_f = LayerNorm(C)                                    # LayerNorm finale
        self.head = Linear(C, config.vocab_size, rng)              # logits

    def logits(self, idx: np.ndarray) -> Tensor:
        """Indici (B, T) -> logits (B, T, vocab)."""
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequenza {T} > block_size {self.config.block_size}"
        tok = self.tok_emb(idx)                       # (B, T, C)
        pos = self.pos_emb(np.arange(T))              # (T, C): firma di ogni posizione
        x = tok + pos                                 # (B, T, C): somma, broadcast su B
        for blk in self.blocks:
            x = blk(x)                                # (B, T, C)
        x = self.ln_f(x)
        return self.head(x)                           # (B, T, vocab)

    def loss(self, idx: np.ndarray, targets: np.ndarray) -> Tensor:
        """Cross-entropy su TUTTE le B*T posizioni (grazie alla maschera causale)."""
        B, T = idx.shape
        V = self.config.vocab_size
        logits = self.logits(idx)                     # (B, T, V)
        return cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))

    # --- salvataggio / caricamento ----------------------------------------
    def save(self, path: str | Path, tokenizer: CharTokenizer) -> None:
        """Salva pesi + config + vocabolario in un unico .npz autosufficiente.

        PERCHE' insieme: un modello char-level caricato con un vocabolario o una config
        diversi produce spazzatura deterministica (pesi giusti, mappa/shape sbagliate).
        Il checkpoint deve bastare a se stesso: pesi + mappa indici + architettura.
        """
        arrays = {f"p{i}": p.data for i, p in enumerate(self.parameters())}
        meta = {"config": asdict(self.config), "chars": tokenizer.chars}
        arrays["_meta"] = np.array([json.dumps(meta, ensure_ascii=False)])
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path, rng: np.random.Generator) -> tuple["GPT", CharTokenizer]:
        """Ricostruisce (modello, tokenizer) da un checkpoint salvato con save()."""
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["_meta"][0]))
        config = GPTConfig(**meta["config"])
        model = cls(config, rng)
        for i, p in enumerate(model.parameters()):
            p.data = data[f"p{i}"]
        tokenizer = CharTokenizer(meta["chars"])
        return model, tokenizer

    def num_params(self) -> int:
        return int(sum(p.data.size for p in self.parameters()))

    def __repr__(self) -> str:
        c = self.config
        return (
            f"GPT(layer={c.n_layer}, head={c.n_head}, embd={c.n_embd}, "
            f"block={c.block_size}, params={self.num_params():,})"
        )
