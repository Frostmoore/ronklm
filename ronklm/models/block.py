"""Il blocco Transformer (Fase 6) — l'unita' che i GPT ripetono N volte.

Assembla il mattone che si impila in profondita'. Tre ingredienti chiave rendono
possibile lo stacking:
    - multi-head: piu' teste di attention in parallelo (sguardi diversi);
    - feed-forward: la parte che ELABORA dopo che l'attention ha COMUNICATO;
    - residual + LayerNorm (pre-norm): il motivo per cui reti profonde sono addestrabili.

Struttura (pre-norm):
    x = x + attn(ln1(x))     # comunica
    x = x + ffn(ln2(x))      # elabora
"""
from __future__ import annotations

import numpy as np

from ronklm.autograd import Tensor, cat
from ronklm.models.attention import Head
from ronklm.nn import LayerNorm, Linear, Module


class MultiHeadAttention(Module):
    """n_head teste di attention in parallelo, concatenate e riproiettate.

    PERCHE' piu' teste invece di una grande: una sola softmax da' UN solo "sguardo" di
    attenzione per posizione. Ma a una posizione possono servire informazioni diverse da
    posti diversi (il carattere precedente per l'ortografia, l'inizio parola per la
    morfologia, ...). Teste separate = sguardi paralleli specializzabili. Si divide n_embd
    per n_head (invece di moltiplicare i parametri) -> costo costante. La proiezione finale
    MESCOLA i contributi delle teste, altrimenti resterebbero segregati in fette separate.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, rng: np.random.Generator) -> None:
        assert n_embd % n_head == 0, "n_embd deve essere divisibile per n_head"
        head_size = n_embd // n_head
        self.heads = [Head(n_embd, head_size, block_size, rng) for _ in range(n_head)]
        self.proj = Linear(n_embd, n_embd, rng)

    def forward(self, x: Tensor) -> Tensor:
        outs = [h(x) for h in self.heads]   # n_head x (B, T, head_size)
        out = cat(outs, axis=-1)            # (B, T, n_embd)
        return self.proj(out)               # mescola le teste


class FeedForward(Module):
    """Rete a due strati applicata a ogni posizione indipendentemente (espansione 4x).

    PERCHE' serve, dopo l'attention: l'attention SPOSTA informazione tra posizioni ma la
    elabora poco (fa medie pesate). La FFN e' il complemento: non guarda altre posizioni,
    ma ELABORA -- con una vera non-linearita' (gelu) -- cio' che l'attention ha raccolto.
    Il ritmo del transformer: comunica (attn) -> pensa (ffn) -> comunica -> pensa.
    PERCHE' 4x: da' alla FFN uno spazio interno piu' largo dove computare prima di
    ricomprimere; e' la convenzione empirica di tutti i GPT.
    """

    def __init__(self, n_embd: int, rng: np.random.Generator) -> None:
        self.fc = Linear(n_embd, 4 * n_embd, rng)
        self.proj = Linear(4 * n_embd, n_embd, rng)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.fc(x).gelu())


class Block(Module):
    """Un blocco transformer pre-norm con connessioni residue.

        x = x + attn(ln1(x))     # comunica  (residual + pre-norm)
        x = x + ffn(ln2(x))      # elabora   (residual + pre-norm)

    PERCHE' le residual (x + ...): il backward della somma distribuisce il gradiente
    INVARIATO (Fase 3). Il ramo 'x' nudo e' un'autostrada su cui il gradiente arriva ai
    primi strati intatto, qualunque cosa facciano gli strati in mezzo -> reti profonde
    addestrabili. Inoltre ogni blocco parte dall'identita' (se attn/ffn ~0) e impara
    CORREZIONI incrementali a un segnale che scorre, invece di ricostruire tutto.
    PERCHE' pre-norm (LN dentro il ramo, prima di attn/ffn): cosi' l'autostrada 'x' resta
    intonsa da input a output; la post-norm (LN dopo la somma) disturberebbe il flusso del
    gradiente. GPT-2 e successori sono tutti pre-norm.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, rng: np.random.Generator) -> None:
        self.ln1 = LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, rng)
        self.ln2 = LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, rng)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
