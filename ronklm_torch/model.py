"""RonkLM in PyTorch — architettura IDENTICA a ronklm/models/gpt.py (Fase 9.1).

Ogni classe qui rispecchia una classe del Percorso A, riga per riga, con gli stessi
identici dettagli numerici (scaling 1/sqrt(H), maschera causale riempita con -1e9,
gelu in approssimazione tanh, LayerNorm con eps=1e-5). Questo e' cio' che rende
possibile il test di equivalenza della Fase 9.2: se un dettaglio divergesse, il test
lo griderebbe.

DETTAGLIO CRUCIALE — la trasposizione dei pesi lineari:
    il NOSTRO Linear tiene W di forma (n_in, n_out) e calcola  x @ W + b
    il Linear di PyTorch tiene weight di forma (n_out, n_in) e calcola x @ weight.T + b
Quindi trasferire i pesi richiede una TRASPOSIZIONE. E' esattamente il tipo di
dettaglio che "non crasha, degrada soltanto" e che il test di equivalenza cattura.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Valore usato per mascherare il futuro PRIMA della softmax. Deve combaciare con
# quello del Percorso A (-1e9, non -inf) perche' i due motori diano gli stessi numeri.
MASK_VALUE = -1e9


@dataclass
class GPTConfig:
    """Stessi campi di ronklm.models.gpt.GPTConfig."""
    vocab_size: int
    block_size: int = 32
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64


class Head(nn.Module):
    """Una testa di self-attention causale. Specchio di ronklm/models/attention.py."""

    def __init__(self, n_embd: int, head_size: int, block_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.scale = head_size ** -0.5
        # buffer (non parametro): la maschera causale, True sopra la diagonale
        self.register_buffer(
            "mask", torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        scores = (q @ k.transpose(-2, -1)) * self.scale          # (B, T, T)
        scores = scores.masked_fill(self.mask[:T, :T], MASK_VALUE)
        att = scores.softmax(dim=-1)
        return att @ v                                           # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """n_head teste in parallelo, concatenate e riproiettate."""

    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class CausalSelfAttention(nn.Module):
    """Multi-head attention OTTIMIZZATA — matematicamente identica a MultiHeadAttention,
    ma scritta come la scrivono i GPT veri. Tre differenze, tutte di sola efficienza:

    1. **QKV fuso**: una sola `nn.Linear(n_embd, 3*n_embd)` invece di 3 Linear separate
       per ognuna delle n_head teste (da 3*n_head piccole matmul a UNA grande).
    2. **Teste in parallelo via reshape**: invece di un ciclo Python sulle teste, si
       ricompone il tensore in (B, n_head, T, head_size) e si calcola tutto insieme.
    3. **FlashAttention** (`F.scaled_dot_product_attention`): non materializza mai la
       matrice di attenzione T x T -- la calcola a blocchi nella memoria veloce della
       GPU. E' cio' che fa crollare l'uso di VRAM (la matrice T x T per ogni testa e
       ogni layer era il vero costo).

    Il risultato numerico e' lo STESSO (il test di equivalenza lo verifica): il
    mascheramento causale con -inf e lo scaling 1/sqrt(head_size) sono quelli di prima.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)  # bias=False come le nostre k/q/v
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)                       # 3 x (B, T, C)
        # (B, T, C) -> (B, n_head, T, head_size): le teste diventano una dimensione di batch
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # scaling e maschera inclusi
        y = y.transpose(1, 2).contiguous().view(B, T, C)            # ri-concatena le teste
        return self.proj(y)


class FeedForward(nn.Module):
    """Espansione 4x + gelu + ricompressione. Nota: approximate='tanh' per combaciare
    con la nostra gelu composita del Percorso A."""

    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    """Blocco pre-norm con connessioni residue (specchio di ronklm/models/block.py).

    `fast=True` usa CausalSelfAttention (fusa + FlashAttention): stessa matematica,
    molta meno VRAM e molto piu' veloce. `fast=False` usa la versione "didattica",
    testa per testa, identica al Percorso A.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, fast: bool = False) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd, eps=1e-5)
        self.attn = (
            CausalSelfAttention(n_embd, n_head, block_size) if fast
            else MultiHeadAttention(n_embd, n_head, block_size)
        )
        self.ln2 = nn.LayerNorm(n_embd, eps=1e-5)
        self.ffn = FeedForward(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    """Il GPT completo, specchio di ronklm/models/gpt.py."""

    def __init__(self, config: GPTConfig, fast: bool = False) -> None:
        super().__init__()
        self.config = config
        self.fast = fast
        C = config.n_embd
        self.tok_emb = nn.Embedding(config.vocab_size, C)
        self.pos_emb = nn.Embedding(config.block_size, C)
        self.blocks = nn.ModuleList(
            [Block(C, config.n_head, config.block_size, fast=fast) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(C, eps=1e-5)
        self.head = nn.Linear(C, config.vocab_size)

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        """Indici (B, T) -> logits (B, T, vocab)."""
        B, T = idx.shape
        assert T <= self.config.block_size
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)     # (B, T, C)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Ritorna (logits, loss). loss=None se non passi i target."""
        lg = self.logits(idx)
        if targets is None:
            return lg, None
        B, T = idx.shape
        loss = F.cross_entropy(lg.reshape(B * T, self.config.vocab_size), targets.reshape(B * T))
        return lg, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        c = self.config
        return (
            f"GPT-torch(layer={c.n_layer}, head={c.n_head}, embd={c.n_embd}, "
            f"block={c.block_size}, params={self.num_params():,})"
        )


# --------------------------------------------------------------------------
# Trasferimento dei pesi dal modello NumPy (riferimento) a questo modello torch
# --------------------------------------------------------------------------
def load_weights_from_numpy(torch_model: GPT, np_model) -> None:
    """Copia i pesi da un `ronklm.models.gpt.GPT` (NumPy) dentro `torch_model`.

    E' la funzione che rende possibile il test di equivalenza: stessi pesi, stesso
    input -> devono uscire gli stessi logit. Attenzione alle TRASPOSIZIONI dei Linear
    (vedi docstring del modulo).
    """
    import numpy as np

    def copy_(dst: torch.Tensor, src: np.ndarray) -> None:
        assert tuple(dst.shape) == tuple(src.shape), f"shape mismatch: {dst.shape} vs {src.shape}"
        with torch.no_grad():
            dst.copy_(torch.from_numpy(np.ascontiguousarray(src)).to(dst.dtype))

    def copy_linear(dst_lin: nn.Linear, src_lin) -> None:
        # nostro W (in, out)  ->  torch weight (out, in): trasposizione obbligatoria
        copy_(dst_lin.weight, src_lin.W.data.T)
        if dst_lin.bias is not None:
            copy_(dst_lin.bias, src_lin.b.data)

    def copy_ln(dst_ln: nn.LayerNorm, src_ln) -> None:
        copy_(dst_ln.weight, src_ln.gamma.data)
        copy_(dst_ln.bias, src_ln.beta.data)

    copy_(torch_model.tok_emb.weight, np_model.tok_emb.weight.data)
    copy_(torch_model.pos_emb.weight, np_model.pos_emb.weight.data)

    for tb, nb in zip(torch_model.blocks, np_model.blocks):
        copy_ln(tb.ln1, nb.ln1)
        if isinstance(tb.attn, CausalSelfAttention):
            # Layout FUSO: qkv.weight ha forma (3C, C); le righe sono
            #   [0:C]   = query di tutte le teste, concatenate nell'ordine delle teste
            #   [C:2C]  = key    "        "
            #   [2C:3C] = value  "        "
            # Ogni nostra W e' (n_embd, head_size) -> serve .T per avere (head_size, n_embd).
            q = np.concatenate([h.query.W.data.T for h in nb.attn.heads], axis=0)
            k = np.concatenate([h.key.W.data.T for h in nb.attn.heads], axis=0)
            v = np.concatenate([h.value.W.data.T for h in nb.attn.heads], axis=0)
            copy_(tb.attn.qkv.weight, np.concatenate([q, k, v], axis=0))
        else:
            for th, nh in zip(tb.attn.heads, nb.attn.heads):
                copy_linear(th.key, nh.key)
                copy_linear(th.query, nh.query)
                copy_linear(th.value, nh.value)
        copy_linear(tb.attn.proj, nb.attn.proj)
        copy_ln(tb.ln2, nb.ln2)
        copy_linear(tb.ffn.fc, nb.ffn.fc)
        copy_linear(tb.ffn.proj, nb.ffn.proj)

    copy_ln(torch_model.ln_f, np_model.ln_f)
    copy_linear(torch_model.head, np_model.head)


def config_from_numpy(np_config) -> GPTConfig:
    """Costruisce la GPTConfig torch da quella NumPy (stessi campi)."""
    return GPTConfig(
        vocab_size=np_config.vocab_size,
        block_size=np_config.block_size,
        n_layer=np_config.n_layer,
        n_head=np_config.n_head,
        n_embd=np_config.n_embd,
    )
