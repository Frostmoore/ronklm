"""optim.py — ottimizzatori (Fase 4).

Un ottimizzatore prende la lista dei parametri e, dato il loro gradiente (gia'
calcolato da backward()), decide come aggiornarli. Qui: SGD (il passo puro) e AdamW
(lo stesso usato per addestrare i GPT reali), scritti a mano.
"""
from __future__ import annotations

import numpy as np

from ronklm.autograd import Tensor


class SGD:
    """Discesa del gradiente stocastica pura: p -= lr * grad. Il passo essenziale."""

    def __init__(self, params: list[Tensor], lr: float) -> None:
        self.params = list(params)
        self.lr = lr

    def step(self) -> None:
        for p in self.params:
            p.data -= self.lr * p.grad

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()


class AdamW:
    """AdamW: SGD + momento + scaling adattivo per-parametro + weight decay disaccoppiato.

    Pezzo per pezzo (dettagli in explain.md):
      - m (primo momento): media mobile dei gradienti -> filtra il rumore dei minibatch
        e accumula velocita' nelle direzioni costanti (come una palla che rotola).
      - v (secondo momento): media mobile dei gradienti al quadrato -> ogni parametro
        ottiene di fatto un learning rate su misura (chi riceve gradienti grandi fa
        passi piu' piccoli, e viceversa).
      - bias-correction (/(1-beta^t)): m e v partono da 0, quindi nei primi passi sono
        sottostimati; la correzione compensa questo transitorio.
      - weight decay DISACCOPPIATO (la 'W' di AdamW): spinge i pesi verso 0 fuori dal
        meccanismo adattivo (regolarizzazione), a differenza di Adam classico dove
        finiva dentro il gradiente e veniva ri-scalato.
    """

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 3e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)   # bias-correction
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            # passo adattivo + weight decay disaccoppiato (applicato direttamente ai pesi)
            p.data -= self.lr * (mhat / (np.sqrt(vhat) + self.eps) + self.wd * p.data)

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()
