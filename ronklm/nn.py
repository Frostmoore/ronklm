"""nn.py — layer riusabili costruiti su ronkgrad (Fase 4+).

Sopra il motore di autograd mettiamo i mattoni con cui si assemblano le reti: una
classe base `Module` che sa raccogliere i propri parametri (ricorsivamente), e i layer
`Linear` ed `Embedding`. E' l'equivalente di torch.nn: ogni classe qui corrisponde a
qualcosa che PyTorch offre, ma noi ne vediamo l'interno.

PERCHE' l'astrazione Module. Il GPT finale avra' decine di sotto-componenti annidati,
ognuno coi suoi pesi. L'ottimizzatore ha bisogno della LISTA COMPLETA dei parametri;
raccoglierla a mano e' il modo di dimenticarne uno (che quindi non si addestra mai --
bug silenzioso). `Module.parameters()` ricorsivo risolve il problema una volta per tutte.
"""
from __future__ import annotations

import numpy as np

from ronklm.autograd import Tensor


class Module:
    """Classe base: raccoglie parametri e sotto-moduli per introspezione di __dict__."""

    def parameters(self) -> list[Tensor]:
        """Tutti i Tensor-parametro di questo modulo e dei sotto-moduli (ricorsivo)."""
        params: list[Tensor] = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor):
                params.append(v)
            elif isinstance(v, Module):
                params += v.parameters()
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        params += item.parameters()
                    elif isinstance(item, Tensor):
                        params.append(item)
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):  # da sovrascrivere
        raise NotImplementedError


class Linear(Module):
    """Strato lineare: y = x @ W + b.

    Inizializzazione dei pesi ~ 1/sqrt(n_in) (Kaiming/LeCun). PERCHE': l'output di un
    neurone e' la somma di n_in termini; se ogni peso avesse varianza fissa, la varianza
    della somma crescerebbe con n_in, le attivazioni esploderebbero e le non-linearita'
    saturerebbero (gradiente ~0, il segnale muore). Con lo scaling 1/sqrt(n_in) la
    varianza dell'output resta ~1 indipendentemente dalla larghezza.
    """

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator, bias: bool = True) -> None:
        scale = 1.0 / np.sqrt(n_in)
        self.W = Tensor(rng.normal(0.0, scale, size=(n_in, n_out)))
        self.b = Tensor(np.zeros(n_out)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.W
        if self.b is not None:
            out = out + self.b  # broadcast di (n_out,) su (..., n_out)
        return out


class Embedding(Module):
    """Tabella di embedding: a ogni indice (0..num-1) associa un vettore di `dim` numeri
    ADDESTRABILI. forward(idx) seleziona le righe corrispondenti.

    E' la stessa "selezione di riga" osservata in Fase 2 (onehot @ W), qui resa
    ufficiale ed efficiente via gather_rows. Compressione: 69 caratteri descritti da
    vettori corti invece che da one-hot lunghi 69; e il training puo' avvicinare tra
    loro i vettori di caratteri simili (geometria della somiglianza).
    """

    def __init__(self, num: int, dim: int, rng: np.random.Generator, std: float = 1.0) -> None:
        # init ~N(0, std): gli embedding non sono somme, non serve lo scaling di Linear;
        # std~1 va bene, il training li sistema.
        self.weight = Tensor(rng.normal(0.0, std, size=(num, dim)))

    def forward(self, idx: np.ndarray) -> Tensor:
        return self.weight.gather_rows(idx)


class LayerNorm(Module):
    """Normalizza ogni vettore (ultima dimensione) a media 0 e varianza 1, poi lo
    riscala con due parametri APPRESI: gamma (guadagno) e beta (offset).

    PERCHE' normalizzare: in una rete profonda la scala delle attivazioni di uno strato
    dipende da tutti i precedenti, che stanno cambiando durante il training -> scale che
    esplodono/collassano. LayerNorm ristabilisce a ogni blocco un riferimento fisso ->
    training stabile e learning rate piu' alti.
    PERCHE' gamma e beta: la normalizzazione pura toglie alla rete anche la liberta' di
    VOLERE una scala diversa; i due parametri gliela restituiscono (default sano: gamma=1,
    beta=0). PERCHE' LayerNorm e non BatchNorm: LayerNorm normalizza ogni posizione per
    conto suo (nessun accoppiamento tra esempi, identica in training e generazione) ->
    ideale per le sequenze. Tutta l'operazione e' composta da primitivi: backward automatico.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        self.gamma = Tensor(np.ones(dim))
        self.beta = Tensor(np.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        xhat = (x - mu) / ((var + self.eps) ** 0.5)  # standardizza l'ultima dimensione
        return xhat * self.gamma + self.beta          # riscala/trasla (parametri appresi)

