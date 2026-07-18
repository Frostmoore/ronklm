"""ronkgrad — un micro-motore di differenziazione automatica (Fase 3).

E' un mini-PyTorch tensoriale in ~300 righe. Registra le operazioni mentre le esegui
(costruendo un "grafo computazionale"), e poi calcola i gradienti di una loss rispetto
a QUALSIASI tensore percorrendo il grafo all'indietro e applicando la regola della
catena. Capito questo, PyTorch non ha piu' segreti strutturali: fa esattamente questo,
solo con ingegneria (C++/CUDA) migliore.

Idea centrale:
    Ogni Tensor ricorda da quali tensori e' nato (._prev) e una funzione (._backward)
    che sa spingere il gradiente dal proprio output ai propri input. backward() ordina
    il grafo (ordinamento topologico) e chiama i ._backward in ordine inverso.

Regole d'oro (spiegate in explain.md):
    - i gradienti si ACCUMULANO (+=), perche' un tensore usato piu' volte riceve un
      contributo da ogni uso (regola della derivata totale);
    - il broadcasting va "disfatto" all'indietro sommando lungo le dimensioni che erano
      state replicate (funzione _unbroadcast): il punto tecnicamente piu' insidioso.
"""
from __future__ import annotations

import math

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Riporta `grad` alla forma `shape`, sommando lungo le dimensioni che il forward
    aveva espanso col broadcasting.

    Esempio: se nel forward un vettore (V,) e' stato sommato a una matrice (B, V), nel
    backward ha ricevuto un gradiente (B, V); ma il vettore originale ha forma (V,),
    quindi i B contributi vanno SOMMATI: grad.sum(axis=0). Dimenticarlo produce
    gradienti di forma sbagliata (crash: bug fortunato) o silenziosamente errati.
    """
    # 1. elimina le dimensioni in piu' davanti (grad ha piu' assi dell'originale)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # 2. somma lungo gli assi che nell'originale erano di taglia 1 (broadcastati)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    """Un array NumPy che sa differenziarsi.

    Attributi:
        data (np.ndarray[float64]) : i valori.
        grad (np.ndarray[float64]) : il gradiente accumulato della loss rispetto a data.
        _backward (callable)       : spinge il gradiente ai genitori (no-op sui foglia).
        _prev (tuple[Tensor])      : i tensori da cui questo e' stato prodotto.
        _op (str)                  : etichetta dell'operazione (solo per debug).
    """

    def __init__(self, data, _prev: tuple = (), _op: str = "") -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = _prev
        self._op = _op

    # ---- utilita' ---------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    def _wrap(self, other) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other)

    # ---- somma / sottrazione / negazione ----------------------------------
    def __add__(self, other) -> "Tensor":
        other = self._wrap(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            # la somma DISTRIBUISCE il gradiente invariato a entrambi gli addendi
            # (poi _unbroadcast lo riporta alla forma di ciascuno). Questo semplice
            # fatto sara' la ragione per cui le connessioni residue (Fase 6) funzionano.
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other) -> "Tensor":
        return self + other

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other) -> "Tensor":
        return self + (-self._wrap(other))

    def __rsub__(self, other) -> "Tensor":
        return (-self) + other

    # ---- prodotto elemento-per-elemento -----------------------------------
    def __mul__(self, other) -> "Tensor":
        other = self._wrap(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            # regola del prodotto: ognuno riceve il gradiente moltiplicato per l'ALTRO
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other) -> "Tensor":
        return self * other

    # ---- potenza (esponente scalare) --------------------------------------
    def __pow__(self, p: float) -> "Tensor":
        assert isinstance(p, (int, float)), "esponente scalare"
        out = Tensor(self.data ** p, (self,), f"**{p}")

        def _backward():
            self.grad += (p * self.data ** (p - 1)) * out.grad

        out._backward = _backward
        return out

    # ---- divisione --------------------------------------------------------
    def __truediv__(self, other) -> "Tensor":
        if isinstance(other, Tensor):
            return self * other ** -1
        return self * (1.0 / other)

    def __rtruediv__(self, other) -> "Tensor":
        return (self ** -1) * other

    # ---- prodotto matriciale ----------------------------------------------
    def __matmul__(self, other) -> "Tensor":
        other = self._wrap(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            # Regole standard del gradiente del matmul C = A @ B:
            #   dA = dC @ B^T ,  dB = A^T @ dC
            # (le derivazioni sono in explain.md; il controllo mnemonico e' che le
            # SHAPE devono tornare). swapaxes(-1,-2) traspone le ultime due dimensioni,
            # cosi' funziona anche col matmul "a batch" (3D) di Fase 5.
            ga = out.grad @ np.swapaxes(other.data, -1, -2)
            gb = np.swapaxes(self.data, -1, -2) @ out.grad
            self.grad += _unbroadcast(ga, self.data.shape)
            other.grad += _unbroadcast(gb, other.data.shape)

        out._backward = _backward
        return out

    # ---- riduzioni --------------------------------------------------------
    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            g = out.grad
            # se abbiamo ridotto senza keepdims, reinseriamo gli assi persi cosi' che
            # il gradiente si ri-espanda (broadcast) sulla forma originale.
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.ones_like(self.data) * g

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        # media = somma / numero-di-elementi-mediati. Il gradiente fluisce da solo
        # attraverso sum e la moltiplicazione per lo scalare 1/n.
        s = self.sum(axis=axis, keepdims=keepdims)
        n = self.data.size / s.data.size
        return s * (1.0 / n)

    # ---- non-linearita' ---------------------------------------------------
    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(0.0, self.data), (self,), "relu")

        def _backward():
            # derivata: 1 dove l'input era positivo, 0 altrove
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad  # derivata elegante: 1 - tanh^2

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        e = np.exp(self.data)
        out = Tensor(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad  # derivata di exp e' exp

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def gelu(self) -> "Tensor":
        """GELU (approssimazione tanh), usata dai GPT reali. La costruiamo come
        COMPOSIZIONE di primitivi: il suo backward e' quindi automatico e verificato
        di riflesso dai gradient check dei mattoni. Formula:
            0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
        """
        c = math.sqrt(2.0 / math.pi)
        inner = (self + (self ** 3) * 0.044715) * c
        return (self * 0.5) * (inner.tanh() + 1.0)

    def softmax(self, axis: int = -1) -> "Tensor":
        """Softmax stabile lungo `axis`, come operazione dedicata (backward analitico)."""
        m = self.data.max(axis=axis, keepdims=True)
        e = np.exp(self.data - m)
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(s, (self,), "softmax")

        def _backward():
            # Jacobiano-per-vettore della softmax: dx = s * (g - somma(g*s))
            g = out.grad
            dot = (g * s).sum(axis=axis, keepdims=True)
            self.grad += s * (g - dot)

        out._backward = _backward
        return out

    # ---- backpropagation --------------------------------------------------
    def backward(self) -> None:
        """Calcola i gradienti di questo tensore (di solito la loss) rispetto a tutti
        i tensori del grafo, via ordinamento topologico + regola della catena."""
        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(v: "Tensor"):
            # DFS: un nodo entra in `topo` DOPO tutti i suoi genitori. Cosi', scorrendo
            # topo al contrario, ogni nodo riceve out.grad gia' completo prima di
            # propagarlo (nessun gradiente parziale).
            if id(v) not in visited:
                visited.add(id(v))
                for parent in v._prev:
                    build(parent)
                topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)  # dL/dL = 1: il seme della catena
        for v in reversed(topo):
            v._backward()

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, op={self._op!r})"


# ---- funzioni a livello di modulo -----------------------------------------
def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Cross-entropy media tra `logits` (B, V) e i target interi (B,), fusa e stabile.

    PERCHE' e' un'operazione unica e non log(softmax(...)) composta: (1) STABILITA' --
    la forma fusa (log-sum-exp) evita log(0)=-inf; (2) il gradiente e' la sottrazione
    semplice (softmax - onehot)/B, gia' incontrata in Fase 2. Anche PyTorch la fonde
    (F.cross_entropy) per gli stessi motivi.
    """
    x = logits.data
    B, V = x.shape
    m = x.max(axis=1, keepdims=True)
    logsumexp = m + np.log(np.exp(x - m).sum(axis=1, keepdims=True))  # (B, 1)
    rows = np.arange(B)
    nll = float((logsumexp[rows, 0] - x[rows, targets]).mean())
    out = Tensor(nll, (logits,), "cross_entropy")

    def _backward():
        p = np.exp(x - logsumexp)          # softmax(logits), stabile
        p[rows, targets] -= 1.0            # softmax - onehot(y)
        p /= B                             # media sul batch
        logits.grad += p * out.grad

    out._backward = _backward
    return out
