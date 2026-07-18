"""BigramNeural — lo stesso bigram, ma IMPARATO invece che contato (Fase 2).

E' il "hello world" della backpropagation. Il modello e' una sola matrice di pesi W;
il forward e' un prodotto + softmax; il gradiente e' derivato A MANO (la derivazione
completa e' in explain.md). Siccome sappiamo gia' dove si deve arrivare (la P della
Fase 1), ogni pezzo del meccanismo di training e' verificabile contro una verita'
nota.

Il ciclo che qui vediamo per la prima volta -- forward, loss, backward, update -- e'
lo stesso, identico, di ogni rete neurale fino a GPT-4. Cambia solo cosa c'e' dentro
il forward.
"""
from __future__ import annotations

import numpy as np


class BigramNeural:
    """Regressione softmax a un solo strato: logits = onehot(x) @ W.

    onehot(x) @ W seleziona semplicemente la riga x-esima di W (l'osservazione che in
    Fase 4 diventera' l'Embedding). Quindi W[i] sono i "logits" per il carattere che
    segue i; softmax li trasforma in probabilita'.

    Attributi:
        vocab_size (int)          : V, dimensione del vocabolario.
        W (np.ndarray[float64])   : pesi, shape (V, V). W[i, j] = logit di j dato i.
    """

    def __init__(self, vocab_size: int, rng: np.random.Generator, init_std: float = 0.01) -> None:
        self.vocab_size = vocab_size
        # Init casuale PICCOLO: con W~0 tutti i logit sono ~uguali -> il modello parte
        # dalla distribuzione ~uniforme, e la loss iniziale vale ~log(V). E' il primo
        # sanity check di ogni training: se all'avvio la loss non e' ~log(V), l'init
        # e' sbagliato. Piccolo (non zero) per non partire "sicuri e a caso".
        self.W = rng.normal(0.0, init_std, size=(vocab_size, vocab_size))

    # --- forward -----------------------------------------------------------
    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Softmax stabile lungo l'ultimo asse. Il '- max' non cambia il risultato
        (softmax e' invariante per traslazione) ma evita exp(numero enorme)=inf."""
        z = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def forward(self, x_idx: np.ndarray) -> np.ndarray:
        """Indici correnti (B,) -> probabilita' (B, V) sul prossimo carattere."""
        logits = self.W[x_idx]           # (B, V): selezione di riga = onehot @ W
        return self._softmax(logits)

    # --- loss e gradiente (derivato a mano) --------------------------------
    def loss_and_grad(self, x_idx: np.ndarray, y_idx: np.ndarray) -> tuple[float, np.ndarray]:
        """Cross-entropy media e gradiente dW, calcolati esplicitamente.

        Derivazione (dettagli in explain.md), il risultato e' sorprendentemente
        semplice: sui logit il gradiente e' (probs - onehot(y)); su W, siccome
        l'input e' one-hot, si aggiorna solo la riga del carattere visto.

            dlogits = (probs - onehot(y)) / B      # (B, V)
            dW[i]   = somma dei dlogits delle posizioni in cui x == i
        """
        B = len(x_idx)
        logits = self.W[x_idx]                       # (B, V)
        probs = self._softmax(logits)                # (B, V)

        # loss = media di -log(prob assegnata al carattere giusto)
        loss = float(-np.log(probs[np.arange(B), y_idx]).mean())

        # gradiente sui logit: probs - onehot(y), mediato sul batch
        dlogits = probs.copy()
        dlogits[np.arange(B), y_idx] -= 1.0
        dlogits /= B                                 # (B, V)

        # gradiente su W: ogni riga i riceve la somma dei dlogits dove x_idx == i.
        # np.add.at accumula correttamente anche con indici (caratteri) ripetuti.
        dW = np.zeros_like(self.W)
        np.add.at(dW, x_idx, dlogits)
        return loss, dW

    def loss(self, x_idx: np.ndarray, y_idx: np.ndarray) -> float:
        """Solo la loss (senza gradiente): comodo per il gradient check numerico."""
        B = len(x_idx)
        probs = self.forward(x_idx)
        return float(-np.log(probs[np.arange(B), y_idx]).mean())

    # --- training: discesa del gradiente -----------------------------------
    def train(
        self,
        data: np.ndarray,
        steps: int,
        lr: float,
        rng: np.random.Generator | None = None,
        batch_size: int | None = None,
        log_every: int = 0,
    ) -> list[float]:
        """Addestra per `steps` passi di discesa del gradiente. Ritorna la storia della loss.

        Il ciclo canonico, lo stesso di tutto il deep learning:
            1) prendi i dati   2) prevedi (forward)   3) misura l'errore (loss)
            4) calcola i gradienti (backward)   5) aggiorna: W -= lr * dW

        Se batch_size e' None -> full-batch (tutte le coppie del corpus a ogni passo):
        problema convesso, converge al minimo globale = distribuzione empirica dei
        conteggi (la P della Fase 1). Se batch_size e' dato -> minibatch stocastico.
        """
        a_all = data[:-1]
        b_all = data[1:]
        history: list[float] = []
        for t in range(steps):
            if batch_size is None:
                xa, yb = a_all, b_all
            else:
                if rng is None:
                    raise ValueError("Con batch_size serve un rng.")
                ix = rng.integers(0, len(a_all), size=batch_size)
                xa, yb = a_all[ix], b_all[ix]
            loss, dW = self.loss_and_grad(xa, yb)
            self.W -= lr * dW                        # il passo di discesa
            history.append(loss)
            if log_every and (t % log_every == 0 or t == steps - 1):
                print(f"  step {t:5d}  loss {loss:.4f}")
        return history

    def train_from_counts(
        self,
        N: np.ndarray,
        steps: int,
        lr: float,
        log_every: int = 0,
    ) -> list[float]:
        """Discesa del gradiente FULL-BATCH esatta, calcolata dai conteggi N.

        PERCHE' e' equivalente (e velocissima). Nel full-batch, tutte le posizioni con
        lo stesso carattere di partenza i hanno gli STESSI logit W[i], quindi le stesse
        probabilita' softmax(W[i]). Il gradiente sommato sulla riga i diventa allora,
        in forma chiusa:

            dW[i] = ( (numero di volte che i appare come "da") * softmax(W[i]) - N[i] ) / totale

        Cioe' non serve toccare le 216.000 coppie una per una: bastano i conteggi. E'
        la prova che il modello neurale e quello a conteggio (Fase 1) sono la stessa
        cosa vista da due lati. Costo: O(V^2) per passo invece di O(coppie).
        """
        Nf = N.astype(np.float64)
        total = Nf.sum()
        row_tot = Nf.sum(axis=1, keepdims=True)      # (V, 1): occorrenze di ogni "da"
        history: list[float] = []
        for t in range(steps):
            P = self._softmax(self.W)                 # (V, V)
            # loss full-batch = media pesata (per conteggio) di -log P
            loss = float(-(Nf * np.log(P)).sum() / total)
            dW = (row_tot * P - Nf) / total           # gradiente esatto in forma chiusa
            self.W -= lr * dW
            history.append(loss)
            if log_every and (t % log_every == 0 or t == steps - 1):
                print(f"  step {t:5d}  loss {loss:.4f}")
        return history

    # --- valutazione e generazione ----------------------------------------
    def nll(self, data: np.ndarray) -> float:
        """NLL media (nats) sulle coppie consecutive di `data`. Stessa metrica del bigram a conteggio."""
        return self.loss(data[:-1], data[1:])

    def probabilities(self) -> np.ndarray:
        """Ritorna softmax(W): la matrice (V, V) di probabilita' appresa, da confrontare con la P contata."""
        return self._softmax(self.W)

    def generate(self, rng: np.random.Generator, n: int, start: int = 0) -> list[int]:
        """Genera n indici campionando dalle probabilita' apprese (come il bigram a conteggio)."""
        out = [start]
        cur = start
        P = self.probabilities()
        for _ in range(n - 1):
            cur = int(rng.choice(self.vocab_size, p=P[cur]))
            out.append(cur)
        return out

    def __repr__(self) -> str:
        return f"BigramNeural(vocab_size={self.vocab_size})"
