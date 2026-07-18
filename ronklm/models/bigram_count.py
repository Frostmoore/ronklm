"""BigramCount — il language model piu' semplice che esista (Fase 1).

Idea: predire il prossimo carattere guardando SOLO l'ultimo, e imparare le
probabilita' CONTANDO, non addestrando. Nessun gradiente. Serve a separare i
concetti (cosa vuol dire modellare il linguaggio) dai meccanismi (come si addestra
una rete): mischiarli e' il modo classico di non capire ne' gli uni ne' gli altri.

Concetti introdotti: distribuzione sul prossimo token, smoothing, campionamento e
soprattutto la LOSS (negative log-likelihood / cross-entropy), la bussola di ogni
addestramento futuro.
"""
from __future__ import annotations

import numpy as np


class BigramCount:
    """Modello a bigrammi basato su conteggi.

    Attributi:
        vocab_size (int)          : numero di token distinti.
        N (np.ndarray[int64])     : matrice (V, V) dei conteggi: N[i, j] = quante
                                    volte al carattere i segue j.
        P (np.ndarray[float64])   : matrice (V, V) di probabilita' per riga (None
                                    finche' non si chiama fit()).
        smoothing (float)         : conteggio fittizio aggiunto a tutte le coppie.
    """

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.N = np.zeros((vocab_size, vocab_size), dtype=np.int64)
        self.P: np.ndarray | None = None
        self.smoothing: float = 1.0

    # --- addestramento (che qui e' solo conteggio) -------------------------
    def fit(self, data: np.ndarray, smoothing: float = 1.0) -> "BigramCount":
        """Conta i bigrammi in `data` (1-D di indici) e calcola le probabilita' P.

        PERCHE' np.add.at e non un ciclo Python: `np.add.at(N, (a, b), 1)` incrementa
        N[a[k], b[k]] per ogni k, in C, in un colpo solo. Contare ~240k coppie con un
        ciclo Python sarebbe lento; vettorizzato e' istantaneo.
        """
        self.smoothing = smoothing
        self.N[:] = 0
        a = data[:-1]  # carattere "da"
        b = data[1:]   # carattere "a" (il successivo)
        np.add.at(self.N, (a, b), 1)

        # Da conteggi a probabilita': ogni riga divisa per la sua somma. Il +1
        # (smoothing di Laplace) evita zeri: una coppia mai vista nel train avrebbe
        # probabilita' 0 -> log(0) = -inf -> loss infinita al primo evento raro in
        # validation. "Fingiamo di aver visto tutto almeno una volta".
        smoothed = self.N.astype(np.float64) + smoothing
        # keepdims=True: la somma per riga ha shape (V, 1) e non (V,), cosi' la
        # divisione si propaga (broadcasting) riga per riga come vogliamo.
        self.P = smoothed / smoothed.sum(axis=1, keepdims=True)
        return self

    # --- valutazione: la LOSS ---------------------------------------------
    def nll(self, data: np.ndarray) -> float:
        """Negative Log-Likelihood media (in nats) sulle coppie consecutive di `data`.

        E' la cross-entropy: -log(probabilita' assegnata al carattere realmente
        accaduto), mediata. Bassa = il modello "si sorprende poco" = buono.
        Riferimento: un modello uniforme vale log(vocab_size) (~4.23 con V=69).
        """
        if self.P is None:
            raise RuntimeError("Chiama fit() prima di nll().")
        a = data[:-1]
        b = data[1:]
        probs = self.P[a, b]  # probabilita' assegnata a ciascuna coppia realizzata
        return float(-np.log(probs).mean())

    @staticmethod
    def uniform_nll(vocab_size: int) -> float:
        """NLL di un modello che tira a caso: log(vocab_size). Il riferimento da battere."""
        return float(np.log(vocab_size))

    # --- generazione: campionamento autoregressivo ------------------------
    def generate(
        self,
        rng: np.random.Generator,
        n: int,
        start: int = 0,
    ) -> list[int]:
        """Genera `n` indici partendo da `start`, campionando dalla riga di P.

        PERCHE' si CAMPIONA e non si prende sempre il massimo (greedy): col massimo,
        da uno stesso carattere uscirebbe sempre la stessa catena -> testo ciclico e
        degenere. Campionare secondo la distribuzione mantiene la varieta' del
        linguaggio: output diverso a ogni run ma fedele alle statistiche del corpus.
        """
        if self.P is None:
            raise RuntimeError("Chiama fit() prima di generate().")
        out = [start]
        cur = start
        for _ in range(n - 1):
            cur = int(rng.choice(self.vocab_size, p=self.P[cur]))
            out.append(cur)
        return out

    def __repr__(self) -> str:
        fitted = "fitted" if self.P is not None else "non-fitted"
        return f"BigramCount(vocab_size={self.vocab_size}, {fitted})"
