"""Dataset e batching (Fase 0.4) — da testo a batch di tensori interi.

Trasforma il corpus in materiale di training: lo codifica una volta sola in interi,
lo divide in train/validation, e produce batch (X, Y) di finestre di caratteri.

Il concetto chiave e' come da una finestra di block_size+1 caratteri consecutivi
si ottengano block_size esempi in un colpo solo, con Y = X spostato di 1:

    finestra "ciao " (block_size=5), il carattere dopo e' 'm':
        X = "ciao "     Y = "iao m"
        pos 0: visto "c"      -> predici "i"
        pos 1: visto "ci"     -> predici "a"
        ...
        pos 4: visto "ciao "  -> predici "m"

Ogni posizione e' un esempio con contesto di lunghezza diversa. E' il motivo per
cui i transformer si addestrano cosi' in fretta: una finestra da T caratteri = T
predizioni da imparare insieme (in Fase 5 la maschera causale le calcolera' tutte
in un solo passaggio).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ronklm.tokenizer import CharTokenizer


def load_text(path: str | Path) -> str:
    """Legge il corpus come UTF-8. newline='' per non tradurre i fine-riga: il
    file e' gia' a LF puro (vedi data/prepare_corpus.py), lo vogliamo intatto."""
    return Path(path).read_text(encoding="utf-8", newline="")


class Dataset:
    """Corpus codificato e suddiviso in train/val, con estrazione di batch.

    Attributi:
        tokenizer (CharTokenizer) : la mappa carattere<->intero usata.
        data  (np.ndarray[int64]) : l'intero corpus codificato.
        train (np.ndarray[int64]) : primo (1-val_frac) del corpus.
        val   (np.ndarray[int64]) : ultimo val_frac del corpus.
    """

    def __init__(
        self,
        text: str,
        tokenizer: CharTokenizer,
        val_frac: float = 0.1,
    ) -> None:
        self.tokenizer = tokenizer
        # Codifica UNA volta sola l'intero corpus: da str a array di interi.
        # int64 e' sovrabbondante per ~69 simboli ma e' il tipo naturale per
        # indicizzare (embedding, one-hot) e non ci facciamo problemi ora.
        self.data = np.asarray(tokenizer.encode(text), dtype=np.int64)

        # PERCHE' lo split e' CONTIGUO (ultimo 10%) e non a caratteri sparsi: il
        # testo e' sequenziale. Se prendessimo caratteri a caso per la val, ognuno
        # avrebbe i suoi vicini nel train e la "verifica su testo mai visto"
        # sarebbe una finzione. Un blocco contiguo tenuto da parte e' testo
        # davvero non visto durante il training.
        n_train = int(len(self.data) * (1.0 - val_frac))
        self.train = self.data[:n_train]
        self.val = self.data[n_train:]

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        tokenizer: CharTokenizer | None = None,
        val_frac: float = 0.1,
    ) -> "Dataset":
        """Carica il corpus da file; se non passi un tokenizer, lo costruisce dal
        corpus stesso (il caso tipico in Fase 0)."""
        text = load_text(path)
        if tokenizer is None:
            tokenizer = CharTokenizer.from_text(text)
        return cls(text, tokenizer, val_frac=val_frac)

    def get_batch(
        self,
        split: str,
        block_size: int,
        batch_size: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estrae un batch casuale di finestre dal train o dalla val.

        Ritorna (X, Y), entrambi shape (batch_size, block_size), interi.
        Y e' X spostato di un carattere: Y[b, t] e' il carattere che segue X[b, t].

        rng va passato dall'esterno (np.random.default_rng(seed)) per la
        riproducibilita': stesso seed -> stessi batch -> bug inseguibili.
        """
        data = self._split(split)
        if len(data) < block_size + 1:
            raise ValueError(
                f"split '{split}' troppo corto ({len(data)}) per block_size={block_size}."
            )

        # PERCHE' posizioni CASUALI e non in ordine: la discesa del gradiente
        # stocastica (Fase 2.4) rende meglio con esempi scorrelati; finestre
        # consecutive del libro sarebbero quasi identiche e i gradienti
        # seguirebbero la trama invece della lingua.
        #
        # high = len(data) - block_size: l'ultima partenza valida, perche' Y ha
        # bisogno di un carattere in piu' (i+block_size deve esistere). integers()
        # ha high ESCLUSIVO, quindi i+block_size <= len(data)-1: sempre in range.
        ix = rng.integers(0, len(data) - block_size, size=batch_size)
        X = np.stack([data[i : i + block_size] for i in ix])
        Y = np.stack([data[i + 1 : i + 1 + block_size] for i in ix])
        return X, Y

    def _split(self, split: str) -> np.ndarray:
        if split == "train":
            return self.train
        if split == "val":
            return self.val
        raise ValueError(f"split deve essere 'train' o 'val', non {split!r}.")

    def __repr__(self) -> str:
        return (
            f"Dataset(total={len(self.data):,}, train={len(self.train):,}, "
            f"val={len(self.val):,}, vocab={self.tokenizer.vocab_size})"
        )
