"""BPE — Byte Pair Encoding scritto a mano (Fase 10).

E' il pezzo che nel Percorso A avevamo VOLUTAMENTE rimandato (piano I.3): a livello di
carattere il tokenizer non era un tema, e toglierlo di mezzo ci ha fatto concentrare su
gradienti e attention. Adesso serve davvero, e lo costruiamo da zero come tutto il
resto. E' lo stesso algoritmo dei tokenizer dei GPT reali.

L'IDEA, in una frase: partire dai byte e fondere ripetutamente la COPPIA ADIACENTE PIU'
FREQUENTE in un nuovo simbolo, finche' il vocabolario non raggiunge la taglia voluta.
E' compressione guidata dai dati: le sequenze frequenti si meritano un simbolo proprio.

PERCHE' partire dai BYTE e non dai caratteri: cosi' il tokenizer non puo' MAI incontrare
qualcosa che non sa rappresentare. Ogni testo, in qualsiasi lingua, con qualsiasi
simbolo strano, e' una sequenza di byte; nel caso peggiore lo scompone in byte singoli.
E' la proprieta' che il word-level non poteva dare (piano I.3): niente "fuori
vocabolario", mai. I 256 byte sono quindi i primi 256 token, sempre.
"""
from __future__ import annotations

import heapq
import json
import pickle
import re
from collections import Counter
from pathlib import Path

# Pre-tokenizzazione: prima di applicare il BPE spezziamo il testo in "parole".
# PERCHE': senza questo, il BPE potrebbe fondere attraverso gli spazi e creare token
# come "della_casa" che sprecano vocabolario e generalizzano male. Il pattern (stile
# GPT-2) tiene lo spazio ATTACCATO alla parola che segue (" casa"), cosi' il modello
# distingue "casa" a inizio riga da " casa" dentro una frase, senza sprecare un token
# per lo spazio isolato.
# NB: le classi Unicode \p{L}/\p{N} esistono solo nel pacchetto `regex`, non nella
# stdlib. Usiamo gli equivalenti di `re`: [^\W\d_] = "lettera" (parola senza cifre e
# underscore), \d = cifra. Cosi' restiamo a zero dipendenze in piu'.
SPLIT_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


def _split_words(text: str) -> list[str]:
    """Spezza il testo in unita' pre-tokenizzate (parole con lo spazio iniziale)."""
    return SPLIT_PATTERN.findall(text)


def _get_pair_counts(word_freqs: dict[tuple[int, ...], int]) -> Counter:
    """Conta le coppie adiacenti, PESATE per quante volte compare ogni parola.

    Trucco fondamentale per la velocita': non scorriamo il corpus, scorriamo il
    DIZIONARIO delle parole uniche con la loro frequenza. Wikipedia ha miliardi di
    token ma solo qualche milione di parole distinte -> il conteggio diventa fattibile.
    """
    counts: Counter = Counter()
    for word, freq in word_freqs.items():
        for a, b in zip(word, word[1:]):
            counts[(a, b)] += freq
    return counts


def _merge_word(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    """Sostituisce ogni occorrenza di `pair` in `word` col nuovo simbolo `new_id`."""
    out: list[int] = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    """Tokenizer BPE a livello di byte.

    Attributi:
        merges (dict[(int,int), int]) : coppia -> nuovo id, NELL'ORDINE in cui e' stata
                                        appresa (l'ordine conta: encode le riapplica cosi').
        vocab  (dict[int, bytes])     : id -> sequenza di byte che rappresenta.
        special (dict[str, int])      : token speciali (usati nel Percorso C per la chat).
    """

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special: dict[str, int] = {}

    # --- addestramento ------------------------------------------------------
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        """Impara le fusioni dal testo, fino a raggiungere `vocab_size` token.

        vocab_size include i 256 byte di base: con vocab_size=16384 si imparano
        16384-256 = 16128 fusioni.

        IMPLEMENTAZIONE INCREMENTALE (e perche' serve). La versione ingenua, dopo ogni
        fusione, ricconta TUTTE le coppie di TUTTE le parole: costo
        n_merges x n_parole_uniche, che su Wikipedia (milioni di parole uniche x 16.000
        merge) sono decine di miliardi di operazioni -> ore. Qui invece manteniamo:
          - `pair_counts`: il conteggio corrente di ogni coppia;
          - `pair_words` : per ogni coppia, QUALI parole la contengono;
          - un max-heap per pescare in fretta la coppia piu' frequente.
        A ogni fusione tocchiamo SOLO le parole che contengono quella coppia, e
        aggiorniamo i conteggi in differenza. Stesso identico risultato, ordini di
        grandezza piu' veloce.
        """
        assert vocab_size >= 256, "il vocabolario parte dai 256 byte"
        n_merges = vocab_size - 256

        # 1. pre-tokenizza e conta le parole uniche (non le occorrenze!)
        raw_freqs = Counter(_split_words(text))
        agg: dict[tuple[int, ...], int] = {}
        for w, f in raw_freqs.items():
            key = tuple(w.encode("utf-8"))
            agg[key] = agg.get(key, 0) + f
        words: list[list[int]] = [list(k) for k in agg]
        freqs: list[int] = list(agg.values())
        del raw_freqs, agg
        if verbose:
            print(f"  parole uniche: {len(words):,}", flush=True)

        # 2. indice iniziale: conteggi delle coppie e in quali parole stanno
        pair_counts: dict[tuple[int, int], int] = {}
        pair_words: dict[tuple[int, int], set[int]] = {}
        for wi, w in enumerate(words):
            f = freqs[wi]
            for p in zip(w, w[1:]):
                pair_counts[p] = pair_counts.get(p, 0) + f
                pair_words.setdefault(p, set()).add(wi)

        # max-heap con "cancellazione pigra": le voci stantie si scartano al momento
        # del pop confrontandole col conteggio corrente.
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        for k in range(n_merges):
            # pesca la coppia realmente piu' frequente (scartando le voci obsolete)
            pair = None
            while heap:
                negc, cand = heapq.heappop(heap)
                if pair_counts.get(cand, 0) == -negc and -negc >= 2:
                    pair = cand
                    freq = -negc
                    break
            if pair is None:
                if verbose:
                    print(f"  nessuna coppia ripetuta: mi fermo a {256 + k} token")
                break

            new_id = 256 + k
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            touched: set[tuple[int, int]] = set()
            for wi in list(pair_words.get(pair, ())):
                w = words[wi]
                f = freqs[wi]
                # la parola potrebbe non contenere piu' la coppia (indice pigro)
                if not any(w[i] == pair[0] and w[i + 1] == pair[1] for i in range(len(w) - 1)):
                    continue
                new_w = list(_merge_word(tuple(w), pair, new_id))
                # aggiorna i conteggi in DIFFERENZA: togli le vecchie coppie, metti le nuove
                for p in zip(w, w[1:]):
                    pair_counts[p] = pair_counts.get(p, 0) - f
                    touched.add(p)
                for p in zip(new_w, new_w[1:]):
                    pair_counts[p] = pair_counts.get(p, 0) + f
                    pair_words.setdefault(p, set()).add(wi)
                    touched.add(p)
                words[wi] = new_w
            pair_counts.pop(pair, None)
            pair_words.pop(pair, None)
            for p in touched:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))

            if verbose and (k % 500 == 0 or k == n_merges - 1):
                shown = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"  merge {k:6d}/{n_merges}  {shown!r}  (freq {freq:,})", flush=True)
        return self

    # --- codifica / decodifica ---------------------------------------------
    def _encode_word(self, word: str) -> list[int]:
        """Applica le fusioni apprese a una singola parola pre-tokenizzata."""
        ids = list(word.encode("utf-8"))
        if len(ids) < 2:
            return ids
        while True:
            # trova la fusione applicabile IMPARATA PIU' PRESTO (id piu' basso):
            # l'ordine di apprendimento e' l'ordine di applicazione.
            best: tuple[int, int] | None = None
            best_id = None
            for a, b in zip(ids, ids[1:]):
                mid = self.merges.get((a, b))
                if mid is not None and (best_id is None or mid < best_id):
                    best, best_id = (a, b), mid
            if best is None:
                break
            ids = list(_merge_word(tuple(ids), best, best_id))
        return ids

    def encode(self, text: str) -> list[int]:
        """Testo -> lista di id. Non fallisce MAI: al peggio scompone in byte."""
        out: list[int] = []
        for word in _split_words(text):
            out.extend(self._encode_word(word))
        return out

    def decode(self, ids: list[int]) -> str:
        """Id -> testo. Concatena i byte dei token e decodifica in UTF-8."""
        parts = [self.vocab[int(i)] for i in ids]
        return b"".join(parts).decode("utf-8", errors="replace")

    # --- proprieta' e serializzazione --------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: str | Path) -> None:
        """Salva le fusioni (pickle: le chiavi sono tuple, JSON non le supporta)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({"merges": self.merges, "special": self.special}, f)
        # copia leggibile per ispezione umana (i primi merge dicono molto sul corpus)
        readable = {
            str(new_id): self.vocab[new_id].decode("utf-8", errors="replace")
            for (_, _), new_id in list(self.merges.items())[:500]
        }
        p.with_suffix(".preview.json").write_text(
            json.dumps(readable, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        with open(path, "rb") as f:
            data = pickle.load(f)
        tok = cls()
        tok.merges = data["merges"]
        tok.special = data.get("special", {})
        # ricostruisce il vocabolario applicando le fusioni in ordine
        for (a, b), new_id in sorted(tok.merges.items(), key=lambda kv: kv[1]):
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        return tok

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"
