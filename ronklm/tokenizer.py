"""CharTokenizer — il ponte testo <-> interi (Fase 0.3).

Un modello non vede testo: vede numeri. Il tokenizer e' il traduttore, e per noi
lavora a livello di CARATTERE (piano I.3): ogni carattere e' un token, il
vocabolario e' l'insieme dei caratteri unici del corpus (~69 per Pinocchio).

Questa classe e' il CONTRATTO tra il mondo del testo e il mondo dei tensori: ogni
esperimento di ogni fase successiva ci passa attraverso. Per questo e' minuscola,
deterministica e testata subito (test_tokenizer.py), e non la tocchiamo piu'.
"""
from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    """Mappa bidirezionale carattere <-> indice intero.

    Attributi:
        chars (list[str]) : vocabolario, lista ORDINATA di caratteri unici.
        stoi  (dict)      : string-to-int, carattere -> indice.
        itos  (dict)      : int-to-string, indice -> carattere.
    """

    def __init__(self, chars: list[str]) -> None:
        # PERCHE' ordinato (sorted): il determinismo. Se l'ordine dei caratteri
        # cambiasse tra due esecuzioni, gli indici cambierebbero e un modello
        # salvato diventerebbe illeggibile (i pesi parlerebbero un'altra mappa).
        # Ordinando sempre allo stesso modo, lo stesso corpus da' lo stesso vocab.
        self.chars: list[str] = sorted(set(chars))
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(self.chars)}
        self.itos: dict[int, str] = {i: c for i, c in enumerate(self.chars)}

    # --- costruzione -------------------------------------------------------
    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Costruisce il vocabolario dai caratteri unici presenti nel testo."""
        return cls(list(text))

    # --- proprieta' --------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """Numero di token distinti. E' la dimensione dell'ultimo strato di ogni
        modello (un punteggio per token), quindi un numero che ricorrera' sempre."""
        return len(self.chars)

    # --- codifica / decodifica --------------------------------------------
    def encode(self, text: str) -> list[int]:
        """Testo -> lista di indici interi.

        PERCHE' fallisce rumorosamente su un carattere sconosciuto: un carattere
        mai visto non ha indice. Meglio un errore chiaro SUBITO che un indice
        inventato che sballa silenziosamente tutto a valle. (Nel Percorso B il BPE
        gestira' gli sconosciuti scomponendoli; qui a livello di carattere no.)
        """
        try:
            return [self.stoi[c] for c in text]
        except KeyError as e:
            raise ValueError(
                f"Carattere fuori vocabolario: {e.args[0]!r}. "
                f"Il tokenizer conosce solo i {self.vocab_size} caratteri del corpus."
            ) from None

    def decode(self, indices: list[int]) -> str:
        """Lista di indici -> testo. Errore chiaro su un indice fuori range."""
        try:
            return "".join(self.itos[int(i)] for i in indices)
        except KeyError as e:
            raise ValueError(
                f"Indice fuori vocabolario: {e.args[0]}. "
                f"Ammessi 0..{self.vocab_size - 1}."
            ) from None

    # --- serializzazione ---------------------------------------------------
    # PERCHE' serve salvare il vocabolario: in Fase 7 il checkpoint del modello
    # dovra' contenerlo. Un modello char-level caricato con una mappa stoi diversa
    # da quella di training produce spazzatura deterministica (pesi giusti, indici
    # sbagliati). Vocabolario e pesi vanno SEMPRE insieme.
    def save(self, path: str | Path) -> None:
        """Salva il vocabolario (la lista ordinata di caratteri) come JSON."""
        Path(path).write_text(
            json.dumps({"chars": self.chars}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """Ricostruisce il tokenizer da un file salvato con save()."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["chars"])

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"
