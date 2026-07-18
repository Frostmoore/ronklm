"""Prepara il corpus del Percorso A: scarica e pulisce *Le avventure di Pinocchio*.

PERCHE' uno script e non un file scaricato a mano (piano 0.2):
    Riproducibilita'. Questo script documenta ESATTAMENTE come `data/input.txt` e'
    stato prodotto: da quale fonte, con quali tagli e quali normalizzazioni. Se un
    domani il file si corrompe o vogliamo cambiare edizione, la provenienza non e'
    persa. Nei progetti reali la provenienza dei dati e' la prima cosa che sparisce
    e la piu' costosa da ricostruire.

Fonte: Project Gutenberg, ebook #52484 ("Storia di un burattino", ed. Bemporad
1902, testo integrale). NB: l'ebook #19517 con lo stesso titolo e' solo una
*lettura audio* (elenco capitoli con timestamp), non il testo: da non usare.

Uso:
    python data/prepare_corpus.py            # usa la cache se presente
    python data/prepare_corpus.py --force    # riscarica ignorando la cache

Output:
    data/input.txt          il corpus pulito, UTF-8 (committato nel repo)
    data/pinocchio_raw.txt  cache del download grezzo (gitignored)
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

# --- Configurazione ---------------------------------------------------------
URL = "https://www.gutenberg.org/files/52484/52484-0.txt"
HERE = Path(__file__).resolve().parent
RAW_PATH = HERE / "pinocchio_raw.txt"
OUT_PATH = HERE / "input.txt"

# Ancore testuali per isolare la STORIA dal paratesto (frontespizio + indice).
# Le scegliamo come stringhe uniche e stabili invece che come offset numerici:
# un offset si rompe alla prima ri-edizione, un'ancora di contenuto no.
STORY_START = "I.\n\nCome andò che Maestro Ciliegia"  # inizio Capitolo I
STORY_END = "FINE."                                    # chiusura della storia
# Dopo "FINE." nel file c'e' l'INDICE (elenco capitoli con numeri di pagina) e la
# nota del trascrittore: li tagliamo, sono rumore non-narrativo.


def download(force: bool = False) -> str:
    """Scarica il testo grezzo (o lo legge dalla cache) e lo restituisce come str.

    NB: leggiamo/scriviamo la cache in BINARIO (read_bytes/write_bytes) e
    decodifichiamo a mano. Se usassimo read_text/write_text, su Windows la
    traduzione automatica dei fine-riga (\\n <-> \\r\\n) applicata due volte
    corromperebbe i CRLF del file, sballando le ancore testuali a valle.
    """
    if RAW_PATH.exists() and not force:
        print(f"[cache] leggo {RAW_PATH.name}")
        return RAW_PATH.read_bytes().decode("utf-8")
    print(f"[rete]  scarico {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 RonkLM"})
    raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
    RAW_PATH.write_bytes(raw.encode("utf-8"))
    print(f"[cache] salvato {RAW_PATH.name} ({len(raw)} char)")
    return raw


def extract_story(raw: str) -> str:
    """Ritaglia la sola narrazione, scartando frontespizio e indice finale."""
    # Uniformiamo subito i fine-riga: il file Gutenberg usa CRLF (\r\n) stile
    # Windows; li portiamo a \n cosi' le ancore e i conteggi sono affidabili.
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    s = text.find(STORY_START)
    if s < 0:
        sys.exit("ERRORE: ancora di inizio storia non trovata (edizione cambiata?)")
    e = text.find(STORY_END, s)
    if e < 0:
        sys.exit("ERRORE: ancora di fine storia non trovata (edizione cambiata?)")
    story = text[s : e + len(STORY_END)]

    # Guardia di sanita': ci aspettiamo ~250 KB. Se il ritaglio e' minuscolo,
    # qualcosa e' andato storto e vogliamo saperlo ORA, non a training avviato.
    if len(story) < 200_000:
        sys.exit(f"ERRORE: ritaglio troppo corto ({len(story)} char): confini errati?")
    return story


def normalize(story: str) -> str:
    """Normalizzazione MINIMA: il testo e' gia' pulito, tocchiamo solo il rumore.

    PERCHE' normalizzare (piano 0.2): ogni variante tipografica di uno stesso segno
    e' un token in piu' nel vocabolario e diluisce le statistiche. Riduciamo le
    varianti al minimo SENZA distruggere informazione linguistica.

    Cosa TENIAMO (e perche'):
      —  (em dash)      marcatore dei dialoghi italiani: essenziale, il modello lo impara
      « »               virgolette caporali: punteggiatura italiana legittima
      ò è ì ù à È        vocali accentate: sono la lingua
      maiuscole, a-capo  il modello imparera' "dopo il punto la maiuscola" e i paragrafi

    Cosa RIMUOVIAMO (artefatti Gutenberg, non-narrativa):
      [Illustrazione: ...]  didascalie editoriali delle illustrazioni (79 blocchi):
                            non sono prosa di Collodi e spezzano il flusso della
                            frase -> eliminate per intero, anche su piu' righe.
      _ (underscore)        marcatore di corsivo Gutenberg (_ohi_, _tac!_): la
                            parola e' vera, togliamo solo i due underscore attorno.

    Cosa NORMALIZZIAMO (e perche'):
      \\xa0 (nbsp)       spazio "invisibile diverso": lo rendiamo spazio normale
      “ „ (2 refusi)    2 sole occorrenze: sarebbero 2 voci-fantasma nel vocabolario
    """
    # 1. Via le didascalie delle illustrazioni. re.DOTALL perche' possono andare
    #    a capo; *? (non-greedy) per chiudere al primo ']' e non mangiare oltre.
    story = re.sub(r"\[Illustrazione:.*?\]", "", story, flags=re.DOTALL)

    replacements = {
        "\xa0": " ",   # no-break space -> spazio normale
        "_": "",       # marcatore di corsivo -> via, tenendo la parola
        "“": '"',  # " left double quote (1 occorrenza) -> " ASCII
        "„": '"',  # „ low double quote  (1 occorrenza) -> " ASCII
    }
    for a, b in replacements.items():
        story = story.replace(a, b)

    # Ripulitura degli spazi: togliamo gli spazi a fine riga (invisibili, rumore)
    # e collassiamo 3+ righe vuote consecutive in una sola riga vuota (un solo
    # stacco di paragrafo), cosi' la struttura resta ma senza buchi enormi.
    lines = [ln.rstrip() for ln in story.split("\n")]
    out_lines: list[str] = []
    blank_run = 0
    for ln in lines:
        if ln == "":
            blank_run += 1
            if blank_run <= 1:
                out_lines.append(ln)
        else:
            blank_run = 0
            out_lines.append(ln)
    return "\n".join(out_lines).strip() + "\n"


def report(text: str) -> None:
    """Stampa le statistiche di verifica richieste dal piano (0.2)."""
    vocab = sorted(set(text))
    counts = Counter(text)
    print("\n===== STATISTICHE CORPUS =====")
    print(f"caratteri totali : {len(text):,}")
    print(f"dimensione vocab : {len(vocab)}")

    # Elenco completo del vocabolario a occhio: verifichiamo che NON ci siano
    # caratteri-sorpresa (piano 0.2). Mostriamo il codepoint dei non stampabili.
    def show(ch: str) -> str:
        if ch == "\n":
            return "\\n"
        if ch == " ":
            return "' '"
        return ch

    print("\nvocabolario (ordinato):")
    print("  " + " ".join(show(c) for c in vocab))

    print("\ntop-20 caratteri per frequenza:")
    for ch, n in counts.most_common(20):
        print(f"  U+{ord(ch):04X}  {show(ch):>4}  {n:,}")

    # Guardia: il piano fissa vocab < ~120. Se sfora, c'e' rumore da ripulire.
    if len(vocab) > 120:
        print(f"\n[ATTENZIONE] vocab {len(vocab)} > 120: rivedere la normalizzazione.")
    else:
        print(f"\n[OK] vocab {len(vocab)} <= 120.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepara data/input.txt (Pinocchio).")
    ap.add_argument("--force", action="store_true", help="riscarica ignorando la cache")
    args = ap.parse_args()

    raw = download(force=args.force)
    story = extract_story(raw)
    text = normalize(story)
    # write_bytes (non write_text) per non far tradurre \n in \r\n su Windows:
    # il file su disco deve combaciare al byte con il testo qui analizzato.
    OUT_PATH.write_bytes(text.encode("utf-8"))
    print(f"[out]   scritto {OUT_PATH.name} ({len(text):,} char)")
    report(text)


if __name__ == "__main__":
    # Su Windows la console e' cp1252: forziamo UTF-8 in output per stampare — « ecc.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
