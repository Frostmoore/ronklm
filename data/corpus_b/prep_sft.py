"""Prepara i dati per l'SFT (Fase 13): domande -> risposte in italiano.

Da un dataset di istruzioni (alpaca-gpt4-italian) costruisce due file binari:
  sft_tokens.bin (uint16) : la sequenza di token di tutti gli esempi concatenati
  sft_mask.bin   (uint8)  : 1 dove il token e' parte della RISPOSTA, 0 sul resto

PERCHE' la maschera sulla loss: vogliamo insegnare al modello a RISPONDERE, non a
ri-generare la domanda. Calcolando la loss solo sui token della risposta (mask=1), il
modello impara il formato "domanda -> risposta" e a produrre la risposta, ignorando il
compito di predire la domanda (che l'utente scrive, non lui). E' lo standard dell'SFT.

Template (senza token speciali, per restare compatibili con la conversione GGUF):
    ### Domanda:
    {domanda}

    ### Risposta:
    {risposta}

    ### Domanda:
Il "### Domanda:" finale, incluso nella regione con loss, insegna al modello a
CHIUDERE la risposta (in generazione ci fermiamo quando lo riproduce).

Filtri (per un modello da 50M): risposte brevi e in prosa, niente liste/markdown/codice,
niente disclaimer da assistente ("Come intelligenza artificiale...").

Uso:
    python data/corpus_b/prep_sft.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ronklm.bpe import BPETokenizer  # noqa: E402

SRC = Path("D:/RonkLM_corpus/sft/gpt4.json")
OUT = Path("D:/RonkLM_corpus/sft")
BPE = Path("D:/RonkLM_corpus/tokens/bpe_16384.pkl")

PROMPT_TPL = "### Domanda:\n{q}\n\n### Risposta:\n"
ANSWER_SUFFIX = "\n\n### Domanda:\n"   # insegna a chiudere (stop-string in generazione)

# disclaimer da assistente: rovinerebbero la voce narrativa
BAD_PHRASES = [
    "intelligenza artificiale", "modello linguistico", "modello di linguaggio",
    "come un'ia", "come ia", "come assistente", "language model", "as an ai",
    "non ho la capacità", "non sono in grado di", "non ho accesso",
]


def is_clean(q: str, a: str) -> bool:
    if not (10 <= len(q) <= 300 and 20 <= len(a) <= 350):
        return False
    low = a.lower()
    if any(p in low for p in BAD_PHRASES):
        return False
    # niente liste/markdown/codice/tabelle
    if any(sym in a for sym in ["```", "|", "#"]):
        return False
    lines = a.split("\n")
    listish = sum(1 for ln in lines if ln[:2] in ("- ", "* ") or ln[:2].strip().rstrip(".").isdigit())
    if listish >= 2:
        return False
    return True


def main() -> None:
    tok = BPETokenizer.load(BPE)
    data = json.load(open(SRC, encoding="utf-8"))

    tokens: list[int] = []
    mask: list[int] = []
    kept = 0
    for d in data:
        c = d["conversations"]
        if len(c) != 2:
            continue
        q = c[0]["value"].strip()
        a = c[1]["value"].strip()
        if not is_clean(q, a):
            continue
        prompt_ids = tok.encode(PROMPT_TPL.format(q=q))
        answer_ids = tok.encode(a + ANSWER_SUFFIX)
        tokens.extend(prompt_ids)
        mask.extend([0] * len(prompt_ids))     # loss NON sulla domanda
        tokens.extend(answer_ids)
        mask.extend([1] * len(answer_ids))     # loss SOLO sulla risposta (+ chiusura)
        kept += 1

    tok_arr = np.array(tokens, dtype=np.uint16)
    mask_arr = np.array(mask, dtype=np.uint8)
    tok_arr.tofile(OUT / "sft_tokens.bin")
    mask_arr.tofile(OUT / "sft_mask.bin")
    print(f"[sft] {kept:,} esempi tenuti (su {len(data):,})")
    print(f"[sft] {len(tok_arr)/1e6:.1f}M token totali, {mask_arr.mean()*100:.0f}% con loss (risposte)")
    print(f"[sft] -> {OUT/'sft_tokens.bin'} , {OUT/'sft_mask.bin'}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
