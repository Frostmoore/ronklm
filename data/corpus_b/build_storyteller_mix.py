"""Costruisce il corpus misto per il continued-pretraining "narratore" (storyteller).

Strategia (decisa con l'utente): partire dal pilota Wikipedia (che gia' SA cose) e
CONTINUARE l'addestramento su dati Gutenberg-dominanti, cosi' lo stile vira al
narrativo mentre il sapere di Wikipedia resta.

Composizione del file di training:
  - Gutenberg (~100M token): la parte dominante -> registro da romanzo ottocentesco;
  - una FETTA di Wikipedia (~25M token): "replay" per non dimenticare del tutto il
    linguaggio moderno e i fatti (evita il catastrophic forgetting).
  Rapporto ~80/20 a favore di Gutenberg.

La validation e' una coda di SOLO Gutenberg: cosi' la loss misura proprio quanto il
modello e' bravo nel registro-obiettivo (il narrativo), non nell'enciclopedico.

PERCHE' basta poco Wikipedia: il modello di partenza ha gia' visto TUTTA Wikipedia nel
pretraining; qui la fetta serve solo come promemoria, non come nuova conoscenza.

Uso:
    python data/corpus_b/build_storyteller_mix.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

TOK = Path("D:/RonkLM_corpus/tokens")


def main() -> None:
    ap = argparse.ArgumentParser(description="Costruisce il mix Gutenberg-dominante.")
    ap.add_argument("--tokens", type=Path, default=TOK)
    ap.add_argument("--out", type=Path, default=Path("D:/RonkLM_corpus/tokens_story"))
    ap.add_argument("--wiki-slice", type=int, default=25_000_000, help="token di Wikipedia (replay)")
    ap.add_argument("--val", type=int, default=2_000_000, help="token di validation (coda Gutenberg)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    gut = np.memmap(args.tokens / "gutenberg.bin", dtype=np.uint16, mode="r")
    wiki = np.memmap(args.tokens / "train.bin", dtype=np.uint16, mode="r")
    print(f"[dati] gutenberg {len(gut)/1e6:.1f}M | wikipedia {len(wiki)/1e6:.1f}M disponibili")

    # validation: coda di Gutenberg (il registro-obiettivo)
    val = np.asarray(gut[len(gut) - args.val:])
    gut_train = np.asarray(gut[: len(gut) - args.val])
    wiki_slice = np.asarray(wiki[: args.wiki_slice])

    # training: Gutenberg + fetta Wikipedia. Mescoliamo a blocchi (non token per token,
    # per non spezzare il contesto) cosi' il modello alterna i due registri durante il
    # training invece di vedere prima tutto Gutenberg e poi tutto Wikipedia.
    rng = np.random.default_rng(args.seed)
    block = 8192
    g_blocks = [gut_train[i:i + block] for i in range(0, len(gut_train), block)]
    w_blocks = [wiki_slice[i:i + block] for i in range(0, len(wiki_slice), block)]
    # lista di blocchi etichettati, poi mescolata: cosi' il modello alterna i due
    # registri durante il training invece di vedere prima tutto Gutenberg e poi Wikipedia.
    tagged = [("g", b) for b in g_blocks] + [("w", b) for b in w_blocks]
    order = rng.permutation(len(tagged))
    train = np.concatenate([tagged[i][1] for i in order])

    train.tofile(args.out / "train.bin")
    val.tofile(args.out / "val.bin")
    ratio = len(gut_train) / max(1, len(gut_train) + len(wiki_slice)) * 100
    print(f"[out]  train {len(train)/1e6:.1f}M (Gutenberg {ratio:.0f}% / Wikipedia {100-ratio:.0f}%)")
    print(f"       val   {len(val)/1e6:.1f}M (solo Gutenberg)")
    print(f"       -> {args.out}")
    (args.out / "meta.txt").write_text(
        f"train={len(train)}\nval={len(val)}\ngutenberg={len(gut_train)}\nwiki_replay={len(wiki_slice)}\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
