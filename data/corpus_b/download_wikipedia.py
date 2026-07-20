"""Scarica il dump ZIM di Wikipedia italiana da Kiwix (Percorso B, Fase 11).

PERCHE' uno script e non un download a mano: la stessa ragione della Fase 0.2 --
riproducibilita'. Questo file documenta ESATTAMENTE quale dump abbiamo usato.

PERCHE' su disco veloce (D:, NVMe) e non nel repo (E:, HDD SATA): il file binario dei
token prodotto a valle verra' letto ad ACCESSO CASUALE a ogni batch durante il
training. Da un disco meccanico il caricamento dati diventerebbe il collo di
bottiglia e la GPU resterebbe affamata (regola: in un training ben fatto il collo di
bottiglia dev'essere la GPU, mai il disco).

Uso:
    python data/corpus_b/download_wikipedia.py
    python data/corpus_b/download_wikipedia.py --variant mini   # versione leggera
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

# Mirror. PERCHE' non usiamo l'origine: misurato il 2026-07-19, download.kiwix.org
# dava ~3,8 MB/s mentre il mirror dotsrc ~9,1 MB/s (su fibra 1 Gbit: e' il server a
# limitare, non la linea). Su 8,3 GB la differenza e' 2 ore contro 15 minuti.
MIRRORS = {
    "dotsrc": "https://mirrors.dotsrc.org/kiwix/zim/wikipedia/",       # il piu' veloce nei test
    "umu": "https://saimei.ftp.acc.umu.se/mirror/kiwix.org/zim/wikipedia/",
    "kiwix": "https://download.kiwix.org/zim/wikipedia/",              # origine
}
BASE = MIRRORS["dotsrc"]

# Varianti disponibili (verificate il 2026-07-19):
#   all_nopic : Wikipedia IT completa, articoli interi, senza immagini  -> 8,29 GB
#   all_mini  : solo le introduzioni degli articoli                     -> 2,23 GB
#   top_nopic : solo gli articoli piu' visitati                         -> 0,82 GB
VARIANTS = {
    "all_nopic": "wikipedia_it_all_nopic_2026-05.zim",
    "all_mini": "wikipedia_it_all_mini_2026-05.zim",
    "top_nopic": "wikipedia_it_top_nopic_2026-07.zim",
}

DEFAULT_DEST = Path("D:/RonkLM_corpus")


def download(url: str, dest: Path) -> None:
    """Scarica `url` in `dest`, RIPRENDENDO se esiste gia' un file parziale.

    La ripresa usa l'header HTTP `Range`: chiediamo al server solo i byte mancanti.
    Utile sia dopo un'interruzione sia per cambiare mirror a meta' strada (il file e'
    identico ovunque, quindi i byte gia' presi restano validi).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    # dimensione totale attesa
    head = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0 RonkLM"})
    total = int(urllib.request.urlopen(head, timeout=60).headers.get("Content-Length", 0))

    already = dest.stat().st_size if dest.exists() else 0
    if already == total and total > 0:
        print(f"[skip] {dest.name} gia' completo ({total/1024**3:.2f} GB)")
        return
    if already > total:
        print(f"[warn] file locale piu' grande dell'atteso: riparto da zero")
        already = 0

    headers = {"User-Agent": "Mozilla/5.0 RonkLM"}
    if already:
        headers["Range"] = f"bytes={already}-"
        print(f"[ripresa] {already/1024**3:.2f} GB gia' presenti, scarico il resto")
    print(f"[rete] {url}")
    print(f"[out]  {dest}  ({total/1024**3:.2f} GB totali)")

    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    done = already
    last = 0.0
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "ab" if already else "wb") as f:
        while True:
            chunk = r.read(1024 * 1024 * 4)  # 4 MB
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            now = time.time()
            if now - last > 5.0:
                last = now
                speed = (done - already) / (now - t0) / 1024**2
                pct = 100 * done / total if total else 0
                eta = (total - done) / max(1e-9, (done - already) / (now - t0))
                print(f"  {pct:5.1f}%  {done/1024**3:6.2f} GB  {speed:6.1f} MB/s  ETA {eta/60:4.1f} min", flush=True)
    dt = time.time() - t0
    print(f"[ok] completato in {dt:.0f}s (media {(done-already)/dt/1024**2:.1f} MB/s)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Scarica Wikipedia IT (ZIM) da Kiwix.")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="all_nopic")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = ap.parse_args()

    name = VARIANTS[args.variant]
    download(BASE + name, args.dest / name)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
