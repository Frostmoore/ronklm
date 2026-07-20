"""Addestra il BPE e tokenizza il corpus in binario (Fase 11.3).

Due passi:
  1. TRAIN del BPE su un CAMPIONE del corpus (non su tutto): le fusioni frequenti si
     imparano benissimo da qualche decina di MB, e addestrare su 4 GB costerebbe ore
     senza cambiare quasi nulla. E' prassi standard.
  2. ENCODE di tutti gli shard in un unico file binario di uint16, con multiprocessing.

PERCHE' pre-tokenizzare in binario invece che al volo durante il training: tokenizzare
costa CPU, e rifarlo a ogni epoca terrebbe la GPU affamata (regola: il collo di
bottiglia dev'essere la GPU, mai il caricamento dati). uint16 basta perche' il nostro
vocabolario e' < 65.536; 2 byte/token -> 1 mld di token = 2 GB su disco, che poi
leggiamo con np.memmap senza caricarli in RAM.

PERCHE' multiprocessing: il nostro BPE e' Python puro, ~1 MB/s per core. Su GB di testo
sarebbe un'ora e passa; con tutti i core scende a pochi minuti.

Uso:
    python data/corpus_b/tokenize_corpus.py --vocab-size 16384
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ronklm.bpe import BPETokenizer  # noqa: E402

DEFAULT_TEXT = Path("D:/RonkLM_corpus/text")
DEFAULT_OUT = Path("D:/RonkLM_corpus/tokens")

# PERCHE' lasciamo core liberi (e non usiamo tutti e 24): con la CPU al 100% il PC
# diventa inusabile e persino fermare il processo puo' richiedere secondi. Lasciando
# qualche core al sistema, la macchina resta reattiva e il lavoro dura poco di piu'.
# Regolabile con --workers; il default e' ~2/3 dei core.
def default_workers() -> int:
    n = os.cpu_count() or 4
    return max(1, min(n - 6, int(n * 0.66)))


_TOK: BPETokenizer | None = None  # per-processo, inizializzato una volta sola


def _init_worker(tok_path: str) -> None:
    """Ogni processo carica il tokenizer una volta (non a ogni chunk), e si mette a
    priorita' bassa: cosi' qualunque altra cosa tu stia facendo ha la precedenza."""
    global _TOK
    try:  # priorita' bassa (Windows e POSIX)
        import psutil  # type: ignore

        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        try:
            os.nice(10)  # POSIX
        except Exception:
            pass
    _TOK = BPETokenizer.load(tok_path)


def _encode_chunk(text: str) -> np.ndarray:
    assert _TOK is not None
    return np.asarray(_TOK.encode(text), dtype=np.uint16)


def _iter_chunks(paths: list[Path], chunk_chars: int):
    """Legge gli shard e li spezza in blocchi, tagliando SEMPRE su un confine di
    documento (la riga vuota tripla) per non spezzare una parola a meta'."""
    for p in paths:
        text = p.read_text(encoding="utf-8", newline="")
        start = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            if end < len(text):
                cut = text.rfind("\n\n\n", start, end)
                if cut > start:
                    end = cut + 3
            yield text[start:end]
            start = end


def main() -> None:
    ap = argparse.ArgumentParser(description="Addestra il BPE e tokenizza il corpus.")
    ap.add_argument("--text-dir", type=Path, default=DEFAULT_TEXT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--sample-mb", type=int, default=40, help="MB di testo per addestrare il BPE")
    ap.add_argument("--val-frac", type=float, default=0.005, help="frazione tenuta per la validation")
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help="processi paralleli (default: ~2/3 dei core, per tenere il PC reattivo)")
    ap.add_argument("--chunk-chars", type=int, default=2_000_000)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(args.text_dir.glob("wiki_it_*.txt"))
    if not shards:
        sys.exit(f"nessuno shard in {args.text_dir}: esegui prima extract_wikipedia.py")
    total_mb = sum(p.stat().st_size for p in shards) / 1024**2
    print(f"[corpus] {len(shards)} shard, {total_mb:,.0f} MB")

    # --- 1. addestra il BPE su un campione -------------------------------
    tok_path = args.out_dir / f"bpe_{args.vocab_size}.pkl"
    if tok_path.exists():
        print(f"[bpe] gia' addestrato: {tok_path.name}")
        tok = BPETokenizer.load(tok_path)
    else:
        need = args.sample_mb * 1024 * 1024
        sample_parts, got = [], 0
        for p in shards:                       # campiona da shard diversi
            t = p.read_text(encoding="utf-8", newline="")[: need // min(4, len(shards)) + 1]
            sample_parts.append(t)
            got += len(t.encode("utf-8"))
            if got >= need:
                break
        sample = "".join(sample_parts)
        print(f"[bpe] addestro su {len(sample)/1024**2:.0f} MB -> vocab {args.vocab_size}")
        t0 = time.time()
        tok = BPETokenizer().train(sample, vocab_size=args.vocab_size, verbose=True)
        tok.save(tok_path)
        print(f"[bpe] fatto in {(time.time()-t0)/60:.1f} min -> {tok_path.name}")
        del sample, sample_parts

    # --- 2. tokenizza tutto in parallelo, SCRIVENDO IN STREAMING ---------
    # PERCHE' in streaming e non accumulando in RAM: 1,2 mld di token sono 2,4 GB come
    # uint16, e la concatenazione finale ne richiederebbe altrettanti (picco ~5 GB).
    # Scrivendo ogni blocco appena pronto, la memoria resta di pochi MB.
    print(f"[encode] {args.workers} processi (priorita' bassa, {os.cpu_count()} core totali)")
    t0 = time.time()
    all_path = args.out_dir / "all.bin"
    n = 0
    with open(all_path, "wb") as out, \
            Pool(args.workers, initializer=_init_worker, initargs=(str(tok_path),)) as pool:
        for i, arr in enumerate(pool.imap(_encode_chunk, _iter_chunks(shards, args.chunk_chars), chunksize=1)):
            out.write(arr.tobytes())
            n += len(arr)
            if i % 20 == 0:
                el = time.time() - t0
                print(f"  blocco {i:5d}  {n/1e6:8.1f}M token  "
                      f"({el/60:.1f} min, {n/1e6/max(el,1e-9):.2f}M tok/s)", flush=True)

    # --- 3. split train/val -----------------------------------------------
    # Split in CODA e contiguo (stessa logica della Fase 0.4): la validation e' un
    # blocco di testo che il modello non vede mai durante il training.
    n_val = int(n * args.val_frac)
    src = np.memmap(all_path, dtype=np.uint16, mode="r", shape=(n,))
    with open(args.out_dir / "train.bin", "wb") as f:      # copia a blocchi: RAM costante
        step = 50_000_000
        for s in range(0, n - n_val, step):
            f.write(src[s: min(s + step, n - n_val)].tobytes())
    with open(args.out_dir / "val.bin", "wb") as f:
        f.write(src[n - n_val:].tobytes())
    del src
    all_path.unlink()   # il file intero non serve piu': liberiamo 2,4 GB

    ratio = (total_mb * 1024**2) / max(1, n)
    print(f"\n[fine] {n/1e6:,.1f}M token totali  ({n*2/1024**3:.2f} GB)")
    print(f"       train {(n-n_val)/1e6:,.1f}M  |  val {n_val/1e6:,.1f}M")
    print(f"       compressione: {ratio:.2f} byte/token")
    print(f"       tempo: {(time.time()-t0)/60:.1f} min")
    (args.out_dir / "meta.txt").write_text(
        f"vocab_size={tok.vocab_size}\ntokens={n}\ntrain={n - n_val}\nval={n_val}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
