"""Estrae il testo pulito dal dump ZIM di Wikipedia italiana (Fase 11.2).

Uno ZIM contiene gli articoli in HTML. Qui li scorriamo, togliamo il markup, filtriamo
la robaccia e deduplichiamo, producendo file di testo semplice pronti per il tokenizer.

PERCHE' i filtri (e non "prendiamo tutto"): un corpus sporco insegna cose sbagliate.
Wikipedia e' piena di pagine che NON sono prosa: disambiguazioni, liste, redirect,
categorie, stub di due righe. Se le lasciamo, il modello impara a generare elenchi
puntati invece che italiano.

PERCHE' la deduplicazione: i duplicati (a) fanno MEMORIZZARE invece di generalizzare,
(b) contaminano la validation -- se lo stesso testo sta in train e in val, la loss di
validation mente, e quella e' l'unica bussola di cui ci fidiamo (piano 4.3).

Uso:
    python data/corpus_b/extract_wikipedia.py
    python data/corpus_b/extract_wikipedia.py --limit 5000     # prova veloce
"""
from __future__ import annotations

import argparse
import hashlib
import html
import os
import re
import sys
import time
from pathlib import Path

DEFAULT_ZIM = Path("D:/RonkLM_corpus/wikipedia_it_all_nopic_2026-05.zim")
DEFAULT_OUT = Path("D:/RonkLM_corpus/text")

# --- pulizia HTML -----------------------------------------------------------
# Elementi il cui CONTENUTO va buttato, non solo i tag.
RE_DROP_BLOCKS = re.compile(
    r"<(script|style|table|sup|figure|figcaption|noscript)\b.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
# Sezioni finali non-narrative (note, bibliografia, collegamenti esterni...).
RE_TAIL_SECTIONS = re.compile(
    r"<h[23][^>]*>\s*(Note|Bibliografia|Voci correlate|Altri progetti|"
    r"Collegamenti esterni|Riferimenti|Fonti)\s*</h[23]>.*",
    re.IGNORECASE | re.DOTALL,
)
RE_TAG = re.compile(r"<[^>]+>")
RE_MULTISPACE = re.compile(r"[ \t ]+")
RE_MULTINEWLINE = re.compile(r"\n{3,}")
RE_REF_MARKS = re.compile(r"\[\d+\]")          # note tipo [1], [23]

# Boilerplate di licenza che Kiwix mette in fondo a OGNI articolo. Se lo lasciassimo,
# il modello lo vedrebbe milioni di volte e imparerebbe a recitarlo a memoria.
RE_LICENSE_FOOTER = re.compile(
    r"Questa voce è stata pubblicata da Wikipedia.*", re.IGNORECASE | re.DOTALL
)


def _collapse_repeated_lines(s: str) -> str:
    """Elimina righe identiche CONSECUTIVE.

    Serve perche' l'HTML degli articoli ZIM contiene il titolo piu' volte (tag <title>,
    intestazione, <h1>): senza questo ogni documento comincia col titolo ripetuto 2-3
    volte, e il modello imparerebbe quel tic.

    NB: il confronto va fatto con l'ultima riga NON VUOTA, non con quella
    immediatamente precedente: nell'HTML i titoli ripetuti sono separati da righe
    vuote, quindi confrontando gli adiacenti non se ne eliminava nemmeno uno."""
    out: list[str] = []
    last_non_empty = None
    for line in s.split("\n"):
        if line and line == last_non_empty:
            continue
        out.append(line)
        if line:
            last_non_empty = line
    return "\n".join(out)


def html_to_text(raw: str) -> str:
    """HTML di un articolo -> testo semplice."""
    s = RE_TAIL_SECTIONS.sub("", raw)
    s = RE_DROP_BLOCKS.sub(" ", s)
    # i blocchi diventano a-capo, cosi' i paragrafi restano separati
    s = re.sub(r"</(p|div|h[1-6]|li|tr|br)>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.IGNORECASE)
    s = RE_TAG.sub("", s)
    s = html.unescape(s)
    s = RE_LICENSE_FOOTER.sub("", s)
    s = RE_REF_MARKS.sub("", s)
    s = s.replace(" ", " ").replace("\r", "")
    s = RE_MULTISPACE.sub(" ", s)
    s = "\n".join(line.strip() for line in s.split("\n"))
    s = _collapse_repeated_lines(s)
    s = RE_MULTINEWLINE.sub("\n\n", s)
    return s.strip()


# --- filtri di qualita' -----------------------------------------------------
BAD_TITLE_PREFIXES = ("Categoria:", "Portale:", "Template:", "Aiuto:", "Discussione:",
                      "Wikipedia:", "File:", "Modulo:", "Progetto:")
RE_ALPHA = re.compile(r"[^\W\d_]", re.UNICODE)


def keep_article(title: str, text: str, min_chars: int) -> bool:
    """Decide se un articolo entra nel corpus.

    NB: il filtro delle disambiguazioni guarda il TESTO, non il titolo. Le pagine di
    disambiguazione hanno titoli normalissimi ("$5,000 Reward"); il marcatore sta nel
    corpo ("Questa è una pagina di disambiguazione"). Cercarlo nel titolo -- come
    facevo nella prima versione -- non ne prendeva nemmeno una.
    """
    if any(title.startswith(p) for p in BAD_TITLE_PREFIXES):
        return False
    if len(text) < min_chars:
        return False                      # stub troppo corti: poco segnale, molto rumore

    head = text[:1200].lower()
    if "pagina di disambiguazione" in head or "disambigua" in title.lower():
        return False                      # elenchi di rimandi, non prosa
    if "a questo titolo corrispondono" in head:
        return False

    # Pagine-lista ("Elenco di...", "Cronologia di..."): righe corte in serie, non prosa.
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if lines:
        short = sum(1 for ln in lines if len(ln) < 80)
        if short / len(lines) > 0.8 and len(lines) > 10:
            return False

    # proporzione di lettere: scarta tabelle di numeri, liste di codici, ecc.
    alpha = len(RE_ALPHA.findall(text[:2000]))
    if alpha / max(1, len(text[:2000])) < 0.6:
        return False
    return True


# --- estrazione -------------------------------------------------------------
def extract(zim_path: Path, out_dir: Path, limit: int | None, min_chars: int,
            shard_mb: int) -> None:
    from libzim.reader import Archive

    out_dir.mkdir(parents=True, exist_ok=True)
    archive = Archive(str(zim_path))
    total = archive.all_entry_count
    print(f"[zim]  {zim_path.name}: {total:,} voci, {archive.article_count:,} articoli")

    seen: set[str] = set()          # hash dei contenuti gia' visti (deduplicazione)
    kept = skipped = dup = 0
    shard_idx = 0
    shard_bytes = 0
    buf: list[str] = []
    t0 = time.time()
    limit = limit or total

    def flush() -> None:
        nonlocal shard_idx, shard_bytes, buf
        if not buf:
            return
        p = out_dir / f"wiki_it_{shard_idx:04d}.txt"
        p.write_text("".join(buf), encoding="utf-8", newline="\n")
        print(f"  [shard] {p.name}  {shard_bytes/1024**2:.0f} MB  ({kept:,} articoli)", flush=True)
        shard_idx += 1
        shard_bytes = 0
        buf = []

    for i in range(min(limit, total)):
        try:
            entry = archive._get_entry_by_id(i)
            if entry.is_redirect:
                continue
            item = entry.get_item()
            if not str(item.mimetype).startswith("text/html"):
                continue
            raw = bytes(item.content).decode("utf-8", errors="replace")
            title = entry.title
            text = html_to_text(raw)
            if not keep_article(title, text, min_chars):
                skipped += 1
                continue
            h = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
            if h in seen:
                dup += 1
                continue
            seen.add(h)
            # NB: NON anteponiamo il titolo -- il testo estratto lo contiene gia'
            # (era la quarta ripetizione). Separatore fra documenti: tripla riga vuota.
            chunk = f"{text}\n\n\n"
            buf.append(chunk)
            shard_bytes += len(chunk.encode("utf-8"))
            kept += 1
            if shard_bytes >= shard_mb * 1024 * 1024:
                flush()
        except Exception:
            skipped += 1
            continue
        if i % 50_000 == 0 and i:
            el = time.time() - t0
            print(f"  {i:,}/{min(limit,total):,}  tenuti {kept:,}  scartati {skipped:,}  "
                  f"dup {dup:,}  ({el:.0f}s, {i/el:.0f} voci/s)", flush=True)

    flush()
    el = time.time() - t0
    print(f"\n[fine] tenuti {kept:,} articoli | scartati {skipped:,} | duplicati {dup:,}")
    print(f"       {shard_idx} shard in {out_dir}  ({el/60:.1f} min)")


def _worker(job: tuple) -> tuple[int, int, int]:
    """Estrae l'intervallo [start, end) di voci in shard propri. Ritorna i conteggi."""
    zim_path, out_dir, start, end, min_chars, shard_mb, wid = job
    from libzim.reader import Archive

    try:  # priorita' bassa: il PC deve restare usabile
        import psutil  # type: ignore

        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass

    archive = Archive(str(zim_path))
    seen: set[str] = set()
    kept = skipped = dup = 0
    shard_idx = 0
    shard_bytes = 0
    buf: list[str] = []

    def flush() -> None:
        nonlocal shard_idx, shard_bytes, buf
        if not buf:
            return
        p = Path(out_dir) / f"wiki_it_w{wid:02d}_{shard_idx:03d}.txt"
        p.write_text("".join(buf), encoding="utf-8", newline="\n")
        shard_idx += 1
        shard_bytes = 0
        buf = []

    for i in range(start, end):
        try:
            entry = archive._get_entry_by_id(i)
            if entry.is_redirect:
                continue
            item = entry.get_item()
            if not str(item.mimetype).startswith("text/html"):
                continue
            text = html_to_text(bytes(item.content).decode("utf-8", errors="replace"))
            if not keep_article(entry.title, text, min_chars):
                skipped += 1
                continue
            h = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
            if h in seen:
                dup += 1
                continue
            seen.add(h)
            chunk = f"{text}\n\n\n"
            buf.append(chunk)
            shard_bytes += len(chunk.encode("utf-8"))
            kept += 1
            if shard_bytes >= shard_mb * 1024 * 1024:
                flush()
        except Exception:
            skipped += 1
    flush()
    return kept, skipped, dup


def extract_parallel(zim_path: Path, out_dir: Path, limit: int | None, min_chars: int,
                     shard_mb: int, workers: int) -> None:
    """Estrazione in parallelo: ogni processo prende un intervallo di ID e scrive shard propri.

    LIMITE NOTO E ACCETTATO: la deduplicazione e' PER PROCESSO, quindi due articoli
    identici capitati in intervalli diversi sopravvivono entrambi. Sui dati reali il
    tasso di duplicati esatti in Wikipedia e' bassissimo (0 su 6.000 nei test), quindi
    il compromesso vale i ~45 minuti risparmiati. Se un domani servisse dedup globale,
    si fa una passata sugli hash a valle.
    """
    from libzim.reader import Archive
    from multiprocessing import Pool

    out_dir.mkdir(parents=True, exist_ok=True)
    archive = Archive(str(zim_path))
    total = min(limit or archive.all_entry_count, archive.all_entry_count)
    del archive
    print(f"[zim] {zim_path.name}: {total:,} voci da processare, {workers} processi")

    step = (total + workers - 1) // workers
    jobs = [
        (zim_path, out_dir, w * step, min((w + 1) * step, total), min_chars, shard_mb, w)
        for w in range(workers)
    ]
    t0 = time.time()
    with Pool(workers) as pool:
        results = pool.map(_worker, jobs)
    kept = sum(r[0] for r in results)
    skipped = sum(r[1] for r in results)
    dup = sum(r[2] for r in results)
    n_shards = len(list(out_dir.glob("wiki_it_*.txt")))
    mb = sum(p.stat().st_size for p in out_dir.glob("wiki_it_*.txt")) / 1024**2
    el = time.time() - t0
    print(f"\n[fine] tenuti {kept:,} | scartati {skipped:,} | duplicati {dup:,}")
    print(f"       {n_shards} shard, {mb:,.0f} MB in {out_dir}  ({el/60:.1f} min)")


def default_workers() -> int:
    """~2/3 dei core: il resto resta al sistema, cosi' il PC non si pianta."""
    n = os.cpu_count() or 4
    return max(1, min(n - 6, int(n * 0.66)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Estrae testo pulito da uno ZIM Wikipedia.")
    ap.add_argument("--zim", type=Path, default=DEFAULT_ZIM)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None, help="processa solo N voci (prova)")
    ap.add_argument("--min-chars", type=int, default=400, help="lunghezza minima articolo")
    ap.add_argument("--shard-mb", type=int, default=256)
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help="processi paralleli (default ~2/3 dei core); 1 = sequenziale")
    args = ap.parse_args()
    if args.workers <= 1:
        extract(args.zim, args.out, args.limit, args.min_chars, args.shard_mb)
    else:
        extract_parallel(args.zim, args.out, args.limit, args.min_chars,
                         args.shard_mb, args.workers)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
