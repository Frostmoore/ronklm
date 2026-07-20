"""Estrae il testo dei libri dallo ZIM di Project Gutenberg italiano (Fase 11, storyteller).

DIVERSO da Wikipedia: qui i libri sono EPUB (zip di file XHTML), non pagine HTML. E
hanno il boilerplate di licenza Project Gutenberg da togliere -- gli STESSI marcatori
`*** START/END OF THE PROJECT GUTENBERG EBOOK ***` che avevamo gestito a mano per
Pinocchio nella Fase 0.2. Qui lo facciamo per ~1.087 libri in automatico.

PERCHE' Gutenberg per lo "storyteller": prosa narrativa (romanzi, novelle, teatro,
fiabe) invece del registro enciclopedico di Wikipedia. E' il corpus che spinge il
modello a raccontare invece che a compilare voci.

Uso:
    python data/corpus_b/extract_gutenberg.py
"""
from __future__ import annotations

import argparse
import hashlib
import html
import io
import os
import re
import sys
import time
import zipfile
from pathlib import Path

DEFAULT_ZIM = Path("D:/RonkLM_corpus/gutenberg_it_all_2026-01.zim")
DEFAULT_OUT = Path("D:/RonkLM_corpus/text_gutenberg")

RE_TAG = re.compile(r"<[^>]+>")
RE_MULTISPACE = re.compile(r"[ \t ]+")
RE_MULTINEWLINE = re.compile(r"\n{3,}")
# Boilerplate Project Gutenberg. NB: questi epub (spesso da BnF/Gallica) NON hanno i
# marcatori `*** START/END ***` (verificato: 0 occorrenze). Hanno invece un header
# ricorrente "The Project Gutenberg eBook of ...", i crediti "Produced by ...", e la
# licenza inglese ("This ebook is for the use of anyone ..."). Li togliamo con pattern
# mirati; se i marcatori *** ci sono (epub recenti), li usiamo come confini.
RE_PG_START = re.compile(r".*?\*\*\*\s*START OF TH[EI][^\n]*\*\*\*", re.IGNORECASE | re.DOTALL)
RE_PG_END = re.compile(r"\*\*\*\s*END OF TH[EI].*", re.IGNORECASE | re.DOTALL)
RE_PG_HEADER = re.compile(r"The Project Gutenberg eBook[^\n]*", re.IGNORECASE)
RE_PRODUCED = re.compile(r"Produced by.*?(?:\n\s*\n|\Z)", re.IGNORECASE | re.DOTALL)
RE_LICENSE = re.compile(r"This ebook is for the use of anyone.*?www\.gutenberg\.org\S*",
                        re.IGNORECASE | re.DOTALL)
RE_PG_FOOTER = re.compile(r"End of (the |this )?Project Gutenberg.*", re.IGNORECASE | re.DOTALL)
# Qualsiasi riga che nomina Gutenberg e' boilerplate: un romanzo italiano non lo fa mai.
# Cattura anche i footer di pagina "Titolo | Project Gutenberg" ripetuti a ogni pagina.
RE_PG_ANY_LINE = re.compile(
    r"^.*(project gutenberg|gutenberg\.org|gutenberg-tm|pglaf|distributed proofreading).*$",
    re.IGNORECASE | re.MULTILINE,
)


def _clean_pg_boilerplate(full: str) -> str:
    """Toglie tutte le forme note di boilerplate Project Gutenberg."""
    m = RE_PG_START.search(full)          # marcatori (epub recenti)
    if m:
        full = full[m.end():]
    full = RE_PG_END.sub("", full)
    full = RE_PG_FOOTER.sub("", full)     # footer testuale (epub vecchi)
    full = RE_LICENSE.sub("", full)       # licenza inglese
    full = RE_PRODUCED.sub("", full)      # crediti trascrittori
    full = RE_PG_HEADER.sub("", full)     # header ricorrente
    full = RE_PG_ANY_LINE.sub("", full)   # ogni riga residua che nomina Gutenberg
    return full


def epub_to_text(data: bytes) -> str:
    """EPUB (bytes) -> testo del libro, ripulito dal boilerplate Gutenberg."""
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return ""
    # concatena tutti gli XHTML in ordine di nome (approssima l'ordine del libro)
    parts = []
    for name in sorted(n for n in z.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))):
        try:
            raw = z.read(name).decode("utf-8", errors="replace")
        except Exception:
            continue
        # <br>/<p>/<div>/heading -> a-capo, cosi' i paragrafi restano separati
        raw = re.sub(r"<(br|/p|/div|/h[1-6]|/tr)\s*/?>", "\n", raw, flags=re.IGNORECASE)
        parts.append(html.unescape(RE_TAG.sub("", raw)))
    full = _clean_pg_boilerplate("\n".join(parts))

    full = full.replace("\r", "")
    # collassa righe identiche consecutive (titoli/header ripetuti tra i file XHTML)
    lines, prev = [], None
    for ln in full.split("\n"):
        ln = ln.strip()
        if ln and ln == prev:
            continue
        lines.append(ln)
        if ln:
            prev = ln
    full = RE_MULTISPACE.sub(" ", "\n".join(lines))
    full = RE_MULTINEWLINE.sub("\n\n", full)
    return full.strip()


def _worker(job: tuple) -> tuple[int, int, int]:
    zim_path, out_dir, ids, min_chars, shard_mb, wid = job
    from libzim.reader import Archive

    try:
        import psutil  # type: ignore
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass

    archive = Archive(str(zim_path))
    seen: set[str] = set()
    kept = skipped = dup = 0
    shard_idx = shard_bytes = 0
    buf: list[str] = []

    def flush():
        nonlocal shard_idx, shard_bytes, buf
        if not buf:
            return
        (Path(out_dir) / f"gut_it_w{wid:02d}_{shard_idx:03d}.txt").write_text(
            "".join(buf), encoding="utf-8", newline="\n")
        shard_idx += 1
        shard_bytes = 0
        buf = []

    for i in ids:
        try:
            data = bytes(archive._get_entry_by_id(i).get_item().content)
            text = epub_to_text(data)
            if len(text) < min_chars:
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
            if shard_bytes >= shard_mb * 1024 * 1024:
                flush()
            kept += 1
        except Exception:
            skipped += 1
    flush()
    return kept, skipped, dup


def main() -> None:
    ap = argparse.ArgumentParser(description="Estrae i libri dallo ZIM Gutenberg IT.")
    ap.add_argument("--zim", type=Path, default=DEFAULT_ZIM)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-chars", type=int, default=1000)  # i libri sono lunghi
    ap.add_argument("--shard-mb", type=int, default=128)
    ap.add_argument("--workers", type=int, default=max(1, min((os.cpu_count() or 4) - 6, int((os.cpu_count() or 4) * 0.66))))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from libzim.reader import Archive
    from multiprocessing import Pool

    archive = Archive(str(args.zim))
    print(f"[zim] {args.zim.name}: cerco gli EPUB tra {archive.all_entry_count:,} voci...")
    epub_ids = []
    for i in range(archive.all_entry_count):
        try:
            e = archive._get_entry_by_id(i)
            if not e.is_redirect and str(e.get_item().mimetype).startswith("application/epub"):
                epub_ids.append(i)
        except Exception:
            pass
    del archive
    print(f"[zim] {len(epub_ids)} libri EPUB, {args.workers} processi")

    # distribuisci gli id a rotazione tra i worker (bilancia il carico)
    buckets = [epub_ids[w::args.workers] for w in range(args.workers)]
    jobs = [(args.zim, args.out, buckets[w], args.min_chars, args.shard_mb, w)
            for w in range(args.workers)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        res = pool.map(_worker, jobs)
    kept = sum(r[0] for r in res)
    mb = sum(p.stat().st_size for p in args.out.glob("gut_it_*.txt")) / 1024**2
    print(f"\n[fine] {kept:,} libri tenuti | scartati {sum(r[1] for r in res)} | dup {sum(r[2] for r in res)}")
    print(f"       {mb:,.0f} MB in {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
