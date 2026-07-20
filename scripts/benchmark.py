"""Benchmark dei motori (Fase 9.3): quanto e' piu' veloce PyTorch, e la GPU?

Misura i token/secondo di TRAINING (forward + backward + update) su tre motori:
    1. NumPy-CPU  (ronkgrad, il nostro Percorso A)
    2. torch-CPU
    3. torch-CUDA (se disponibile)  [+ variante bf16 mixed precision]

PERCHE' misurare invece di citare: i "3-4 ordini di grandezza" promessi nel piano
(I.2) diventano un numero NOSTRO, sul NOSTRO modello, sulla NOSTRA GPU. E il
risultato dimensiona la Fase 12: dato il budget di ~7-8 ore, quanti token possiamo
processare -> quale coppia (dimensione modello, epoche) e' compute-optimal.

Uso:
    python scripts/benchmark.py                    # config della Fase 7 (piccola)
    python scripts/benchmark.py --n-layer 12 --n-embd 768 --block-size 512 --batch-size 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ronklm.models.gpt import GPT as GPTNumpy, GPTConfig as GPTConfigNumpy  # noqa: E402
from ronklm.optim import AdamW as AdamWNumpy  # noqa: E402


def bench_numpy(cfg, batch_size: int, steps: int) -> float:
    """Ritorna token/secondo del motore NumPy."""
    m = GPTNumpy(cfg, np.random.default_rng(0))
    opt = AdamWNumpy(m.parameters(), lr=1e-3)
    rng = np.random.default_rng(1)
    X = rng.integers(0, cfg.vocab_size, size=(batch_size, cfg.block_size))
    Y = rng.integers(0, cfg.vocab_size, size=(batch_size, cfg.block_size))
    # warmup
    m.zero_grad(); m.loss(X, Y).backward(); opt.step()
    t0 = time.perf_counter()
    for _ in range(steps):
        m.zero_grad()
        m.loss(X, Y).backward()
        opt.step()
    dt = time.perf_counter() - t0
    return steps * batch_size * cfg.block_size / dt


def bench_torch(cfg, batch_size: int, steps: int, device: str, amp: bool = False) -> float:
    """Ritorna token/secondo del motore torch su `device`. amp=True -> bf16 autocast."""
    import torch

    from ronklm_torch.model import GPT as GPTTorch, config_from_numpy

    dev = torch.device(device)
    m = GPTTorch(config_from_numpy(cfg)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(1)
    X = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size), generator=g).to(dev)
    Y = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size), generator=g).to(dev)

    def one_step():
        opt.zero_grad(set_to_none=True)
        if amp:
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                _, loss = m(X, Y)
        else:
            _, loss = m(X, Y)
        loss.backward()
        opt.step()

    for _ in range(3):          # warmup (include compilazione kernel / alloc)
        one_step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        one_step()
    if dev.type == "cuda":
        torch.cuda.synchronize()   # la GPU e' asincrona: sincronizza prima di fermare il cronometro
    dt = time.perf_counter() - t0
    return steps * batch_size * cfg.block_size / dt


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark NumPy vs torch (CPU/GPU).")
    ap.add_argument("--vocab-size", type=int, default=69)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--n-layer", type=int, default=3)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--skip-numpy", action="store_true", help="salta il motore NumPy (lento su config grandi)")
    args = ap.parse_args()

    cfg = GPTConfigNumpy(vocab_size=args.vocab_size, block_size=args.block_size,
                         n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    m = GPTNumpy(cfg, np.random.default_rng(0))
    print(f"Config: layer={cfg.n_layer} head={cfg.n_head} embd={cfg.n_embd} "
          f"block={cfg.block_size} batch={args.batch_size} | parametri: {m.num_params():,}")
    print(f"Token per passo: {args.batch_size * cfg.block_size:,}\n")

    results: dict[str, float] = {}

    if not args.skip_numpy:
        print("misuro NumPy-CPU ...", flush=True)
        results["NumPy-CPU (ronkgrad)"] = bench_numpy(cfg, args.batch_size, args.steps)

    import torch
    print("misuro torch-CPU ...", flush=True)
    results["torch-CPU"] = bench_torch(cfg, args.batch_size, args.steps, "cpu")

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"misuro torch-CUDA ({name}) ...", flush=True)
        results[f"torch-CUDA fp32"] = bench_torch(cfg, args.batch_size, args.steps, "cuda")
        print("misuro torch-CUDA bf16 (mixed precision) ...", flush=True)
        results["torch-CUDA bf16"] = bench_torch(cfg, args.batch_size, args.steps, "cuda", amp=True)
    else:
        print("[!] CUDA non disponibile: torch e' installato in versione CPU-only.")
        print("    Per la GPU serve la build CUDA (vedi piano, Fase 9.3).")

    base = next(iter(results.values()))
    print("\n===== RISULTATI (token/secondo di training) =====")
    for k, v in results.items():
        print(f"  {k:28s} {v:12,.0f} tok/s   ({v / base:6.1f}x)")

    # stima pratica per la Fase 12
    best = max(results.values())
    for hours in (8,):
        print(f"\n  Al ritmo migliore: {best * 3600 * hours / 1e9:.2f} miliardi di token in {hours} ore.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
