"""CLI per addestrare RonkLM su GPU (Percorso B, Fase 12).

    # pilota ~50M (una notte breve)
    python scripts/train_torch.py --n-layer 10 --n-embd 512 --n-head 8 --steps 18000

    # generazione dal checkpoint
    python scripts/train_torch.py generate --ckpt D:/RonkLM_corpus/run50m/best.pt --prompt "L'Italia è"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ronklm.bpe import BPETokenizer  # noqa: E402
from ronklm_torch.model import GPT, GPTConfig  # noqa: E402
from ronklm_torch.train import TrainConfig, train  # noqa: E402

DEFAULT_TOKENS = Path("D:/RonkLM_corpus/tokens")


def cmd_train(args: argparse.Namespace) -> None:
    tok = BPETokenizer.load(args.bpe)
    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    # fast=True e' OBBLIGATORIO alla scala vera: la variante didattica satura la VRAM
    # e crolla (misurato in Fase 9.3: 29,4 GB e 1.904 tok/s contro 12,7 GB e 78.314).
    model = GPT(cfg, fast=True)
    print(f"[modello] {model}")

    tcfg = TrainConfig(
        train_bin=args.tokens / "train.bin",
        val_bin=args.tokens / "val.bin",
        out_dir=args.out,
        steps=args.steps,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        block_size=args.block_size,
        base_lr=args.lr,
        min_lr=args.lr / 10,
        warmup=args.warmup,
        eval_every=args.eval_every,
        compile=args.compile,
    )
    res = train(model, tcfg)
    print(f"\n[fine] miglior val loss {res['best_val']:.4f} in {res['minutes']:.1f} min")
    print(f"       checkpoint: {args.out / 'best.pt'}")


@torch.no_grad()
def cmd_generate(args: argparse.Namespace) -> None:
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    model = GPT(ck["config"], fast=True).cuda().eval()
    model.load_state_dict(ck["model"])
    tok = BPETokenizer.load(args.bpe)
    ids = tok.encode(args.prompt) if args.prompt else [tok.encode("\n")[0]]
    ctx = torch.tensor(ids, dtype=torch.long, device="cuda")[None, :]
    for _ in range(args.n):
        window = ctx[:, -model.config.block_size:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.logits(window)
        logits = logits[0, -1].float() / max(args.temperature, 1e-6)
        if args.top_k:
            kth = torch.topk(logits, min(args.top_k, logits.numel())).values[-1]
            logits = torch.where(logits < kth, torch.full_like(logits, -float("inf")), logits)
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        ctx = torch.cat([ctx, nxt[None, :]], dim=1)
    print(tok.decode(ctx[0].tolist()))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RonkLM su GPU (Percorso B).")
    sub = p.add_subparsers(dest="cmd")

    t = sub.add_parser("train", help="addestra")
    t.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    t.add_argument("--bpe", type=Path, default=DEFAULT_TOKENS / "bpe_16384.pkl")
    t.add_argument("--out", type=Path, default=Path("D:/RonkLM_corpus/run50m"))
    t.add_argument("--n-layer", type=int, default=10)
    t.add_argument("--n-head", type=int, default=8)
    t.add_argument("--n-embd", type=int, default=512)
    t.add_argument("--block-size", type=int, default=512)
    t.add_argument("--micro-batch", type=int, default=32)   # tetto misurato in Fase 9.3
    t.add_argument("--grad-accum", type=int, default=4)
    t.add_argument("--steps", type=int, default=18000)
    t.add_argument("--lr", type=float, default=6e-4)
    t.add_argument("--warmup", type=int, default=400)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--compile", action="store_true")
    t.set_defaults(func=cmd_train)

    g = sub.add_parser("generate", help="genera testo")
    g.add_argument("--ckpt", type=Path, required=True)
    g.add_argument("--bpe", type=Path, default=DEFAULT_TOKENS / "bpe_16384.pkl")
    g.add_argument("--prompt", default="")
    g.add_argument("--n", type=int, default=300)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=50)
    g.set_defaults(func=cmd_generate)
    return p


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    a = build_parser().parse_args()
    if not getattr(a, "cmd", None):
        build_parser().print_help()
        sys.exit(1)
    a.func(a)
