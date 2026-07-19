"""CLI di RonkLM (Fase 8): addestra e genera da riga di comando.

    # addestra e salva il miglior checkpoint su validation
    python scripts/train_ronklm.py train --steps 3000 --n-layer 4 --n-embd 96 \
        --out checkpoints/ronklm.npz

    # genera testo da un checkpoint
    python scripts/train_ronklm.py generate --ckpt checkpoints/ronklm.npz \
        --prompt "Pinocchio " --n 400 --temperature 0.8 --top-k 20

PERCHE' una CLI e non notebook: ogni esperimento resta riproducibile dal suo COMANDO
(che, col seed, individua univocamente il run). Modificare il codice a ogni prova
distruggerebbe la confrontabilita' tra esperimenti.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ronklm.dataset import Dataset  # noqa: E402
from ronklm.generate import generate  # noqa: E402
from ronklm.models.gpt import GPT, GPTConfig  # noqa: E402
from ronklm.train import evaluate, train  # noqa: E402


def cmd_train(args: argparse.Namespace) -> None:
    ds = Dataset.from_file(args.data)
    V = ds.tokenizer.vocab_size
    cfg = GPTConfig(
        vocab_size=V,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT(cfg, np.random.default_rng(args.seed))
    print(f"{model}  |  corpus {len(ds.data):,} char, vocab {V}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    result = train(
        model, ds, ds.tokenizer,
        steps=args.steps, batch_size=args.batch_size,
        base_lr=args.lr, min_lr=args.lr / 10, warmup=args.warmup,
        weight_decay=args.weight_decay, eval_every=args.eval_every,
        seed=args.seed, ckpt_path=args.out, log_path=args.log,
    )
    print(f"\nMigliore NLL val: {result['best_val']:.4f}")
    if args.out:
        print(f"Checkpoint (best) salvato in: {args.out}")


def cmd_generate(args: argparse.Namespace) -> None:
    model, tok = GPT.load(args.ckpt, np.random.default_rng(args.seed))
    context = tok.encode(args.prompt) if args.prompt else [0]
    out = generate(
        model, context, args.n, np.random.default_rng(args.seed),
        temperature=args.temperature, top_k=args.top_k,
    )
    text = tok.decode(context + out) if args.prompt else tok.decode(out)
    print(text)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RonkLM — addestra e genera.")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="addestra un GPT")
    t.add_argument("--data", default="data/input.txt")
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--block-size", type=int, default=64)
    t.add_argument("--n-layer", type=int, default=4)
    t.add_argument("--n-head", type=int, default=4)
    t.add_argument("--n-embd", type=int, default=96)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--warmup", type=int, default=100)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--eval-every", type=int, default=250)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--out", default="checkpoints/ronklm.npz")
    t.add_argument("--log", default=None, help="file CSV con la curva di loss")
    t.set_defaults(func=cmd_train)

    g = sub.add_parser("generate", help="genera testo da un checkpoint")
    g.add_argument("--ckpt", required=True)
    g.add_argument("--prompt", default="")
    g.add_argument("--n", type=int, default=400)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=None)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_generate)
    return p


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args()
    args.func(args)
