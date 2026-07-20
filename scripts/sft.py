"""SFT — fine-tuning supervisionato di RonkLM-1 a rispondere (Fase 13).

Parte dal narratore (continued-pretrain su Gutenberg) e gli insegna il formato
domanda->risposta, con la LOSS SOLO SUI TOKEN DELLA RISPOSTA (maschera preparata da
prep_sft.py). La voce narrativa viene ereditata dal pretraining Gutenberg; qui si
insegna solo a *rispondere*, non lo stile.

Uso:
    python scripts/sft.py train
    python scripts/sft.py chat --prompt "Cos'è l'amicizia?"
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ronklm.bpe import BPETokenizer  # noqa: E402
from ronklm_torch.model import GPT  # noqa: E402

SFT = Path("D:/RonkLM_corpus/sft")
BPE = Path("D:/RonkLM_corpus/tokens/bpe_16384.pkl")
NARRATOR = Path("D:/RonkLM_corpus/run_story/best.pt")
OUT = SFT / "ronklm1_chat.pt"

PROMPT_TPL = "### Domanda:\n{q}\n\n### Risposta:\n"
STOP = "### Domanda:"


def cmd_train(args: argparse.Namespace) -> None:
    dev = "cuda"
    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    model = GPT(ck["config"], fast=True).to(dev)
    model.load_state_dict(ck["model"])
    print(f"[init] narratore da {args.init} (val {ck.get('val', float('nan')):.4f})")

    tokens = np.memmap(SFT / "sft_tokens.bin", dtype=np.uint16, mode="r")
    mask = np.memmap(SFT / "sft_mask.bin", dtype=np.uint8, mode="r")
    T = args.block_size
    n = len(tokens)
    steps = int(args.epochs * n / (args.batch * T))
    print(f"[dati] {n/1e6:.2f}M token | {steps} step ({args.epochs} epoche, batch {args.batch})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0, fused=True)
    rng = np.random.default_rng(0)

    def get_batch():
        ix = rng.integers(0, n - T - 1, size=args.batch)
        X = np.stack([tokens[i:i + T] for i in ix]).astype(np.int64)
        Y = np.stack([tokens[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
        M = np.stack([mask[i + 1:i + 1 + T] for i in ix]).astype(np.float32)
        return (torch.from_numpy(X).to(dev), torch.from_numpy(Y).to(dev),
                torch.from_numpy(M).to(dev))

    t0 = time.time()
    model.train()
    for step in range(steps):
        # warmup + cosine
        if step < args.warmup:
            lr = args.lr * (step + 1) / args.warmup
        else:
            r = (step - args.warmup) / max(1, steps - args.warmup)
            lr = args.lr / 10 + 0.5 * (1 + math.cos(math.pi * r)) * (args.lr - args.lr / 10)
        for g in opt.param_groups:
            g["lr"] = lr

        X, Y, M = get_batch()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.logits(X)                                  # (B,T,V)
            V = logits.shape[-1]
            per_tok = F.cross_entropy(logits.reshape(-1, V), Y.reshape(-1), reduction="none")
            loss = (per_tok * M.reshape(-1)).sum() / M.sum().clamp(min=1)  # loss SOLO sulle risposte
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            print(f"  step {step:4d}/{steps}  lr {lr:.2e}  loss {loss.item():.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    torch.save({"model": model.state_dict(), "config": model.config, "sft": True}, OUT)
    print(f"[fine] modello chat salvato in {OUT}  ({(time.time()-t0)/60:.1f} min)")


@torch.no_grad()
def cmd_chat(args: argparse.Namespace) -> None:
    dev = "cuda"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = GPT(ck["config"], fast=True).to(dev).eval()
    model.load_state_dict(ck["model"])
    tok = BPETokenizer.load(BPE)

    ids = tok.encode(PROMPT_TPL.format(q=args.prompt))
    ctx = torch.tensor(ids, dtype=torch.long, device=dev)[None, :]
    out_ids: list[int] = []
    for _ in range(args.n):
        w = ctx[:, -model.config.block_size:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.logits(w)
        lg = logits[0, -1].float() / max(args.temperature, 1e-6)
        if args.top_k:
            kth = torch.topk(lg, min(args.top_k, lg.numel())).values[-1]
            lg = torch.where(lg < kth, torch.full_like(lg, -float("inf")), lg)
        nxt = torch.multinomial(torch.softmax(lg, -1), 1)
        out_ids.append(int(nxt))
        ctx = torch.cat([ctx, nxt[None, :]], dim=1)
        # stop quando il modello comincia una nuova domanda
        if STOP in tok.decode(out_ids[-12:]):
            break
    answer = tok.decode(out_ids).split(STOP)[0].strip()
    print(f"\nD: {args.prompt}\nR: {answer}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SFT di RonkLM-1.")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--init", type=Path, default=NARRATOR)
    t.add_argument("--epochs", type=float, default=3.0)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--block-size", type=int, default=512)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--warmup", type=int, default=30)
    t.set_defaults(func=cmd_train)
    c = sub.add_parser("chat")
    c.add_argument("--ckpt", type=Path, default=OUT)
    c.add_argument("--prompt", required=True)
    c.add_argument("--n", type=int, default=200)
    c.add_argument("--temperature", type=float, default=0.7)
    c.add_argument("--top-k", type=int, default=40)
    c.set_defaults(func=cmd_chat)
    return p


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    a = build_parser().parse_args()
    a.func(a)
