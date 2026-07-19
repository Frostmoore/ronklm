"""Training loop robusto (Fase 8).

Da "gira" a "gira bene e si misura": eval periodica su validation (media su piu'
batch), learning-rate schedule (warmup + cosine decay), salvataggio del MIGLIOR
modello su val, logging. E' il loop usato dalla CLI scripts/train_ronklm.py.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ronklm.dataset import Dataset
from ronklm.optim import AdamW
from ronklm.tokenizer import CharTokenizer


def cosine_lr(step: int, base_lr: float, warmup: int, max_steps: int, min_lr: float) -> float:
    """Learning rate con warmup lineare + decadimento a coseno.

    PERCHE' il warmup: nei primissimi passi le medie mobili di Adam sono stime
    immature (il secondo momento, al denominatore, e' sottostimato -> passi enormi su
    una rete fragile appena inizializzata). Il warmup tiene i passi piccoli finche' le
    stime non maturano. PERCHE' il decay: a fine training si e' vicini a un minimo;
    passi grandi orbitano attorno senza entrarci. Ridurre il lr permette di depositarsi.
    """
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup) / max(1, (max_steps - warmup))
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))  # 1 -> 0
    return min_lr + coeff * (base_lr - min_lr)


def evaluate(model, ds: Dataset, split: str, block_size: int, batch_size: int, n_batches: int, seed: int = 999) -> float:
    """NLL media su `n_batches` batch fissi (seed fisso -> stima stabile e confrontabile)."""
    r = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(n_batches):
        X, Y = ds.get_batch(split, block_size, batch_size, r)
        tot += model.loss(X, Y).data.item()
    return tot / n_batches


def train(
    model,
    ds: Dataset,
    tokenizer: CharTokenizer,
    *,
    steps: int,
    batch_size: int = 32,
    base_lr: float = 3e-3,
    min_lr: float = 3e-4,
    warmup: int = 100,
    weight_decay: float = 1e-4,
    eval_every: int = 250,
    eval_batches: int = 20,
    seed: int = 1,
    ckpt_path: str | Path | None = None,
    log_path: str | Path | None = None,
    grad_clip: float = 1.0,
) -> dict:
    """Addestra `model`, salvando il miglior checkpoint su validation. Ritorna un
    riepilogo (best_val, storia). Usa la config del modello per block_size."""
    block_size = model.config.block_size
    opt = AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    brng = np.random.default_rng(seed)
    best_val = float("inf")
    history: list[tuple[int, float, float]] = []
    log_lines = ["step,lr,train_batch,val_nll"]
    t0 = time.time()

    for step in range(steps):
        lr = cosine_lr(step, base_lr, warmup, steps, min_lr)
        opt.lr = lr
        X, Y = ds.get_batch("train", block_size, batch_size, brng)
        model.zero_grad()
        loss = model.loss(X, Y)
        loss.backward()
        _clip_gradients(model.parameters(), grad_clip)
        opt.step()

        if step % eval_every == 0 or step == steps - 1:
            val = evaluate(model, ds, "val", block_size, batch_size, eval_batches)
            history.append((step, loss.data.item(), val))
            log_lines.append(f"{step},{lr:.6f},{loss.data.item():.4f},{val:.4f}")
            tag = ""
            if val < best_val:
                best_val = val
                if ckpt_path is not None:
                    model.save(ckpt_path, tokenizer)
                    tag = "  <- best (salvato)"
            print(f"step {step:5d}  lr {lr:.5f}  train {loss.data.item():.4f}  val {val:.4f}  ({time.time()-t0:.0f}s){tag}", flush=True)

    if log_path is not None:
        Path(log_path).write_text("\n".join(log_lines), encoding="utf-8")
    return {"best_val": best_val, "history": history}


def _clip_gradients(params, max_norm: float) -> None:
    """Taglia la NORMA globale dei gradienti a `max_norm`.

    PERCHE': su tanti batch qualcuno produce un gradiente anomalo (documento strano,
    coincidenza numerica); un singolo passo gigante puo' buttare il modello in una
    zona da cui non si riprende (loss spike). Il clipping e' un'assicurazione a costo ~0.
    """
    total = math.sqrt(sum(float((p.grad ** 2).sum()) for p in params))
    if total > max_norm and total > 0:
        scale = max_norm / total
        for p in params:
            p.grad *= scale
