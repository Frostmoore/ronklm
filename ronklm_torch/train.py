"""Training loop su GPU per RonkLM-torch (Fase 12).

Riprende il loop della Fase 8 (warmup+cosine, eval mediata, best-checkpoint, grad
clipping) e ci aggiunge le tecniche che servono alla scala vera:

  - mixed precision bf16 (autocast): usa i tensor core, ~1,3x piu' veloce, meta' memoria;
  - gradient accumulation: batch EFFETTIVO grande a memoria costante;
  - dati via np.memmap: il corpus da GB non entra in RAM e non serve che ci entri;
  - GUARDIA ANTI-SPILLING: se la VRAM allocata si avvicina al limite o il throughput
    crolla, avvisa. Senza, un traboccamento silenzioso (vedi explain.md 9.5) farebbe
    girare 8 ore a un decimo della velocita' senza dare alcun errore.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class TrainConfig:
    train_bin: Path
    val_bin: Path
    out_dir: Path
    steps: int = 20_000
    micro_batch: int = 32          # batch FISICO (tetto misurato in Fase 9.3)
    grad_accum: int = 4            # batch effettivo = micro_batch * grad_accum
    block_size: int = 512
    base_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup: int = 400
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 500
    eval_batches: int = 40
    seed: int = 1337
    compile: bool = False          # torch.compile: piu' veloce ma lento a partire


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    """Warmup lineare + decadimento a coseno (stessa formula della Fase 8.3)."""
    if step < cfg.warmup:
        return cfg.base_lr * (step + 1) / cfg.warmup
    if step >= cfg.steps:
        return cfg.min_lr
    r = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (cfg.base_lr - cfg.min_lr)


class BinDataset:
    """Legge i token da un file binario uint16 via memmap.

    PERCHE' memmap: il file (GB) resta su disco e il sistema operativo porta in RAM solo
    le finestre effettivamente lette. Il training parte in un istante e usa memoria
    costante, qualunque sia la dimensione del corpus.
    """

    def __init__(self, path: Path, block_size: int, device: str) -> None:
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.device = device
        if len(self.data) <= block_size + 1:
            raise ValueError(f"{path} troppo corto: {len(self.data)} token")

    def __len__(self) -> int:
        return len(self.data)

    def batch(self, batch_size: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch casuale (X, Y) con Y = X spostato di 1 (il trucco della Fase 0.8)."""
        ix = rng.integers(0, len(self.data) - self.block_size - 1, size=batch_size)
        x = np.stack([self.data[i: i + self.block_size] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1: i + 1 + self.block_size] for i in ix]).astype(np.int64)
        xt = torch.from_numpy(x).to(self.device, non_blocking=True)
        yt = torch.from_numpy(y).to(self.device, non_blocking=True)
        return xt, yt


@torch.no_grad()
def evaluate(model, ds: BinDataset, cfg: TrainConfig, n_batches: int) -> float:
    """NLL media su batch fissi (seed fisso -> numero stabile e confrontabile)."""
    model.eval()
    rng = np.random.default_rng(1234)
    tot = 0.0
    for _ in range(n_batches):
        X, Y = ds.batch(cfg.micro_batch, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(X, Y)
        tot += loss.item()
    model.train()
    return tot / n_batches


def train(model, cfg: TrainConfig, device: str = "cuda") -> dict:
    """Addestra il modello. Ritorna un riepilogo con la miglior loss di validation."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_ds = BinDataset(cfg.train_bin, cfg.block_size, device)
    val_ds = BinDataset(cfg.val_bin, cfg.block_size, device)
    tokens_per_step = cfg.micro_batch * cfg.grad_accum * cfg.block_size
    print(f"[dati]  train {len(train_ds)/1e6:,.1f}M token | val {len(val_ds)/1e6:,.1f}M")
    print(f"[batch] fisico {cfg.micro_batch} x accum {cfg.grad_accum} x {cfg.block_size} "
          f"= {tokens_per_step:,} token/step effettivi")
    print(f"[piano] {cfg.steps:,} step -> {cfg.steps*tokens_per_step/1e9:.2f} mld di token "
          f"({cfg.steps*tokens_per_step/len(train_ds):.2f} epoche)")

    model = model.to(device)
    if cfg.compile:
        model = torch.compile(model)
    # niente weight decay su bias e LayerNorm: e' la prassi (regolarizzare un parametro
    # di scala o un bias non ha lo stesso senso che regolarizzare una matrice di pesi).
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.base_lr, betas=(0.9, 0.95), eps=1e-8, fused=(device == "cuda"),
    )

    rng = np.random.default_rng(cfg.seed)
    best_val = float("inf")
    t0 = time.time()
    log_path = cfg.out_dir / "training_log.csv"
    log_path.write_text("step,lr,train_loss,val_loss,tok_s,vram_gb\n", encoding="utf-8")
    warned_spill = False
    ref_tok_s = None

    for step in range(cfg.steps):
        lr = cosine_lr(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        t_step = time.time()
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(cfg.grad_accum):
            X, Y = train_ds.batch(cfg.micro_batch, rng)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(X, Y)
                loss = loss / cfg.grad_accum   # media, non somma, sui micro-batch
            loss.backward()
            loss_sum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t_step
        tok_s = tokens_per_step / dt
        if ref_tok_s is None and step == 10:
            ref_tok_s = tok_s

        # --- guardia anti-spilling (lezione di explain.md 9.5) ---
        if device == "cuda" and not warned_spill:
            vram = torch.cuda.max_memory_allocated() / 1024**3
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if vram > 0.90 * total_vram:
                print(f"  [ATTENZIONE] VRAM {vram:.1f}/{total_vram:.1f} GB: rischio di "
                      f"traboccamento in RAM. Riduci micro_batch!", flush=True)
                warned_spill = True
            elif ref_tok_s and step > 20 and tok_s < 0.4 * ref_tok_s:
                print(f"  [ATTENZIONE] throughput crollato ({tok_s:,.0f} vs {ref_tok_s:,.0f} "
                      f"tok/s): probabile spilling o altra app sulla GPU.", flush=True)
                warned_spill = True

        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            val = evaluate(model, val_ds, cfg, cfg.eval_batches)
            vram = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0
            tag = ""
            if val < best_val:
                best_val = val
                torch.save({"model": model.state_dict(), "config": model.config,
                            "step": step, "val": val}, cfg.out_dir / "best.pt")
                tag = "  <- best"
            el = (time.time() - t0) / 60
            print(f"step {step:6d}  lr {lr:.2e}  train {loss_sum:.4f}  val {val:.4f}  "
                  f"{tok_s:7,.0f} tok/s  {vram:4.1f}GB  {el:5.1f}min{tag}", flush=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{step},{lr:.6e},{loss_sum:.4f},{val:.4f},{tok_s:.0f},{vram:.2f}\n")

    return {"best_val": best_val, "minutes": (time.time() - t0) / 60}
