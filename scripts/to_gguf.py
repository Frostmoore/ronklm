"""Converte RonkLM-1 in formato GGUF per llama.cpp / Ollama (Fase 14).

La nostra architettura E' un GPT-2 (token+pos embedding appresi, pre-norm, GELU-tanh,
MHA, testa lineare). Quindi la mappiamo sull'arch "gpt2" di llama.cpp, che Ollama
esegue nativamente. Tre punti delicati, gestiti qui:

1. ORIENTAMENTO PESI. GPT-2 in HuggingFace usa Conv1D (pesi (in,out)) e i convertitori
   li traspongono. Noi usiamo nn.Linear (pesi (out,in)) = gia' l'orientamento che
   llama.cpp si aspetta. Nessuna trasposizione.
2. BIAS. La nostra attention non ha bias su QKV -> scriviamo un bias di ZERI (no-op).
   La nostra testa HA un bias che GPT-2 non prevede -> lo scartiamo (approssimazione:
   verifichiamo che l'impatto sia piccolo).
3. TOKENIZER. Il nostro BPE e' byte-level; GPT-2 rappresenta i byte via la mappa
   bytes_to_unicode. Traduciamo i token e i merge in quella forma.

Uso:
    python scripts/to_gguf.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import gguf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ronklm.bpe import BPETokenizer  # noqa: E402
from ronklm_torch.model import GPT  # noqa: E402

CKPT = Path("D:/RonkLM_corpus/sft/ronklm1_chat.pt")
BPE = Path("D:/RonkLM_corpus/tokens/bpe_16384.pkl")
OUT = Path("D:/RonkLM_corpus/ronklm1.gguf")


def bytes_to_unicode() -> dict[int, str]:
    """La mappa GPT-2: ogni byte 0..255 -> un carattere unicode stampabile e univoco."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def main() -> None:
    b2u = bytes_to_unicode()

    def tokstr(bs: bytes) -> str:
        return "".join(b2u[x] for x in bs)

    # --- carica modello + tokenizer ---------------------------------------
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = GPT(cfg, fast=True)
    model.load_state_dict(ck["model"])
    sd = model.state_dict()
    tok = BPETokenizer.load(BPE)
    V = tok.vocab_size
    print(f"[modello] {cfg.n_layer} layer, {cfg.n_embd} embd, {cfg.n_head} teste, vocab {V}")

    # --- tokenizer in formato GPT-2 ---------------------------------------
    tokens = [tokstr(tok.vocab[i]) for i in range(V)]
    token_types = [gguf.TokenType.NORMAL] * V
    # i merge, nell'ordine di apprendimento: "pezzoA pezzoB"
    merges = []
    for (a, b), _new in sorted(tok.merges.items(), key=lambda kv: kv[1]):
        merges.append(tokstr(tok.vocab[a]) + " " + tokstr(tok.vocab[b]))
    print(f"[tokenizer] {len(tokens)} token, {len(merges)} merge")

    # --- scrittura GGUF ----------------------------------------------------
    w = gguf.GGUFWriter(str(OUT), "gpt2")
    w.add_name("RonkLM-1")
    w.add_context_length(cfg.block_size)
    w.add_embedding_length(cfg.n_embd)
    w.add_block_count(cfg.n_layer)
    w.add_feed_forward_length(4 * cfg.n_embd)
    w.add_head_count(cfg.n_head)
    w.add_layer_norm_eps(1e-5)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("gpt-2")
    w.add_token_list(tokens)
    w.add_token_types(token_types)
    w.add_token_merges(merges)
    w.add_bos_token_id(0)
    w.add_eos_token_id(0)
    w.add_add_bos_token(False)
    w.add_add_eos_token(False)

    def t(name: str) -> np.ndarray:
        return sd[name].to(torch.float32).numpy()

    # embedding e testa
    w.add_tensor("token_embd.weight", t("tok_emb.weight"))
    w.add_tensor("position_embd.weight", t("pos_emb.weight"))
    w.add_tensor("output_norm.weight", t("ln_f.weight"))
    w.add_tensor("output_norm.bias", t("ln_f.bias"))
    w.add_tensor("output.weight", t("head.weight"))
    head_bias_mag = float(np.abs(t("head.bias")).mean())
    print(f"[nota] head.bias scartato (GPT-2 non lo prevede). Magnitudine media: {head_bias_mag:.4f}")

    zero_qkv = np.zeros(3 * cfg.n_embd, dtype=np.float32)
    for i in range(cfg.n_layer):
        p = f"blocks.{i}."
        w.add_tensor(f"blk.{i}.attn_norm.weight", t(p + "ln1.weight"))
        w.add_tensor(f"blk.{i}.attn_norm.bias", t(p + "ln1.bias"))
        w.add_tensor(f"blk.{i}.attn_qkv.weight", t(p + "attn.qkv.weight"))
        w.add_tensor(f"blk.{i}.attn_qkv.bias", zero_qkv)          # la nostra QKV non ha bias
        w.add_tensor(f"blk.{i}.attn_output.weight", t(p + "attn.proj.weight"))
        w.add_tensor(f"blk.{i}.attn_output.bias", t(p + "attn.proj.bias"))
        w.add_tensor(f"blk.{i}.ffn_norm.weight", t(p + "ln2.weight"))
        w.add_tensor(f"blk.{i}.ffn_norm.bias", t(p + "ln2.bias"))
        w.add_tensor(f"blk.{i}.ffn_up.weight", t(p + "ffn.fc.weight"))
        w.add_tensor(f"blk.{i}.ffn_up.bias", t(p + "ffn.fc.bias"))
        w.add_tensor(f"blk.{i}.ffn_down.weight", t(p + "ffn.proj.weight"))
        w.add_tensor(f"blk.{i}.ffn_down.bias", t(p + "ffn.proj.bias"))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"[fine] GGUF scritto: {OUT}  ({OUT.stat().st_size/1024**2:.0f} MB)")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
