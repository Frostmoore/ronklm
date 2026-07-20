# codebase_reference.md — Atlante del codice di RonkLM

> **Cos'è questo documento**: l'atlante della codebase, non un riassunto. Criterio:
> un agente o una persona deve poter capire, trovare e modificare il codice **senza
> aprire i file**. Se per sapere la firma di un metodo bisogna leggere il sorgente,
> questo documento ha fallito.
>
> **Stato**: aggiornato durante la **Fase 12** (2026-07-20). Percorso A completo; port
> PyTorch con equivalenza dimostrata (F9); BPE scritto a mano (F10); corpus Wikipedia IT
> da 1,13 mld di token (F11); training di RonkLM-50M in corso. Milestone M1–M7 completate. Piano in [`plan_ronklm_system.md`](plan_ronklm_system.md).
>
> **Verifica meccanica firme**: eseguita a fine Fase 0 con estrazione `def`/`class`
> via grep e confronto con le tabelle qui sotto. ✅ Allineato.

---

## 1. Indice "dove sta cosa" (la porta d'ingresso)

| Cerchi… | Vai in… |
|---|---|
| Scaricare/pulire il corpus | [`data/prepare_corpus.py`](../data/prepare_corpus.py) |
| Il corpus pulito pronto all'uso | `data/input.txt` (generato, committato) |
| Convertire testo ⇄ interi | `CharTokenizer` in [`ronklm/tokenizer.py`](../ronklm/tokenizer.py) |
| Split train/val e batch (X, Y) | `Dataset` in [`ronklm/dataset.py`](../ronklm/dataset.py) |
| Leggere un file di corpus | `load_text()` in [`ronklm/dataset.py`](../ronklm/dataset.py) |
| Bigram per conteggio (Fase 1) | `BigramCount` in [`ronklm/models/bigram_count.py`](../ronklm/models/bigram_count.py) |
| Bigram neurale (Fase 2) | `BigramNeural` in [`ronklm/models/bigram_neural.py`](../ronklm/models/bigram_neural.py) |
| Autograd (`Tensor`, backward) | [`ronklm/autograd.py`](../ronklm/autograd.py) — `Tensor`, `cross_entropy` (Fase 3) |
| Layer riusabili (Module/Linear/Embedding) | [`ronklm/nn.py`](../ronklm/nn.py) (Fase 4) |
| Ottimizzatori (SGD/AdamW) | [`ronklm/optim.py`](../ronklm/optim.py) (Fase 4) |
| MLP a contesto (Fase 4) | `MLP` in [`ronklm/models/mlp.py`](../ronklm/models/mlp.py) |
| Self-attention causale (Fase 5) | `Head`, `AttentionLM` in [`ronklm/models/attention.py`](../ronklm/models/attention.py) |
| Blocco transformer (Fase 6) | `Block`, `MultiHeadAttention`, `FeedForward` in [`ronklm/models/block.py`](../ronklm/models/block.py) |
| LayerNorm (Fase 6) | `LayerNorm` in [`ronklm/nn.py`](../ronklm/nn.py) · `cat` in [`ronklm/autograd.py`](../ronklm/autograd.py) |
| GPT completo (Fase 7) | `GPT`, `GPTConfig` in [`ronklm/models/gpt.py`](../ronklm/models/gpt.py) |
| Generazione (temperature/top-k) | `generate()` in [`ronklm/generate.py`](../ronklm/generate.py) |
| Training loop + schedule (Fase 8) | `train()`, `cosine_lr()`, `evaluate()` in [`ronklm/train.py`](../ronklm/train.py) |
| CLI addestra/genera (Fase 8) | [`scripts/train_ronklm.py`](../scripts/train_ronklm.py) |
| **Port PyTorch (Fase 9)** | [`ronklm_torch/model.py`](../ronklm_torch/model.py) — `GPT`, `load_weights_from_numpy` |
| Prova di equivalenza NumPy↔torch | [`tests/test_equivalence.py`](../tests/test_equivalence.py) |
| Benchmark motori (CPU/GPU) | [`scripts/benchmark.py`](../scripts/benchmark.py) |
| **Tokenizer BPE (Fase 10)** | `BPETokenizer` in [`ronklm/bpe.py`](../ronklm/bpe.py) |
| Scaricare Wikipedia IT (Kiwix) | [`data/corpus_b/download_wikipedia.py`](../data/corpus_b/download_wikipedia.py) |
| Estrarre/pulire il corpus da ZIM | [`data/corpus_b/extract_wikipedia.py`](../data/corpus_b/extract_wikipedia.py) |
| Addestrare BPE + tokenizzare in binario | [`data/corpus_b/tokenize_corpus.py`](../data/corpus_b/tokenize_corpus.py) |
| Training su GPU (Fase 12) | [`ronklm_torch/train.py`](../ronklm_torch/train.py) — `train()`, `BinDataset` |
| CLI training/generazione GPU | [`scripts/train_torch.py`](../scripts/train_torch.py) |
| Eseguire tutti i test | `python run_tests.py` (radice) |
| Runner di test senza pytest | [`tests/_runner.py`](../tests/_runner.py) |
| Versione del pacchetto | `__version__` in [`ronklm/__init__.py`](../ronklm/__init__.py) |
| Spiegazione didattica profonda | [`explain.md`](../explain.md) (radice) — il "libro di testo" |

---

## 2. Albero dei file (solo codice nostro)

```
RonkLM/
├── data/
│   ├── prepare_corpus.py      # [A] download + pulizia -> data/input.txt
│   ├── input.txt              # [GENERATO, committato] corpus Pinocchio, 240.920 char
│   ├── pinocchio_raw.txt      # [GENERATO, gitignored] cache download grezzo
│   └── corpus_b/              # [B] pipeline del corpus grande (Fase 11)
│       ├── download_wikipedia.py   # ZIM da Kiwix, con mirror e ripresa
│       ├── extract_wikipedia.py    # ZIM -> testo pulito (filtri, dedup, parallelo)
│       └── tokenize_corpus.py      # BPE + tokenizzazione in uint16 (streaming)
├── ronklm/
│   ├── __init__.py            # docstring pacchetto + __version__
│   ├── tokenizer.py           # CharTokenizer
│   ├── bpe.py                 # BPETokenizer byte-level (Fase 10)
│   ├── dataset.py             # load_text(), Dataset
│   ├── autograd.py            # ronkgrad: Tensor + cross_entropy + cat (Fase 3/6)
│   ├── nn.py                  # Module, Linear, Embedding, LayerNorm (Fase 4/6)
│   ├── optim.py               # SGD, AdamW (Fase 4)
│   ├── generate.py            # generate() con temperature/top-k (Fase 7)
│   ├── train.py               # train(), cosine_lr(), evaluate() (Fase 8)
│   └── models/
│       ├── __init__.py
│       ├── bigram_count.py    # BigramCount (Fase 1)
│       ├── bigram_neural.py   # BigramNeural (Fase 2)
│       ├── mlp.py             # MLP (Fase 4)
│       ├── attention.py       # Head, AttentionLM (Fase 5)
│       ├── block.py           # Block, MultiHeadAttention, FeedForward (Fase 6)
│       └── gpt.py             # GPT, GPTConfig (Fase 7)
├── ronklm_torch/              # [PERCORSO B] port PyTorch (Fase 9+)
│   ├── __init__.py            # tabella di corrispondenza ronkgrad <-> PyTorch
│   ├── model.py               # GPT torch, CausalSelfAttention, load_weights_from_numpy
│   └── train.py               # training GPU: bf16, grad accum, memmap, anti-spilling
├── scripts/
│   ├── train_ronklm.py        # [A] CLI: train / generate (Fase 8)
│   ├── benchmark.py           # benchmark NumPy/torch, CPU/GPU (Fase 9.3)
│   └── train_torch.py         # [B] CLI training/generazione su GPU (Fase 12)
├── tests/
│   ├── _runner.py             # run(namespace) -> n_fallimenti
│   ├── _gradcheck.py          # grad_check condiviso (autograd + block)
│   ├── test_tokenizer.py      # 8 test
│   ├── test_dataset.py        # 8 test
│   ├── test_bigram_count.py   # 6 test
│   ├── test_bigram_neural.py  # 5 test
│   ├── test_autograd.py       # 24 gradient check
│   ├── test_mlp.py            # 5 test
│   ├── test_attention.py      # 4 test
│   ├── test_block.py          # 6 test
│   ├── test_gpt.py            # 8 test
│   ├── test_bpe.py            # 9 test
│   ├── test_train.py          # 4 test
│   └── test_equivalence.py    # 7 test (NumPy <-> PyTorch)
├── run_tests.py               # lancia tutti i tests/test_*.py
├── explain.md                 # libro di testo: spiegazione didattica per fase
├── requirements.txt           # numpy (+ matplotlib opz., torch dalla Fase 9)
├── README.md
├── .gitattributes             # forza LF ovunque (no \r nel vocab)
└── .gitignore
```

**NON esiste ancora** (per evitare ricerche a vuoto): nessun **weight tying** nel GPT
torch (valutato per il run da 150M, non implementato); nessun **SFT/chatbot** (Percorso
C, Fase 13); nessun corpus Gutenberg (previsto ma non ancora estratto: per il pilota si
usa la sola Wikipedia).

**Esiste**: tutto il Percorso A (Fasi 0–8, NumPy); il port PyTorch con equivalenza
dimostrata e benchmark (F9, due varianti di attention); il **BPE byte-level** (F10); la
**pipeline del corpus** Wikipedia IT → 1,13 mld di token (F11); il **training su GPU**
con CLI (F12).

> **Dati fuori dal repo**: il corpus del Percorso B vive in `D:/RonkLM_corpus/`
> (NVMe, non versionato): `wikipedia_it_all_nopic_2026-05.zim` (8,29 GB), `text/`
> (28 shard, 4,14 GB), `tokens/` (`train.bin` 2,25 GB, `val.bin`, `bpe_16384.pkl`),
> `run50m/` (checkpoint + `training_log.csv`).

---

## 3. Moduli, classi e funzioni (firme complete)

### 3.1 `ronklm/tokenizer.py` — classe `CharTokenizer`

Tokenizer a livello di carattere: mappa bidirezionale carattere ⇄ indice intero.
Vocabolario = caratteri unici del corpus, **ordinati** (determinismo → indici
stabili → modelli salvati ricaricabili).

Attributi d'istanza:

| Attributo | Tipo | Significato |
|---|---|---|
| `chars` | `list[str]` | vocabolario, lista ordinata di caratteri unici |
| `stoi` | `dict[str, int]` | string-to-int: carattere → indice |
| `itos` | `dict[int, str]` | int-to-string: indice → carattere |

Metodi:

| Metodo | Firma | Effetto |
|---|---|---|
| `__init__` | `(self, chars: list[str]) -> None` | costruisce `chars=sorted(set(chars))`, `stoi`, `itos` |
| `from_text` | `(cls, text: str) -> CharTokenizer` | *classmethod*: vocab dai caratteri unici di `text` |
| `vocab_size` | `(self) -> int` *(property)* | numero di token distinti (`len(chars)`) |
| `encode` | `(self, text: str) -> list[int]` | testo → indici; **`ValueError`** su carattere ignoto |
| `decode` | `(self, indices: list[int]) -> str` | indici → testo; **`ValueError`** su indice fuori range |
| `save` | `(self, path: str \| Path) -> None` | salva `{"chars": [...]}` come JSON UTF-8 |
| `load` | `(cls, path: str \| Path) -> CharTokenizer` | *classmethod*: ricostruisce da file `save()` |
| `__repr__` | `(self) -> str` | `CharTokenizer(vocab_size=N)` |

### 3.2 `ronklm/dataset.py`

Funzione a livello di modulo:

| Funzione | Firma | Effetto |
|---|---|---|
| `load_text` | `(path: str \| Path) -> str` | legge UTF-8 con `newline=''` (non traduce i fine-riga) |

Classe `Dataset` — corpus codificato in interi e suddiviso train/val, con batching.

Attributi d'istanza:

| Attributo | Tipo | Significato |
|---|---|---|
| `tokenizer` | `CharTokenizer` | la mappa usata per codificare |
| `data` | `np.ndarray[int64]` | intero corpus codificato |
| `train` | `np.ndarray[int64]` | primi `(1-val_frac)` del corpus (contiguo) |
| `val` | `np.ndarray[int64]` | ultimo `val_frac` del corpus (contiguo) |

Metodi:

| Metodo | Firma | Effetto |
|---|---|---|
| `__init__` | `(self, text: str, tokenizer: CharTokenizer, val_frac: float = 0.1) -> None` | codifica il testo, split contiguo train/val |
| `from_file` | `(cls, path, tokenizer: CharTokenizer \| None = None, val_frac: float = 0.1) -> Dataset` | *classmethod*: carica da file; se `tokenizer=None` lo costruisce dal corpus |
| `get_batch` | `(self, split: str, block_size: int, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]` | batch casuale `(X, Y)`, shape `(batch_size, block_size)`, `int64`; `Y` = `X` shiftato di 1; **`ValueError`** se split ignoto o troppo corto |
| `_split` | `(self, split: str) -> np.ndarray` | ritorna `train`/`val`; `ValueError` altrimenti |
| `__repr__` | `(self) -> str` | `Dataset(total=…, train=…, val=…, vocab=…)` |

**Contratto di `get_batch`** (invarianti garantite):
- `X.shape == Y.shape == (batch_size, block_size)`, dtype `int64`.
- `Y[b, t]` è il carattere che segue `X[b, t]` nel corpus → `X[:, 1:] == Y[:, :-1]`.
- indici di partenza in `[0, len(data)-block_size)` (mai fuori bordo).
- deterministico rispetto a `rng`: stesso seed → stesso batch.

### 3.3 `ronklm/models/bigram_count.py` — classe `BigramCount`

Language model a bigrammi basato su conteggi (Fase 1). Predice il prossimo carattere
dal solo precedente; impara contando, senza gradienti.

Attributi d'istanza:

| Attributo | Tipo | Significato |
|---|---|---|
| `vocab_size` | `int` | numero di token distinti |
| `N` | `np.ndarray[int64]` shape `(V, V)` | conteggi: `N[i, j]` = quante volte a `i` segue `j` |
| `P` | `np.ndarray[float64]` shape `(V, V)` \| `None` | probabilità per riga (dopo `fit`) |
| `smoothing` | `float` | conteggio fittizio aggiunto (Laplace) |

Metodi:

| Metodo | Firma | Effetto |
|---|---|---|
| `__init__` | `(self, vocab_size: int) -> None` | alloca `N` a zeri, `P=None` |
| `fit` | `(self, data: np.ndarray, smoothing: float = 1.0) -> BigramCount` | conta i bigrammi (`np.add.at`) e calcola `P` normalizzando per riga con smoothing |
| `nll` | `(self, data: np.ndarray) -> float` | NLL media in nats sulle coppie di `data`; `RuntimeError` se non fittato |
| `uniform_nll` | `(vocab_size: int) -> float` *(staticmethod)* | `log(vocab_size)`: la NLL del modello uniforme |
| `generate` | `(self, rng: np.random.Generator, n: int, start: int = 0) -> list[int]` | campiona `n` indici autoregressivamente da `start` |
| `__repr__` | `(self) -> str` | `BigramCount(vocab_size=…, fitted/non-fitted)` |

**Numeri di riferimento** (corpus Pinocchio, smoothing=1): NLL uniforme **4.2341**,
train **2.3340**, val **2.3455** nats; perplexity val **10.44**.

### 3.4 `ronklm/models/bigram_neural.py` — classe `BigramNeural`

Bigram appreso per discesa del gradiente (Fase 2): regressione softmax a un layer,
`logits = onehot(x) @ W`. Gradiente derivato a mano (`probs − onehot(y)`).

Attributi: `vocab_size: int`; `W: np.ndarray[float64]` shape `(V, V)` (pesi, init
gaussiana piccola).

Metodi:

| Metodo | Firma | Effetto |
|---|---|---|
| `__init__` | `(self, vocab_size: int, rng: np.random.Generator, init_std: float = 0.01) -> None` | `W` gaussiana piccola |
| `_softmax` | `(logits: np.ndarray) -> np.ndarray` *(staticmethod)* | softmax stabile (`- max`) sull'ultimo asse |
| `forward` | `(self, x_idx: np.ndarray) -> np.ndarray` | indici `(B,)` → probabilità `(B, V)` |
| `loss_and_grad` | `(self, x_idx, y_idx) -> tuple[float, np.ndarray]` | cross-entropy media e `dW` analitico |
| `loss` | `(self, x_idx, y_idx) -> float` | solo la loss (per il gradient check) |
| `train` | `(self, data, steps, lr, rng=None, batch_size=None, log_every=0) -> list[float]` | GD full-batch o minibatch stocastico; ritorna storia loss |
| `train_from_counts` | `(self, N, steps, lr, log_every=0) -> list[float]` | GD full-batch **esatta** dai conteggi (O(V²)/passo); ritorna storia loss |
| `nll` | `(self, data: np.ndarray) -> float` | NLL media in nats sulle coppie |
| `probabilities` | `(self) -> np.ndarray` | `softmax(W)`: la `(V, V)` appresa, da confrontare con la `P` contata |
| `generate` | `(self, rng, n, start=0) -> list[int]` | campionamento autoregressivo dalle prob. apprese |
| `__repr__` | `(self) -> str` | `BigramNeural(vocab_size=…)` |

**Numeri**: loss iniziale **4.2339** ≈ log 69 (sanity init); dopo 500 step full-batch
NLL train/val **2.373/2.385** (→ converge al bigram a conteggio); gradient check
errore relativo < 1e-4.

### 3.5 `ronklm/autograd.py` — ronkgrad (motore di autograd)

Mini-PyTorch tensoriale (Fase 3). Costruisce un grafo computazionale nel forward e ne
percorre l'inverso nel backward. **Da qui in poi ogni modello si fida di questo file.**

Funzione di modulo `_unbroadcast(grad, shape) -> np.ndarray`: riporta un gradiente
alla forma originale sommando lungo le dimensioni broadcastate (il punto più insidioso).

Classe `Tensor`. Attributi: `data: np.ndarray[float64]`, `grad: np.ndarray[float64]`,
`_backward: callable`, `_prev: tuple[Tensor]`, `_op: str`; proprietà `shape`.

| Metodo/operatore | Firma | Effetto (e formula del backward) |
|---|---|---|
| `__init__` | `(self, data, _prev=(), _op="") -> None` | wrappa un array, `grad` a zeri |
| `zero_grad` | `(self) -> None` | azzera `grad` |
| `__add__`/`__radd__` | `(self, other) -> Tensor` | somma; backward: distribuisce `out.grad` a entrambi |
| `__neg__`/`__sub__`/`__rsub__` | `(self, other) -> Tensor` | negazione/sottrazione (via add+mul) |
| `__mul__`/`__rmul__` | `(self, other) -> Tensor` | prodotto elem.; backward: `other*g` e `self*g` |
| `__pow__` | `(self, p: float) -> Tensor` | potenza scalare; backward: `p*x^(p-1)*g` |
| `__truediv__`/`__rtruediv__` | `(self, other) -> Tensor` | divisione (via `pow(-1)`) |
| `__matmul__` | `(self, other) -> Tensor` | matmul (anche batch); backward: `g@Bᵀ`, `Aᵀ@g` |
| `sum` | `(self, axis=None, keepdims=False) -> Tensor` | riduzione; backward: broadcast di `g` |
| `mean` | `(self, axis=None, keepdims=False) -> Tensor` | media (= sum/n) |
| `relu` | `(self) -> Tensor` | `max(0,x)`; backward: `(x>0)*g` |
| `tanh` | `(self) -> Tensor` | backward: `(1-tanh²)*g` |
| `exp` | `(self) -> Tensor` | backward: `exp(x)*g` |
| `log` | `(self) -> Tensor` | backward: `(1/x)*g` |
| `gelu` | `(self) -> Tensor` | GELU-tanh **composita** (backward automatico) |
| `softmax` | `(self, axis=-1) -> Tensor` | backward: `s*(g - Σ(g*s))` |
| `backward` | `(self) -> None` | topo-sort + `grad=1` alla radice + `_backward` in ordine inverso |

Funzione di modulo:

| Funzione | Firma | Effetto |
|---|---|---|
| `cross_entropy` | `(logits: Tensor, targets: np.ndarray) -> Tensor` | CE media fusa/stabile; backward: `(softmax − onehot)/B` |

Operazioni aggiunte in Fase 4 (per embedding, attention, layernorm):

| Metodo | Firma | Effetto (backward) |
|---|---|---|
| `reshape` | `(self, *shape) -> Tensor` | cambia forma; backward: reshape inverso |
| `transpose` | `(self, axis1, axis2) -> Tensor` | scambia due assi; backward: ri-scambia |
| `gather_rows` | `(self, idx: np.ndarray) -> Tensor` | seleziona righe (embedding); backward: scatter-add |
| `masked_fill` | `(self, mask: np.ndarray, value: float) -> Tensor` | mette `value` dove `mask`; backward: 0 sulle celle mascherate |
| `var` | `(self, axis, keepdims=True) -> Tensor` | varianza (composita mean+pow) per LayerNorm |

**Validazione**: 23 gradient check < 1e-5; riproduce il gradiente manuale della Fase 2
a **2e-17** (precisione macchina).

### 3.6 `ronklm/nn.py` — layer riusabili

| Classe/metodo | Firma | Effetto |
|---|---|---|
| `Module.parameters` | `(self) -> list[Tensor]` | raccoglie ricorsivamente i Tensor-parametro (da `__dict__`) |
| `Module.zero_grad` | `(self) -> None` | azzera i gradienti dei parametri |
| `Module.__call__` | `(self, *a, **k)` | invoca `forward` |
| `Linear.__init__` | `(self, n_in, n_out, rng, bias=True)` | `W` init `1/√n_in`, `b` a zeri |
| `Linear.forward` | `(self, x: Tensor) -> Tensor` | `x @ W (+ b)` |
| `Embedding.__init__` | `(self, num, dim, rng, std=1.0)` | tabella `weight` `(num, dim)` |
| `Embedding.forward` | `(self, idx: np.ndarray) -> Tensor` | `weight.gather_rows(idx)` |

### 3.7 `ronklm/optim.py` — ottimizzatori

| Classe/metodo | Firma | Effetto |
|---|---|---|
| `SGD.__init__` | `(self, params, lr)` | — |
| `SGD.step` | `(self) -> None` | `p.data -= lr * p.grad` |
| `AdamW.__init__` | `(self, params, lr=3e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0)` | inizializza momenti `m`, `v` |
| `AdamW.step` | `(self) -> None` | momento + scaling adattivo + bias-correction + weight decay disaccoppiato |
| `*.zero_grad` | `(self) -> None` | azzera i gradienti |

### 3.8 `ronklm/models/mlp.py` — classe `MLP`

MLP a contesto fisso (Fase 4): `Embedding → concat → Linear+tanh → Linear`.

| Metodo | Firma | Effetto |
|---|---|---|
| `__init__` | `(self, vocab_size, block_size, n_embd, n_hidden, rng)` | crea `emb`, `h`, `head` |
| `logits` | `(self, x_idx: np.ndarray) -> Tensor` | contesto `(B,T)` → logits `(B, vocab)` |
| `loss` | `(self, x_idx, y_idx) -> Tensor` | cross-entropy (Tensor, per backward) |
| `targets_from_batch` | `(Y: np.ndarray) -> np.ndarray` *(staticmethod)* | `Y[:, -1]` (char dopo il contesto) |
| `nll` | `(self, data, block_size, rng, n_batches=20, batch_size=256) -> float` | NLL media su più batch |
| `generate` | `(self, rng, n, tokenizer=None, seed_ctx=None) -> list[int]` | generazione con contesto scorrevole |

**Numeri**: NLL val **1.897** (< bigram 2.346), train **1.81** (overfitting gap).

### 3.9 `ronklm/models/attention.py` — self-attention (Fase 5)

Classe `Head` (una testa causale). Attributi: `head_size`; `key`,`query`,`value`
(`Linear` senza bias); `mask` (bool `(block,block)`, True sopra diagonale); `scale`
(`1/√head_size`); `last_att` (np.ndarray, l'ultima matrice di attenzione per l'ispezione).

| Metodo | Firma | Effetto |
|---|---|---|
| `__init__` | `(self, n_embd, head_size, block_size, rng)` | crea le 3 proiezioni + maschera |
| `forward` | `(self, x: Tensor) -> Tensor` | `(B,T,C)` → `(B,T,head_size)`: `softmax(mask(q·kᵀ/√H)) · v`, salva `last_att` |

Classe `AttentionLM` (mini-LM di prova). Attributi: `tok` (Embedding), `head` (Head),
`lm_head` (Linear).

| Metodo | Firma | Effetto |
|---|---|---|
| `__init__` | `(self, vocab_size, block_size, n_embd, rng)` | — |
| `logits` | `(self, x_idx) -> Tensor` | `(B,T)` → `(B,T,vocab)` |
| `loss` | `(self, x_idx, y_idx) -> Tensor` | CE su tutte le `B·T` posizioni |

**Numeri**: NLL val **2.328** (≈ bigram — una testa senza positional embedding è cieca
all'ordine; la potenza arriva in F6/F7). Attention verificata causale e normalizzata.

### 3.10 `ronklm/models/block.py` — blocco Transformer (Fase 6)

| Classe | Firma `__init__` / `forward` | Effetto |
|---|---|---|
| `MultiHeadAttention` | `(n_embd, n_head, block_size, rng)` / `(x)->Tensor` | `n_head` `Head` in parallelo → `cat` → `proj`; shape invariata |
| `FeedForward` | `(n_embd, rng)` / `(x)->Tensor` | `Linear(→4·n_embd) → gelu → Linear(→n_embd)` |
| `Block` | `(n_embd, n_head, block_size, rng)` / `(x)->Tensor` | `x = x + attn(ln1(x)); x = x + ffn(ln2(x))` (pre-norm + residual) |

`nn.LayerNorm(dim, eps=1e-5)`: normalizza l'ultima dim a media 0/var 1, poi
`* gamma + beta` (parametri appresi). Tutto composito → backward automatico.

`autograd.cat(tensors, axis=-1) -> Tensor`: concatena; backward ri-spezza il gradiente.

**Verificato**: LayerNorm normalizza e supera il gradient check; Block preserva la
shape `(B,T,C)` ed è impilabile; ogni parametro riceve gradiente.

### 3.11 `ronklm/models/gpt.py` — il GPT completo (Fase 7)

`@dataclass GPTConfig`: `vocab_size`, `block_size=32`, `n_layer=4`, `n_head=4`,
`n_embd=64`. Salvata dentro il checkpoint.

Classe `GPT(Module)`. Attributi: `config`, `tok_emb`/`pos_emb` (Embedding),
`blocks` (list[Block]), `ln_f` (LayerNorm), `head` (Linear).

| Metodo | Firma | Effetto |
|---|---|---|
| `__init__` | `(self, config: GPTConfig, rng)` | costruisce embedding, N blocchi, ln finale, testa |
| `logits` | `(self, idx: np.ndarray) -> Tensor` | `(B,T)` → `(B,T,vocab)`; `tok+pos → blocchi → ln_f → head` |
| `loss` | `(self, idx, targets) -> Tensor` | CE su tutte le `B·T` posizioni |
| `save` | `(self, path, tokenizer) -> None` | `.npz` con pesi + config + `chars` (autosufficiente) |
| `load` | `(path, rng) -> (GPT, CharTokenizer)` *(classmethod)* | ricostruisce modello e tokenizer |
| `num_params` | `(self) -> int` | numero totale di parametri |

### 3.12 `ronklm/generate.py`

| Funzione | Firma | Effetto |
|---|---|---|
| `generate` | `(model, context: list[int], max_new_tokens, rng, temperature=1.0, top_k=None) -> list[int]` | campionamento autoregressivo; tronca a `block_size`; applica temperature e top-k |

### 3.13 `ronklm/train.py` (Fase 8)

| Funzione | Firma | Effetto |
|---|---|---|
| `cosine_lr` | `(step, base_lr, warmup, max_steps, min_lr) -> float` | warmup lineare + decadimento coseno |
| `evaluate` | `(model, ds, split, block_size, batch_size, n_batches, seed=999) -> float` | NLL media su batch fissi |
| `train` | `(model, ds, tokenizer, *, steps, batch_size=32, base_lr=3e-3, min_lr=3e-4, warmup=100, weight_decay=1e-4, eval_every=250, eval_batches=20, seed=1, ckpt_path=None, log_path=None, grad_clip=1.0) -> dict` | loop con schedule, eval, best-checkpoint, grad clipping, log CSV |
| `_clip_gradients` | `(params, max_norm) -> None` | taglia la norma globale dei gradienti |

### 3.14 `scripts/train_ronklm.py` — CLI (Fase 8)

Sottocomandi `train` (data, steps, batch-size, block-size, n-layer, n-head, n-embd,
lr, warmup, weight-decay, eval-every, seed, out, log) e `generate` (ckpt, prompt, n,
temperature, top-k, seed).

### 3.15 `ronklm_torch/model.py` — port PyTorch (Fase 9)

Pacchetto **separato** da `ronklm/`: i due motori convivono, il NumPy resta il
riferimento. Architettura identica, stessi dettagli numerici (`MASK_VALUE = -1e9`,
gelu `approximate='tanh'`, LayerNorm `eps=1e-5`).

| Classe/funzione | Firma | Note |
|---|---|---|
| `GPTConfig` | `@dataclass(vocab_size, block_size=32, n_layer=4, n_head=4, n_embd=64)` | stessi campi della versione NumPy |
| `Head` | `(n_embd, head_size, block_size)` / `forward(x)` | maschera causale come `register_buffer` |
| `MultiHeadAttention` | `(n_embd, n_head, block_size)` / `forward(x)` | versione **didattica**: `nn.ModuleList` + `torch.cat` + `proj` |
| `CausalSelfAttention` | `(n_embd, n_head, block_size)` / `forward(x)` | versione **ottimizzata**: QKV fuso + `F.scaled_dot_product_attention` (FlashAttention). Stessa matematica, 2,6× più veloce e metà VRAM |
| `FeedForward` | `(n_embd)` / `forward(x)` | `F.gelu(..., approximate='tanh')` |
| `Block` | `(n_embd, n_head, block_size, fast=False)` / `forward(x)` | pre-norm + residual; `fast` sceglie l'attention ottimizzata |
| `GPT` | `(config, fast=False)` / `logits(idx)`, `forward(idx, targets=None) -> (logits, loss)`, `num_params()` | — |
| `load_weights_from_numpy` | `(torch_model: GPT, np_model) -> None` | copia i pesi; **traspone i Linear**; gestisce sia il layout per-testa sia il **QKV fuso** |
| `config_from_numpy` | `(np_config) -> GPTConfig` | — |

> ⚠️ **Trappola disinnescata (la più importante della fase)**: il nostro `Linear`
> tiene `W` di forma `(n_in, n_out)` e calcola `x @ W + b`; `torch.nn.Linear` tiene
> `weight` di forma `(n_out, n_in)` e calcola `x @ weight.T + b`. Il trasferimento dei
> pesi **richiede la trasposizione**, e lo stesso vale confrontando i gradienti. È
> esattamente il tipo di errore che "non crasha, degrada soltanto".

### 3.16 `scripts/benchmark.py` — benchmark dei motori (Fase 9.3)

`bench_numpy(cfg, batch_size, steps) -> float` · `bench_torch(cfg, batch_size, steps,
device, amp=False) -> float` → token/secondo di training. CLI con `--n-layer`,
`--n-embd`, `--block-size`, `--batch-size`, `--steps`, `--skip-numpy`.

### 3.17 `ronklm/bpe.py` — tokenizer BPE (Fase 10)

BPE **byte-level**: parte dai 256 byte, quindi nessun testo è mai fuori vocabolario.

| Metodo/funzione | Firma | Effetto |
|---|---|---|
| `_split_words` | `(text: str) -> list[str]` | pre-tokenizzazione stile GPT-2 (lo spazio resta attaccato alla parola seguente) |
| `_merge_word` | `(word: tuple, pair: tuple, new_id: int) -> tuple` | sostituisce una coppia con il nuovo simbolo |
| `BPETokenizer.train` | `(self, text: str, vocab_size: int, verbose=False) -> BPETokenizer` | impara le fusioni; **implementazione incrementale** (indice coppia→parole + max-heap con cancellazione pigra) |
| `BPETokenizer.encode` | `(self, text: str) -> list[int]` | testo → id; non fallisce mai |
| `BPETokenizer.decode` | `(self, ids: list[int]) -> str` | id → testo |
| `BPETokenizer.vocab_size` | `(self) -> int` *(property)* | — |
| `BPETokenizer.save` / `.load` | `(path)` | pickle (le chiavi sono tuple) + `.preview.json` leggibile |

**Numeri**: vocab 16.384 addestrato su 30 MB in **12 s**; compressione ~**3,6–3,8
caratteri/token** sull'italiano.

### 3.18 `data/corpus_b/` — pipeline del corpus grande (Fase 11)

| Script | Funzioni principali | Effetto |
|---|---|---|
| `download_wikipedia.py` | `download(url, dest)`, `MIRRORS`, `VARIANTS` | scarica lo ZIM con ripresa (header `Range`); default mirror **dotsrc** (47 MB/s vs 3,8 dell'origine) |
| `extract_wikipedia.py` | `html_to_text`, `keep_article`, `_collapse_repeated_lines`, `extract_parallel` | ZIM → testo pulito; filtri di qualità; dedup per hash; multiprocessing a priorità bassa |
| `tokenize_corpus.py` | `default_workers`, `_encode_chunk`, `main` | addestra il BPE su un campione, tokenizza in parallelo, scrive `train.bin`/`val.bin` in streaming |

**Output** (su `D:/RonkLM_corpus/`, fuori dal repo perché su NVMe): 1.099.087 articoli,
4,14 GB di testo → **1.128,4M token** (train 1.122,8M / val 5,6M), `uint16`, 2,25 GB.

### 3.19 `ronklm_torch/train.py` — training su GPU (Fase 12)

| Elemento | Firma | Effetto |
|---|---|---|
| `TrainConfig` | dataclass: `steps, micro_batch=32, grad_accum=4, block_size=512, base_lr, min_lr, warmup, weight_decay, grad_clip, eval_every, eval_batches, seed, compile` | — |
| `cosine_lr` | `(step, cfg) -> float` | warmup + cosine decay |
| `BinDataset` | `(path, block_size, device)`, `.batch(batch_size, rng)` | legge i token via `np.memmap`; `Y` = `X` shiftato di 1 |
| `evaluate` | `(model, ds, cfg, n_batches) -> float` | NLL media su batch fissi |
| `train` | `(model, cfg, device='cuda') -> dict` | loop con bf16 autocast, gradient accumulation, clipping, best-checkpoint, log CSV, **guardia anti-spilling** |

### 3.20 `scripts/train_torch.py` — CLI GPU

Sottocomandi `train` (tokens, bpe, out, n-layer/head/embd, block-size, micro-batch,
grad-accum, steps, lr, warmup, eval-every, compile) e `generate` (ckpt, bpe, prompt, n,
temperature, top-k).

### 3.21 `data/prepare_corpus.py` — script di preparazione corpus (Percorso A)

Funzioni (tutte a livello di modulo; script eseguibile con `python data/prepare_corpus.py [--force]`):

| Funzione | Firma | Effetto |
|---|---|---|
| `download` | `(force: bool = False) -> str` | scarica da `URL` o legge cache; **I/O in binario** per non corrompere i CRLF |
| `extract_story` | `(raw: str) -> str` | ritaglia la narrazione tra `STORY_START` e `STORY_END`; guardia se < 200k char |
| `normalize` | `(story: str) -> str` | rimuove `[Illustrazione:…]` e `_`; normalizza nbsp e 2 refusi; collassa righe vuote |
| `report` | `(text: str) -> None` | stampa statistiche: totale char, vocab, elenco vocab, top-20 freq |
| `main` | `() -> None` | orchestration: download → extract → normalize → scrittura + report |

Costanti di modulo:

| Costante | Valore | Significato |
|---|---|---|
| `URL` | `.../files/52484/52484-0.txt` | Gutenberg ebook #52484 (testo integrale) |
| `RAW_PATH` | `data/pinocchio_raw.txt` | cache download (gitignored) |
| `OUT_PATH` | `data/input.txt` | corpus pulito in output |
| `STORY_START` | `"I.\n\nCome andò che Maestro Ciliegia"` | ancora inizio Cap. I |
| `STORY_END` | `"FINE."` | ancora fine storia |

---

## 4. Il corpus prodotto (`data/input.txt`)

| Proprietà | Valore |
|---|---|
| Fonte | Project Gutenberg, ebook **#52484**, *Le avventure di Pinocchio* (Collodi) |
| Caratteri totali | **240.920** |
| Vocabolario | **69 simboli** |
| Split train / val | 216.828 / 24.092 (90% / 10%, contiguo) |
| Encoding su disco | UTF-8, **solo `\n`** (nessun `\r`) |

Vocabolario completo (69): `\n` `spazio` `! ' ( ) , - . : ; ?` cifre `1 4 8`,
maiuscole `A B C D E F G H I J L M N O P Q R S T U V X Z`, minuscole
`a b c d e f g h i j l m n o p q r s t u v z`, accentate `à è é? ì ò ù È`,
caporali `« »`, em-dash `—`.
*(NB: mancano volutamente K/W/Y/k/w/x/y — non compaiono nel testo italiano di
Collodi; e le cifre sono solo 1/4/8, le uniche presenti.)*

---

## 5. Configurazione

Nessun file di config né variabili d'ambiente al momento. I parametri sono
argomenti di funzione con default:

| Parametro | Dove | Default | Significato |
|---|---|---|---|
| `val_frac` | `Dataset.__init__` | `0.1` | frazione di corpus tenuta per la validation |
| `block_size` | `Dataset.get_batch` | — (obbligatorio) | lunghezza finestra di contesto (in caratteri) |
| `batch_size` | `Dataset.get_batch` | — (obbligatorio) | numero di finestre per batch |
| `--force` | CLI `prepare_corpus.py` | assente | riscarica ignorando la cache |

---

## 6. Catalogo dei test

Runner: `python run_tests.py` (nessun pytest richiesto). **94 test, tutti verdi.**

`tests/test_bpe.py` (9): round-trip italiano; **non fallisce mai** su emoji/cirillico/
giapponese mai visti; stringa vuota; vocab rispettato; la compressione migliora col
vocabolario; training deterministico; save/load; i merge appresi sono italiani; nessun
token attraversa un confine di parola (le sequenze di spazi sono l'eccezione legittima).

`tests/test_equivalence.py` (7) — la prova che il port PyTorch è corretto:

| Test | Cosa dimostra |
|---|---|
| `test_forward_logits_match` | stessi pesi+input → stessi logit (1.8e-15) |
| `test_loss_matches` | stessa loss (4.4e-16) |
| `test_gradients_match` | gradienti identici su **tutti** i 38 parametri (8.7e-17) |
| `test_all_parameters_are_covered` | guardia: i due modelli hanno lo stesso numero di parametri |
| `test_training_steps_match` | 5 passi di AdamW: loss identiche fino alla 14ª cifra |
| `test_fast_attention_forward_matches` | la variante ottimizzata non cambia i numeri |
| `test_fast_attention_gradients_match` | …nemmeno i gradienti (incluso il QKV fuso) |


`tests/test_gpt.py` (8): shape logit/loss, loss iniziale ~log(V), tutti i parametri
con gradiente, generazione valida/riproducibile, top-k restringe il supporto,
temperature→0 = greedy, **save/load** dà logit identici, breve training scende.

`tests/test_train.py` (4): warmup lineare, cosine decay a min_lr, grad-clip limita la
norma e lascia intatti i gradienti piccoli.

`tests/test_block.py` (6):

| Test | Cosa dimostra |
|---|---|
| `test_layernorm_normalizes` | output a media ~0 e varianza ~1 sull'ultima dim |
| `test_layernorm_gradcheck` | gradiente di LayerNorm corretto (x, gamma, beta) |
| `test_multihead_and_ffn_preserve_shape` | MHA e FFN: `(B,T,C)` → `(B,T,C)` |
| `test_block_preserves_shape` | il blocco preserva la shape (impilabile) |
| `test_block_all_params_get_gradient` | ogni parametro del blocco riceve gradiente |
| `test_block_lm_trains` | un mini-LM con un Block addestra (loss in calo) |


`tests/test_attention.py` (4):

| Test | Cosa dimostra |
|---|---|
| `test_head_output_shape` | Head: `(B,T,C)` → `(B,T,head_size)` |
| `test_attention_is_causal_and_normalized` | att triangolare inferiore, righe sommano a 1 |
| `test_attention_gradients_flow` | tutti i parametri (Q,K,V,…) ricevono gradiente |
| `test_attention_lm_trains` | l'AttentionLM addestra: loss scende sotto l'uniforme |


`tests/test_mlp.py` (5):

| Test | Cosa dimostra |
|---|---|
| `test_parameters_collected` | `Module.parameters()` raccoglie tutti e 5 i tensori |
| `test_logits_shape_and_scalar_loss` | logits `(B, vocab)`, loss scalare |
| `test_all_params_get_gradient` | dopo backward ogni parametro ha gradiente ≠ 0 |
| `test_training_decreases_loss` | l'addestramento (SGD) fa scendere la loss |
| `test_mlp_beats_bigram_on_val` | NLL val < 2.2 → batte il bigram (milestone M2) |


`tests/test_autograd.py` (18): un gradient check per ogni operazione — `add`
(broadcast), `sub`, `mul` (broadcast), `div`, `pow`, `matmul` 2D e batch, `sum`
(axis), `mean`, `relu`, `tanh`, `exp`, `log`, `gelu`, `softmax`, `cross_entropy`,
tensore riusato (accumulo), mini-MLP composita. Ognuno dimostra che il backward
analitico coincide col numerico (differenze finite centrali) entro 1e-5.


`tests/test_bigram_neural.py` (5):

| Test | Cosa dimostra |
|---|---|
| `test_initial_loss_is_about_log_vocab` | loss iniziale ≈ log(V): il modello parte ~uniforme (sanity init) |
| `test_minibatch_training_decreases_loss` | il ciclo stocastico su minibatch fa scendere la loss |
| `test_converges_to_count_bigram` | la NLL appresa raggiunge quella del bigram a conteggio (±0.05) |
| `test_learned_rows_match_counts_argmax` | dopo `q`, il carattere più probabile coincide coi conteggi |
| `test_gradient_check_numerical_vs_analytic` | `dW` analitico ≈ numerico (differenze finite, err rel < 1e-4) |


`tests/test_bigram_count.py` (6):

| Test | Cosa dimostra |
|---|---|
| `test_counts_on_toy_corpus` | i conteggi `N` sono corretti su "abab" noto |
| `test_rows_are_probability_distributions` | ogni riga di `P` somma a 1 |
| `test_smoothing_removes_zeros` | con smoothing nessuna probabilità è 0 |
| `test_beats_uniform_on_train_and_val` | NLL bigram < NLL uniforme (impara qualcosa) |
| `test_val_nll_is_finite` | NLL su val finita (lo smoothing evita `-inf`) |
| `test_generation_is_valid_and_reproducible` | indici validi; stesso seed → stessa generazione |


`tests/test_tokenizer.py` (8):

| Test | Cosa dimostra |
|---|---|
| `test_roundtrip_on_full_corpus` | `decode(encode(text)) == text` su tutto Pinocchio (nessuna perdita) |
| `test_roundtrip_short_string` | round-trip su stringa breve con `\n` e punteggiatura |
| `test_vocab_is_deterministic_and_sorted` | due costruzioni → stesso vocab; `chars` ordinato |
| `test_vocab_size_matches_unique_chars` | `vocab_size` = numero di caratteri distinti |
| `test_encode_raises_on_unknown_char` | `encode` solleva `ValueError` su carattere ignoto |
| `test_decode_raises_on_out_of_range_index` | `decode` solleva `ValueError` su indice fuori range |
| `test_indices_are_contiguous_from_zero` | gli indici sono `0..vocab_size-1` senza buchi |
| `test_save_load_roundtrip` | `save`→`load` preserva `chars`, `stoi`, e la codifica |

`tests/test_dataset.py` (8):

| Test | Cosa dimostra |
|---|---|
| `test_batch_shapes` | `X`,`Y` hanno shape `(batch, block)` e dtype `int64` |
| `test_Y_is_X_shifted_by_one` | invariante di shift: `X[:, 1:] == Y[:, :-1]` |
| `test_split_is_contiguous_and_disjoint` | train+val = corpus, contigui, val ≈ 10% |
| `test_reproducibility_same_seed_same_batch` | stesso seed → batch identico |
| `test_different_seed_different_batch` | seed diverso → batch diverso |
| `test_batch_content_matches_source_windows` | valori nei range validi del vocab |
| `test_unknown_split_raises` | split ≠ train/val solleva `ValueError` |
| `test_too_short_split_raises` | `block_size` > dati disponibili solleva `ValueError` |

---

## 7. Regole non negoziabili (attive da subito)

1. **Riproducibilità**: ogni estrazione casuale passa per un `np.random.Generator`
   creato con seed esplicito (`np.random.default_rng(seed)`). Mai `np.random.*`
   globale.
2. **Determinismo del vocabolario**: `CharTokenizer` ordina sempre i caratteri.
   Non cambiare l'ordinamento: romperebbe ogni checkpoint futuro.
3. **Fail-loud**: `encode`/`decode`/`get_batch` sollevano su input illegale invece
   di produrre risultati silenziosamente sbagliati.
4. **Provenienza dei dati**: `data/input.txt` si (ri)genera solo via
   `prepare_corpus.py`. Non modificarlo a mano.

---

## 8. Trappole già disinnescate (per non farsi mordere due volte)

1. **Ebook sbagliato**: Gutenberg #19517 ha lo stesso titolo ma è una *lettura
   audio* (23 KB, solo elenco capitoli). Il testo integrale è **#52484**.
2. **Doppia traduzione dei fine-riga su Windows**: scrivere/leggere la cache con
   `write_text`/`read_text` converte `\n`→`\r\n` due volte e corrompe i CRLF,
   sballando le ancore testuali. → Cache e output scritti/letti **in binario**
   (`read_bytes`/`write_bytes`). Se rivedi `prepare_corpus.py`, non tornare a
   `write_text`.
3. **Artefatti Gutenberg nel vocab**: `[Illustrazione: …]` (didascalie) e `_…_`
   (corsivo) aggiungevano `[ ] _ "` al vocabolario. Rimossi in `normalize()`.
   Se il vocab torna a > 69, controlla che questi filtri siano ancora attivi.
4. **Console Windows cp1252**: stampare `—`, `«` ecc. crasha con `UnicodeEncodeError`.
   → `sys.stdout.reconfigure(encoding='utf-8')` negli script che stampano il corpus.

---

## 9. Debito tecnico aperto

| Voce | Perché rimandato | Quando affrontarlo |
|---|---|---|
| `int64` per il corpus | semplice e corretto; spreco di memoria trascurabile a questa scala | Fase 11 (corpus grande) userà `uint16` con `memmap` |
| Nessun `pyproject.toml`/packaging | non serve per eseguire; `sys.path.insert` nei test basta | quando/se il progetto va installato come pacchetto |
| Test senza pytest | pytest non è installato nell'ambiente | opzionale: aggiungere pytest resta compatibile (i test sono già in stile pytest) |

---

## 10. Il *perché* delle scelte non ovvie (in una riga ciascuna)

- **Char-level, non BPE** (Fase 0): il tokenizer smette di essere un tema; il BPE
  arriva scritto a mano in Fase 10, quando serve davvero (Percorso B).
- **Split contiguo, non a caratteri sparsi**: caratteri sparsi renderebbero la
  "verifica su testo mai visto" una finzione (i vicini sarebbero nel train).
- **Posizioni casuali nei batch**: la SGD rende meglio con esempi scorrelati;
  finestre consecutive del libro seguirebbero la trama, non la lingua.
- **`Y` = `X` shiftato di 1**: una finestra da `T` caratteri produce `T` esempi in
  un colpo — il motore dell'efficienza dei transformer (Fasi 5/7).
- **`save`/`load` del vocab già in Fase 0**: il checkpoint di Fase 7 dovrà legare
  pesi e mappa indici; separarli produce spazzatura deterministica.
