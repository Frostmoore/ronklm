# RonkLM

Un piccolo Language Model costruito **da zero per capire come funziona**, per gradi:
dal modello più stupido che esista (conteggio di bigrammi) fino a un GPT completo,
**senza una sola riga di PyTorch** nel Percorso A — motore di autograd, self-attention
e ottimizzatore AdamW sono tutti scritti a mano, in NumPy puro.

> *Se dopo una fase non sai spiegare **perché** un numero diventa quel numero, la fase
> non è chiusa.*

## Cosa sa fare, in una tabella

Ogni modello è addestrato/valutato sulla stessa metrica (NLL di validation, in nats)
sul testo di *Pinocchio*. Ogni riga è un'idea in più e il miglioramento che ha comprato:

| Modello | Fase | NLL val | Perplexity | Idea aggiunta |
|---|---|---|---|---|
| uniforme (a caso) | — | 4.234 | 69 | niente |
| Bigram (conteggio) | 1 | 2.346 | 10.4 | distribuzione sul prossimo carattere |
| Bigram neurale | 2 | ~2.35 | 10.4 | discesa del gradiente (stesso modello, imparato) |
| MLP (contesto 8) | 4 | 1.897 | 6.7 | embedding + contesto + strato nascosto |
| **GPT (RonkLM v1)** | 7 | **1.632** | **5.1** | attention + posizione + profondità |

## La galleria dell'evoluzione

Stesso compito (generare testo), tre modelli, la differenza si *vede*:

**Bigram (Fase 1)** — pseudo-italiano sillabico:
```
lona s... de filì; — bi E griede pe sccocoll lesil e arevemin facavancomasuto
```

**MLP (Fase 4)** — compaiono parole vere e nomi:
```
luspau trome, introva ibbecio ma cold'omestro: — Mi fiariventi. i burattino e
titasse questo di fuoresono un grate fuino.
```

**GPT / RonkLM v1 (Fase 7)** — parole reali, dialoghi, il nome "Pinocchio":
```
Pinocchio a stare il mio tirò Pinocchio, con mi pesse diretto dalla spaggio dettorna
da di farò a parere di gallina di grande di sè: — Ma in poco da nottere a casa
mozzare a un bel pesce di mani un po' di piedi. — No, rangia si disse:
```

Non è italiano coerente — con ~160k parametri e mezzo MB di testo su CPU non potrebbe
esserlo — ma la *forma* dell'italiano narrativo c'è. È il risultato corretto a questa
scala: lo scopo è **vedere la macchina imparare e capirne ogni pezzo**.

## Struttura del progetto (due percorsi)

- **Percorso A — capire** (Fasi 0–8, ✅ completo): NumPy puro, GPT ~160k–1M parametri,
  char-level, corpus *Pinocchio*. Tutto scritto e testato a mano.
- **Percorso B — scalare** (Fasi 9–12, da fare): port a PyTorch *dimostrato
  equivalente*, tokenizer BPE, corpus da gigabyte, **~50M parametri**.

Il piano dettagliato è in [`memory/plan_ronklm_system.md`](memory/plan_ronklm_system.md);
la spiegazione didattica dai primi principi in [`explain.md`](explain.md); l'atlante
tecnico del codice in [`memory/codebase_reference.md`](memory/codebase_reference.md).

## Quickstart

```bash
pip install numpy                 # unica dipendenza del Percorso A

# 1) prepara il corpus (scarica e pulisce Pinocchio da Project Gutenberg)
python data/prepare_corpus.py

# 2) esegui i test (nessun pytest richiesto)
python run_tests.py               # 78 test, tutti verdi

# 3) addestra un GPT e salva il miglior checkpoint su validation
python scripts/train_ronklm.py train --steps 3000 --n-layer 3 --n-embd 64 \
    --block-size 32 --out checkpoints/ronklm.npz

# 4) genera testo dal checkpoint
python scripts/train_ronklm.py generate --ckpt checkpoints/ronklm.npz \
    --prompt "Pinocchio " --n 400 --temperature 0.8 --top-k 20
```

## Esperimenti (one-factor-at-a-time)

Run brevi (1000 passi, embd 48) che mostrano il *trend* di due iperparametri. Dettagli
e interpretazione in [`explain.md`](explain.md#sec-8-4).

**Profondità** (block 32): più blocchi → NLL più bassa, monotono.

| n_layer | params | NLL val |
|---|---|---|
| 1 | 36k | 2.030 |
| 2 | 65k | 1.942 |
| 4 | 121k | 1.886 |

**Contesto** (n_layer 2): a *budget corto* il contesto lungo non fa in tempo a ripagare.

| block_size | params | NLL val |
|---|---|---|
| 16 | 64k | 1.944 |
| 32 | 65k | 1.942 |
| 64 | 66k | 1.986 |

## Requisiti

- Percorso A: Python 3.10+, `numpy` (e `matplotlib` opzionale per i grafici).
- Percorso B: in aggiunta, `torch` (introdotto solo dalla Fase 9).

## Estensioni naturali (post Percorso A)

Ognuna è il "capitolo successivo" logico, elencata col perché sarebbe il passo giusto:

- **BPE** (Fase 10): il char-level spreca contesto e capacità; i token subword
  densificano la sequenza.
- **Port a PyTorch** (Fase 9): stessa architettura, ma su GPU → 50M parametri diventano
  addestrabili; con un test di *equivalenza numerica* contro questo motore.
- **RoPE, weight tying, dropout**: raffinamenti architetturali standard dei GPT moderni.
- **Fine-tuning istruzioni / RLHF**: trasformare il modello da *completatore* di testo
  ad *assistente* — un altro progetto intero.
