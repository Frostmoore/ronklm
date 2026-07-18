# RonkLM

Un piccolo Language Model costruito **da zero per capire come funziona**, per gradi:
dal modello più stupido che esista (conteggio di bigrammi) fino a un GPT da ~50M di
parametri capace di scrivere italiano sensato a livello di frase e paragrafo.

Il progetto è diviso in due percorsi:

- **Percorso A — capire** (Fasi 0–8): tutto in **NumPy puro**, nessun framework,
  nessun autograd nascosto. Si scrive a mano ogni gradiente, un motore di
  backpropagation completo, l'ottimizzatore AdamW e la self-attention. Modello
  finale ~1M parametri, char-level, addestrato su *Pinocchio*. Scopo: capire ogni
  singola riga.
- **Percorso B — scalare** (Fasi 9–12): port a **PyTorch** *dimostrato numericamente
  equivalente* al motore NumPy, tokenizer **BPE scritto a mano**, corpus italiano da
  gigabyte, training su GPU fino a **~50M di parametri**.

## Stato

In sviluppo. Il piano dettagliato — con il *perché* di ogni scelta implementativa —
è in [`memory/plan_ronklm_system.md`](memory/plan_ronklm_system.md).

## Filosofia

> Se dopo una fase non sai spiegare *perché* un numero diventa quel numero, la fase
> non è chiusa.

## Requisiti

- Percorso A: Python 3.10+, `numpy` (e `matplotlib` opzionale).
- Percorso B: in aggiunta, `torch` (introdotto solo dalla Fase 9).
