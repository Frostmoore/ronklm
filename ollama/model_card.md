# RonkLM-1 (0.05b)

> **Descrizione pubblicata su ollama.com (dall'autore):**
> *"Modellino fatto per divertirmi. Spara un sacco di cazzate, ma è comunque più
> divertente di Emma."*

Un piccolo modello linguistico italiano da ~50 milioni di parametri, costruito
interamente da zero — per *capire* come funziona un LLM, non per competere con i grandi.
Ogni singolo pezzo è scritto a mano e compreso riga per riga: dal primo bigram in NumPy
a un motore di autograd, dalla self-attention al tokenizer BPE byte-level, fino alla
conversione in GGUF. Nessuna scatola nera, nessuna libreria di deep learning nascosta.

## Cosa sa fare (onestamente)

✅ Scrive **italiano fluente e grammaticalmente corretto**
✅ Ha una **voce narrativa** (addestrato anche sui classici di Project Gutenberg)
✅ **Risponde alle domande** nel formato di un assistente

❌ **Inventa i fatti con assoluta sicurezza** — a 50M di parametri non c'è spazio per la
   conoscenza, solo per la lingua
❌ Non ragiona e si perde sui testi lunghi
❌ È un **giocattolo didattico**, non uno strumento affidabile

> **D:** Cos'è la carbonara?
> **R:** *La carbonara si trova in depositi di argilla e calcare...*
>
> (sì, la scambia per un minerale: il tokenizer la spezza vicino a "carbon-ato" e parte
> per la tangente. È esattamente il bello — e il limite — di un modello così piccolo.)

## Come è nato

1. **Pretraining** su Wikipedia italiana (~1,1 miliardi di token) — impara la lingua
2. **Continued-pretraining** su Project Gutenberg italiano — prende la voce narrativa
3. **SFT** su istruzioni italiane — impara a rispondere

Architettura in stile GPT-2 (10 layer, 512 dim, 8 teste, contesto 512), tokenizer BPE
byte-level da 16k, addestrato su una singola RTX 4080 Super.

## Uso

```
ollama run ronconiric/ronklm-1:0.05b
```

Parla italiano, inventa tutto, ma è **onesto sul fatto di essere piccolo**. Divertitici.

---

*RonkLM-1 è nato come progetto per imparare come funzionano i modelli linguistici
partendo dai primi principi. Costruito capendo ogni riga di codice.*
