# Piano di sistema — RonkLM

> **Versione del piano**: 2.1 (aggiunto Percorso B: scalata a 50M parametri, 2026-07-18)
> **Stato**: piano approvato in attesa di inizio Fase 0.
>
> **Obiettivo del progetto**, in due percorsi:
>
> - **Percorso A (Fasi 0–8) — capire**: costruire da zero, in **Python + NumPy
>   puro** (nessun framework di deep learning, nessun autograd nascosto), un
>   piccolo GPT char-level (~1M parametri) addestrato su **testo italiano**, con
>   lo scopo esplicito di **capire ogni singola lettera del codice**:
>   tokenizzazione, embedding, loss, backpropagation, self-attention, blocchi
>   transformer e generazione di testo.
> - **Percorso B (Fasi 9–12) — scalare**: una volta capito tutto, portare RonkLM
>   a **~50M di parametri** — port a PyTorch *dimostrato numericamente
>   equivalente* al nostro motore, tokenizer **BPE scritto a mano**, corpus
>   italiano da **gigabyte**, training su GPU — fino a un modellino che genera
>   **italiano sensato a livello di frase e paragrafo** (non GPT-5.6: prosa
>   coerente, grammatica solida, niente fatti né istruzioni — aspettative
>   dettagliate in Fase 12).
>
> Il Percorso B esiste *perché* esiste il Percorso A: scalare una cosa che non si
> è capita è il modo migliore per non capirla mai più.
>
> **Contratto didattico**: il lettore di questo documento parte da zero sul deep
> learning. Ogni scelta implementativa è quindi accompagnata dal suo *perché*, ogni
> formula è spiegata prima di essere usata, e ogni fase dichiara come si fa a
> *verificare* di aver capito. Se una cosa nel codice non è spiegata qui, è un bug
> del documento.

---

## Indice

- [Parte I — Le fondamenta concettuali (da leggere prima di tutto)](#parte-i)
  - I.1 Cos'è davvero un language model
  - I.2 Perché NumPy puro e non PyTorch
  - I.3 Perché un tokenizer a caratteri (char-level)
  - I.4 Perché un percorso a fasi e non "subito il GPT"
  - I.5 Perché testo italiano e quale testo
  - I.6 Glossario minimo
- [Parte II — Metadati, regole e struttura dei file](#parte-ii)
- [Parte III — Le fasi in dettaglio](#parte-iii)
  - Fase 0 — Fondamenta: dati e tokenizer
  - Fase 1 — Bigram per conteggio
  - Fase 2 — Bigram neurale (gradiente a mano)
  - Fase 3 — `ronkgrad`: il micro-motore di autograd
  - Fase 4 — MLP language model
  - Fase 5 — Self-attention
  - Fase 6 — Il blocco Transformer
  - Fase 7 — RonkLM: il GPT completo
  - Fase 8 — Training serio, esperimenti e CLI
- [Parte III-B — Percorso B: la scalata a 50M](#parte-iii-b)
  - Fase 9 — Port a PyTorch con equivalenza dimostrata
  - Fase 10 — Tokenizer BPE scritto a mano
  - Fase 11 — Il corpus grande
  - Fase 12 — RonkLM-50M: architettura, training run, valutazione
- [Parte IV — Milestone, rituale di fine fase, stato](#parte-iv)

---

<a name="parte-i"></a>
# Parte I — Le fondamenta concettuali

Questa parte non contiene sottofasi da implementare: contiene le **decisioni di
progetto** e il loro perché. Va letta per prima perché tutto il resto ne discende.

---

## I.1 Cos'è davvero un language model

Un language model (LM) è una macchina che risponde a una sola domanda, ripetuta
all'infinito:

> *"Dato il testo visto finora, quanto è probabile ciascun possibile prossimo pezzo
> di testo?"*

Tutto qui. Non "capisce", non "pensa" (non a questo livello di descrizione): assegna
una **distribuzione di probabilità sul prossimo token**. Se il testo finora è
`"Il gatto è sul tett"`, un buon LM italiano assegnerà probabilità altissima al
carattere `o`, bassa a `q`, quasi nulla a `%`.

Da questa unica capacità derivano due cose:

1. **Generazione**: se so dire quanto è probabile ogni prossimo carattere, posso
   *estrarne uno a caso secondo quelle probabilità*, aggiungerlo al testo, e
   ripetere. Questo processo si chiama **campionamento autoregressivo** ed è
   esattamente ciò che fa ChatGPT quando "scrive": un carattere (token) alla volta,
   ognuno condizionato da tutti i precedenti.

2. **Valutazione**: se il modello assegna probabilità alta al testo *vero* che
   effettivamente segue, il modello è buono. Questo ci dà una **misura numerica di
   qualità** (la loss, che introdurremo in Fase 1) senza bisogno di giudizi umani.
   È il motivo per cui il training funziona: possiamo dire alla macchina,
   matematicamente, "stavi per sbagliare, correggiti in questa direzione".

**L'intero progetto è una scala di modelli sempre migliori nel rispondere a quella
stessa unica domanda.** Il bigram della Fase 1 la risponde guardando solo l'ultimo
carattere. L'MLP della Fase 4 guarda gli ultimi N caratteri. Il GPT della Fase 7
guarda tutto il contesto e decide *da solo, per ogni carattere, quali parti del
passato contano*. Ma la domanda non cambia mai. Tenerlo a mente evita la sensazione
di "magia" quando arriveremo all'attention: è sempre lo stesso problema, con uno
strumento più espressivo.

---

## I.2 Perché NumPy puro e non PyTorch

La scelta più importante del progetto, e la più costosa. Motivazione completa:

**Cosa nasconde PyTorch.** In PyTorch scrivi `loss.backward()` e i gradienti
"compaiono". Scrivi `nn.Linear(64, 128)` e l'inizializzazione dei pesi, il
prodotto matrice-vettore e la propagazione all'indietro del gradiente sono tutti
invisibili. Per *usare* i modelli è perfetto; per *capirli* è un ostacolo, perché
il 90% della comprensione di un LLM sta esattamente nelle parti che PyTorch
nasconde: **come si calcola un gradiente, come fluisce all'indietro attraverso una
rete, e perché certe architetture lo fanno fluire meglio di altre**.

**Cosa ci dà NumPy.** NumPy fornisce solo array N-dimensionali e operazioni
matematiche veloci su di essi (somme, prodotti di matrici, esponenziali). Nessun
gradiente, nessun layer, nessun ottimizzatore. Tutto ciò che è "deep learning"
dovremo scriverlo noi. Questo significa che alla fine del progetto avrai scritto
con le tue mani (o letto riga per riga, avendone la derivazione su carta):

- la formula del gradiente della cross-entropy (Fase 2);
- un motore di backpropagation completo (Fase 3);
- l'ottimizzatore AdamW, lo stesso usato per addestrare GPT-4 (Fase 4);
- la self-attention, riga per riga (Fase 5).

**Il costo, dichiarato onestamente.** Tre svantaggi reali:
1. *Lentezza*: NumPy gira su CPU. Il nostro GPT finale dovrà restare piccolo
   (qualche centinaio di migliaia di parametri, non miliardi) e i training
   dureranno minuti/ore, non secondi. Va bene: l'obiettivo è capire, non competere.
2. *Fatica*: scrivere il backward a mano è il punto dove si sbaglia di più. Per
   questo la Fase 3 ha una batteria di test automatici (il "gradient check") che
   confronta ogni nostro gradiente con una stima numerica indipendente. Non ci si
   fida mai di un gradiente non testato.
3. *Niente GPU*: se un giorno vorrai scalare, dovrai riscrivere in PyTorch. Ma a
   quel punto PyTorch sarà *trasparente*: ogni sua API corrisponderà a qualcosa che
   hai già scritto tu. Questo è il vero deliverable del progetto.

**Perché non una via di mezzo** (es. PyTorch solo per l'autograd): perché
l'autograd È il cuore della questione. Un LLM si addestra per discesa del
gradiente; se il gradiente è una scatola nera, il training è una scatola nera.

**E allora i 50M di parametri? — i conti, fatti onestamente.** Addestrare un
transformer costa ≈ `6 × parametri × token` operazioni in virgola mobile (regola
empirica standard: 2 per il forward, 4 per il backward). Per 50M di parametri su
~1 miliardo di token: ~3×10¹⁷ FLOP. Una CPU consumer in NumPy sostiene realmente
~10–50 GFLOP/s su questi carichi → **mesi o anni di calcolo ininterrotto**. Una
GPU consumer moderna (che PyTorch sa usare) sostiene decine di TFLOP/s in mixed
precision → **giorni**. Non è una questione di ottimizzazione del nostro codice:
è un muro di 3–4 ordini di grandezza tra CPU-NumPy e GPU-PyTorch. Da qui la
struttura a due percorsi: **il NumPy puro serve a capire (Percorso A, ~1M
parametri, dove il CPU basta); PyTorch serve a scalare (Percorso B, 50M)** — e
arriva *solo dopo*, in Fase 9, quando ogni sua API corrisponderà a codice che
abbiamo già scritto e testato noi. Il port includerà un test di equivalenza
numerica (9.2): stesso input, stessi pesi → stessi logits nei due motori. È il
momento in cui PyTorch smette per sempre di essere magia.

---

## I.3 Perché un tokenizer a caratteri (char-level)

Un modello non può mangiare testo: mangia numeri. Il **tokenizer** è il traduttore
testo→numeri (e ritorno). Esistono tre famiglie di scelte:

1. **Char-level**: ogni carattere è un token. `"ciao"` → `[12, 18, 10, 24]`
   (indici nel vocabolario dei caratteri). Vocabolario piccolo: per l'italiano,
   ~80–120 simboli (lettere, accentate, punteggiatura, spazio, a-capo).
2. **Word-level**: ogni parola è un token. Vocabolario enorme (centinaia di
   migliaia di parole italiane) e problema irrisolvibile delle parole mai viste.
3. **Subword / BPE** (quello dei GPT veri): pezzi di parola statisticamente
   frequenti (`"mangi"+"ando"`). Il compromesso migliore in produzione.

**Scegliamo char-level, e il perché merita una spiegazione completa:**

- **Il tokenizer smette di essere un tema.** Il BPE è un algoritmo interessante ma
  *ortogonale* a come funziona la rete neurale: è pre-processing statistico. Se lo
  mettessimo nel percorso, spenderemmo una fase intera su un tema che non tocca
  gradienti, attention o training. Char-level si implementa in 20 righe
  (due dizionari: carattere→indice e indice→carattere) e ci leva il pensiero.

- **Vocabolario piccolo = modello piccolo = training fattibile su CPU.** L'ultimo
  layer di ogni LM produce un punteggio *per ogni token del vocabolario*. Con
  vocab ≈ 100 quel layer è minuscolo; con vocab ≈ 50.000 (BPE dei GPT veri)
  sarebbe da solo più grande di tutto il nostro modello.

- **La generazione è più divertente da osservare.** A livello di carattere *vedi*
  il modello imparare per gradi: prima genera spazzatura casuale, poi rispetta le
  frequenze delle lettere italiane, poi inventa parole pronunciabili
  ("il gatto si mangiova"), poi azzecca parole vere, poi pezzi di sintassi. Questa
  progressione visibile è oro didattico: colleghi la discesa della loss a
  qualcosa che si vede a occhio.

- **Il costo**: le sequenze diventano lunghe (una frase = ~60 caratteri, non ~12
  subword), quindi a parità di finestra di contesto il modello "vede" meno testo.
  Per un progetto didattico è un costo accettabilissimo.

- **Il BPE non è perso — è la Fase 10.** Nel Percorso B il char-level non basta
  più (il perché quantitativo è spiegato lì) e il BPE lo scriveremo **a mano**,
  come tutto il resto. Qui nel Percorso A resterebbe solo un ostacolo tra noi e i
  concetti che contano.

---

## I.4 Perché un percorso a fasi e non "subito il GPT"

Si potrebbe scrivere direttamente il GPT (Fase 7) copiando un'architettura nota.
Girerebbe. E non avresti capito **niente**, perché ogni componente del transformer
è la *soluzione a un problema che hanno i modelli più semplici* — e se non hai mai
toccato quei modelli, non hai mai visto il problema:

| Componente del GPT | È la soluzione a questo problema | Dove vivremo il problema |
|---|---|---|
| Loss (cross-entropy) | "come misuro quanto è buono un modello probabilistico?" | Fase 1 |
| Discesa del gradiente | "come *miglioro* i pesi invece di contarli?" | Fase 2 |
| Autograd | "derivare a mano non scala oltre 1 layer" | Fase 3 |
| Embedding | "one-hot spreca spazio e non cattura somiglianze" | Fase 4 |
| Contesto ampio | "il bigram vede 1 solo carattere: troppo poco" | Fasi 1→4 |
| Self-attention | "l'MLP ha contesto rigido e pesi non riusabili tra posizioni" | Fase 5 |
| Residual + LayerNorm | "le reti profonde non si addestrano: i gradienti muoiono" | Fase 6 |
| Positional embedding | "l'attention da sola è cieca all'ordine delle parole" | Fase 7 |

Il percorso a fasi è quindi una catena di *problema → soluzione → nuovo problema*.
Ogni fase produce un modello **funzionante e misurabile** (con la sua loss sulla
stessa metrica), quindi a fine progetto avrai una tabella: bigram ~X, MLP ~Y,
GPT ~Z, e saprai esattamente *quale idea* ha comprato ogni miglioramento.

Questo è il percorso di *makemore* e *nanoGPT* di Andrej Karpathy (i riferimenti
didattici migliori esistenti su questo tema), adattato in due modi: (a) niente
PyTorch — il motore lo scriviamo noi; (b) corpus italiano.

---

## I.5 Perché testo italiano e quale testo

**Requisiti sul corpus**: (1) italiano; (2) abbastanza grande da non essere
memorizzato subito ma abbastanza piccolo da addestrarci sopra su CPU → ideale
0.3–1.5 MB; (3) pubblico dominio, così il progetto è pulito legalmente e
ridistribuibile; (4) prosa omogenea (un solo autore/stile), perché un modello
piccolo su un corpus stilisticamente uniforme produce risultati molto più
riconoscibili e divertenti che su un miscuglio.

**Scelta proposta: *Le avventure di Pinocchio* di Carlo Collodi** (1883, pubblico
dominio, ~500 KB di testo pulito, scaricabile da LiberLiber/Project Gutenberg).
Perché proprio Pinocchio:
- lessico ricco ma ripetitivo il giusto (nomi propri ricorrenti: Geppetto, la Fata,
  il Grillo — il modello li imparerà e li ricombinerà, ed è visibilissimo);
- dialoghi frequenti → il modello impara la struttura `— disse …` e le virgolette,
  altro progresso visibile a occhio;
- prosa ottocentesca ma semplice.

Alternativa se vorremo più dati in Fase 8: *I Promessi Sposi* (~1.3 MB) o la
concatenazione dei due. La decisione finale del file esatto e della sua pulizia
(rimozione header/footer di Gutenberg, normalizzazione apostrofi tipografici) è la
sottofase 0.2.

**Aspettative oneste (per il Percorso A)**: con ~500 KB e un modello da ~1M
parametri char-level, RonkLM genererà *italiano collodiano plausibile ma
incoerente*: frasi grammaticalmente quasi giuste, parole quasi tutte vere, senso
globale assente. È il risultato corretto a questa scala — GPT-2 usava 40 GB di
testo e 1.5 miliardi di parametri. Nel Percorso A lo scopo è vedere la macchina
*imparare*; il testo *sensato* è l'obiettivo del Percorso B, che per questo avrà
bisogno di un corpus da gigabyte (Fase 11: il perché della quantità — le leggi di
scala — è spiegato lì).

---

## I.6 Glossario minimo

Termini usati in tutto il documento. Torna qui ogni volta che serve.

- **Token**: l'unità atomica di testo per il modello. Per noi: un carattere.
- **Vocabolario (vocab)**: l'insieme dei token possibili. `vocab_size` = quanti sono.
- **Tensore**: array di numeri a N dimensioni. Un vettore è 1-D, una matrice 2-D.
- **Shape**: le dimensioni di un tensore. Una matrice 32×100 ha shape `(32, 100)`.
  *Il debugging di reti neurali è per l'80% ragionamento sulle shape.*
- **Logits**: i punteggi grezzi (numeri reali qualsiasi, anche negativi) che il
  modello produce per ogni token del vocabolario, *prima* di trasformarli in
  probabilità. Nome storico, va imparato: si usa ovunque.
- **Softmax**: la funzione che trasforma logits in probabilità (tutte positive, a
  somma 1). Definita e spiegata in Fase 2.
- **Loss**: il numero unico che misura quanto il modello sta sbagliando. Più basso
  = meglio. Tutto il training è "rendi piccola la loss".
- **Gradiente**: per ogni peso del modello, la direzione (e intensità) in cui
  muovere quel peso per far *salire* la loss. Muovendosi in direzione opposta la
  loss scende: questa è la **discesa del gradiente**.
- **Backpropagation (backprop)**: l'algoritmo che calcola i gradienti di tutti i
  pesi in modo efficiente, propagandoli dall'output all'indietro.
- **Parametri / pesi**: i numeri *modificabili* del modello, quelli che il training
  aggiusta. "Modello da 1M di parametri" = 1 milione di numeri regolabili.
- **Batch**: un gruppo di esempi processati insieme (per efficienza e stabilità).
- **Epoca / step**: uno *step* è un aggiornamento dei pesi su un batch; un'epoca è
  un passaggio su tutto il dataset. Noi ragioneremo in step.
- **Overfitting**: il modello memorizza il training set invece di imparare regole
  generali. Si diagnostica confrontando loss di training e di validazione.
- **Inferenza / generazione**: usare il modello addestrato (senza più aggiornarlo).

---

<a name="parte-ii"></a>
# Parte II — Metadati, regole e struttura dei file

## II.1 Metadati

| Voce | Valore |
|---|---|
| Nome | RonkLM |
| Linguaggio | Python 3.10+ |
| Dipendenze runtime (Percorso A) | `numpy` (l'unica). `matplotlib` opzionale, solo per grafici di loss. |
| Anti-dipendenze (Percorso A) | **Vietati** PyTorch, TensorFlow, JAX, scikit-learn, HuggingFace. |
| Dipendenze (Percorso B) | `torch` (dalla Fase 9, dopo il test di equivalenza col nostro motore) |
| Modello Percorso A | Transformer decoder-only, char-level, ~0.5–1.5M parametri |
| Modello Percorso B | Transformer decoder-only, BPE ~16k, **~50M parametri** |
| Corpus Percorso A | *Pinocchio* (Collodi), pubblico dominio — decisione finale in 0.2 |
| Corpus Percorso B | italiano, ordine dei GB (Wikipedia IT + libri PD + web pulito) — Fase 11 |
| Documento atlante | `memory/codebase_reference.md` (creato a fine Fase 1) |
| Versionamento | branch `v1.0.0` in avanti, incrementi per entità della modifica |
| Remote git | `github` → `https://github.com/Frostmoore/ronklm.git` · `gitea` → `https://git.home.varitest.ovh/smp-webmaster/ronklm.git` — **ogni push va su entrambi** |
| Hardware Percorso A | CPU qualsiasi; nessuna GPU richiesta |
| Hardware Percorso B | GPU necessaria per la Fase 12 (propria o noleggiata — decisione aperta, IV.4) |

## II.2 Regole non negoziabili del progetto

1. **Nessun gradiente non testato.** Ogni operazione con un backward scritto a mano
   ha un gradient check numerico in `tests/`. (Il perché è in Fase 3.)
2. **Nessuna formula non spiegata.** Se il codice usa una formula, il
   `codebase_reference.md` ne contiene la derivazione o il rimando alla sezione di
   questo piano che la deriva.
3. **Riproducibilità**: ogni training fissa il seed del generatore casuale
   (`np.random.default_rng(seed)`), così due esecuzioni identiche danno numeri
   identici e i bug sono inseguibili.
4. **Una metrica, sempre la stessa**: la loss di validazione in *nats* (log
   naturale). Ogni fase la riporta, così i modelli sono confrontabili.
5. **Shape commentate**: ogni funzione che manipola tensori dichiara nel codice le
   shape di input e output (es. `# (B, T) -> (B, T, C)`). È la singola abitudine
   che più riduce i bug nelle reti neurali.

## II.3 Struttura dei file (target finale)

```
RonkLM/
├── memory/
│   ├── plan_ronklm_system.md        # questo file
│   └── codebase_reference.md        # atlante del codice (da fine Fase 1)
├── data/
│   ├── input.txt                    # corpus italiano pulito
│   └── prepare_corpus.py            # script scarico+pulizia (riproducibilità)
├── ronklm/
│   ├── __init__.py
│   ├── tokenizer.py                 # CharTokenizer (Fase 0)
│   ├── dataset.py                   # split train/val, get_batch (Fase 0)
│   ├── autograd.py                  # ronkgrad: Tensor con backward (Fase 3)
│   ├── nn.py                        # Module, Linear, Embedding, LayerNorm… (Fase 4+)
│   ├── optim.py                     # SGD, AdamW (Fasi 2/4)
│   ├── models/
│   │   ├── bigram_count.py          # Fase 1
│   │   ├── bigram_neural.py         # Fase 2
│   │   ├── mlp.py                   # Fase 4
│   │   ├── attention.py             # Fase 5
│   │   ├── block.py                 # Fase 6
│   │   └── gpt.py                   # Fase 7
│   ├── train.py                     # loop di training generico (Fase 8)
│   └── generate.py                  # campionamento (Fase 7)
├── tests/
│   ├── test_tokenizer.py
│   ├── test_dataset.py
│   ├── test_autograd.py             # il più importante del progetto
│   ├── test_nn.py
│   └── test_models.py
├── scripts/
│   └── train_ronklm.py              # entrypoint CLI (Fase 8)
├── requirements.txt
└── README.md
```

La struttura nasce incrementalmente: in Fase 0 esistono solo `tokenizer.py`,
`dataset.py`, `data/` e i primi test.

**Il Percorso B aggiungerà** (dettagli nelle Fasi 9–11): `ronklm/bpe.py` (il
tokenizer BPE), `ronklm_torch/` (il port PyTorch, *separato* dal pacchetto NumPy —
i due motori convivono, il primo resta il riferimento didattico e il banco di
prova di equivalenza del secondo), `data/corpus_b/` (pipeline del corpus grande,
fuori da git) e `tests/test_equivalence.py`.

---

<a name="parte-iii"></a>
# Parte III — Le fasi in dettaglio

Formato di ogni fase: **obiettivo didattico** → per ogni sottofase: *cosa* facciamo,
*perché* (la parte lunga), e *come verifichiamo*. I checkbox tracciano lo stato.

---

## ☑ Fase 0 — Fondamenta: dati e tokenizer  ✅ COMPLETATA (2026-07-18)

**Obiettivo didattico.** Prima di qualsiasi modello, interiorizzare la catena
`testo → numeri → tensori → batch`. Ogni LLM del mondo, GPT-4 incluso, inizia
esattamente così; le differenze sono solo di scala.

> **Esito**: pipeline completa e testata (16/16 test verdi). Corpus Pinocchio
> pulito: **240.920 caratteri, vocabolario di 69 simboli**. Dataset: train
> 216.828 / val 24.092. Il `codebase_reference.md` è stato creato già ora (una
> fase in anticipo rispetto al piano) perché c'era codice sostanziale da mappare.

### ☑ 0.1 — Scaffolding del progetto

**Cosa**: `git init`, branch `v1.0.0`, `requirements.txt` (solo `numpy`),
`ronklm/__init__.py`, `README.md` minimale, `.gitignore` (cache Python, checkpoint).

**Perché così**: partire con la struttura a pacchetto (`ronklm/` importabile) e non
con script sciolti evita il refactoring doloroso a metà progetto, quando i moduli
dovranno importarsi a vicenda (il modello importa l'autograd, il training importa
tutto). Il `.gitignore` sui checkpoint evita di committare file binari da megabyte
nella storia git. Git dal primo giorno perché il rituale di fine fase richiede
branch versionati, e perché nei progetti dove "si sbaglia per imparare" la
possibilità di tornare indietro a uno stato funzionante è una rete di sicurezza
didattica, non solo tecnica.

> ✅ Fatto: `git init` su branch `v1.0.0`, remote `github`+`gitea`,
> `requirements.txt`, `ronklm/__init__.py`, `README.md`, `.gitignore`.

### ☑ 0.2 — Il corpus: `data/input.txt` + `prepare_corpus.py`

**Cosa**: script che scarica Pinocchio da fonte di pubblico dominio, rimuove
header/footer editoriali, normalizza i caratteri problematici, salva
`data/input.txt` in UTF-8. Stampa statistiche: numero caratteri totali, vocabolario
risultante, top-20 caratteri per frequenza.

**Perché uno script e non un file scaricato a mano**: riproducibilità. Se tra tre
mesi il file si corrompe o vogliamo cambiare corpus, lo script documenta *esattamente*
come i dati sono stati prodotti. Nei progetti ML reali la provenienza dei dati è la
prima cosa che si perde e la più costosa da ricostruire.

**Perché la normalizzazione è importante (e cosa normalizziamo)**: i testi
digitalizzati contengono varianti tipografiche invisibili a occhio ma diverse per il
modello: apostrofo dritto `'` vs curvo `'`, trattini di tre lunghezze (`-`, `–`, `—`),
virgolette in quattro forme. Ogni variante è **un token in più nel vocabolario** e
diluisce le statistiche: se metà degli apostrofi è dritta e metà curva, il modello
deve imparare due volte la stessa regola ("dopo `l` può venire un apostrofo").
Normalizzare (una sola forma di apostrofo, una di trattino per i dialoghi, ecc.)
concentra i dati. Decisione da prendere qui e documentare: **teniamo le maiuscole**
(il modello imparerà che dopo il punto viene una maiuscola: altro progresso visibile)
e **teniamo gli a-capo** (imparerà i paragrafi).

**Verifica**: il vocabolario risultante deve stare sotto ~120 simboli e non
contenere caratteri "sorpresa" (li elenchiamo tutti a occhio nell'output dello script).

> ✅ Fatto. Fonte: **Gutenberg ebook #52484** (testo integrale; il #19517 era una
> lettura audio, scartato). Confini isolati con ancore testuali (`STORY_START`,
> `STORY_END`) invece di offset numerici. Rimossi 2 artefatti Gutenberg scoperti
> all'ispezione: **`[Illustrazione: …]`** (79 didascalie editoriali) e **`_…_`**
> (76 marcatori di corsivo). Trappola disinnescata: doppia traduzione dei
> fine-riga su Windows in cache → risolta scrivendo/leggendo in binario. Cifre
> `1 4 8` tenute (testo genuino: "avevano 14 anni"). Vocab finale: **69**.

### ☑ 0.3 — `CharTokenizer` (`tokenizer.py`)

**Cosa**: classe con vocabolario costruito dai caratteri unici del corpus (ordinati,
per determinismo), due mappe `stoi` (string→int) e `itos` (int→string), metodi
`encode(testo) -> list[int]` e `decode(indici) -> str`, proprietà `vocab_size`.

**Perché è meno banale di quanto sembri**: (a) l'ordinamento dei caratteri deve
essere deterministico, altrimenti due esecuzioni assegnano indici diversi e un
modello salvato diventa illeggibile — per questo il vocabolario si salva insieme al
modello; (b) `encode` deve fallire *rumorosamente* su caratteri fuori vocabolario
(un carattere mai visto non ha indice: meglio un errore chiaro subito che un
comportamento silenziosamente sbagliato dopo); (c) questa classe è il **contratto**
tra il mondo del testo e il mondo dei tensori — ogni singolo esperimento delle fasi
successive ci passa attraverso, quindi la testiamo subito e non la tocchiamo più.

**Verifica** (`test_tokenizer.py`): round-trip `decode(encode(s)) == s` su tutto il
corpus; `vocab_size` stabile tra esecuzioni; errore su carattere ignoto.

> ✅ Fatto. Aggiunti anche `save()`/`load()` (JSON) in anticipo: serviranno al
> checkpoint di Fase 7. 8 test verdi.

### ☑ 0.4 — Dataset e batching (`dataset.py`)

**Cosa**: caricare il corpus, codificarlo *una volta sola* in un array NumPy di
interi, spezzarlo in train (90%) e validation (10%), e scrivere
`get_batch(split, block_size, batch_size, rng)` che restituisce due array
`X, Y` di shape `(batch_size, block_size)`.

**Perché serve un validation set**: il modello vedrà il train set migliaia di volte
e potrebbe semplicemente *memorizzarlo*. La loss sul train non distingue
"ha imparato l'italiano" da "ha imparato Pinocchio a memoria". Il 10% tenuto
nascosto risponde alla domanda giusta: quanto è brava la rete su testo *dello
stesso tipo* ma *mai visto*? Quando la loss di train scende e quella di val sale,
il modello sta memorizzando: si chiama overfitting ed è la malattia numero uno del
machine learning. **Perché lo split è contiguo (ultimo 10%) e non caratteri a
caso**: il testo è sequenziale — se prendessimo caratteri sparsi, ogni carattere di
val avrebbe i vicini nel train e la "verifica su testo mai visto" sarebbe una
finzione.

**Perché X e Y hanno questa forma — il cuore della sottofase.** Un esempio di
training è una coppia (contesto, carattere-successivo-vero). Il trucco fondamentale:
da una finestra di `block_size+1` caratteri consecutivi si estraggono
**`block_size` esempi in un colpo solo**. Con la finestra `"ciao m"` (block_size=5):

```
X = "ciao "     Y = "iao m"
posizione 0:  visto "c"      → predici "i"
posizione 1:  visto "ci"     → predici "a"
posizione 2:  visto "cia"    → predici "o"
posizione 3:  visto "ciao"   → predici " "
posizione 4:  visto "ciao "  → predici "m"
```

`Y` è semplicemente `X` spostato di un carattere. Ogni posizione della finestra è
un esempio di training indipendente con contesto di lunghezza diversa. Questo è il
motivo per cui i transformer si addestrano così in fretta rispetto a quanto si
potrebbe pensare: una finestra da 256 caratteri = 256 predizioni corrette da
imparare simultaneamente, e (lo vedremo in Fase 5) la maschera causale permette di
calcolarle tutte in un solo passaggio.

**Perché i batch sono estratti a posizioni casuali** e non in ordine: la discesa
del gradiente stocastica funziona meglio con esempi il più possibile scorrelati tra
loro; batch consecutivi del libro sarebbero quasi identici e i gradienti
oscillerebbero seguendo la trama invece della lingua.

**Verifica** (`test_dataset.py`): shape corrette; proprietà `Y[i] == X[i+1]`
sull'array sorgente; train e val non si sovrappongono.

> ✅ **0.5 test** — Fatto. Niente pytest nell'ambiente → runner minimale
> (`tests/_runner.py` + `run_tests.py`). **16 test totali, tutti verdi.**
> Invariante di shift verificata come `X[:, 1:] == Y[:, :-1]`.

**Deliverable di fase**: possiamo trasformare Pinocchio in batch di tensori interi
pronti per qualsiasi modello. Nessun modello esiste ancora — ed è giusto così.
✅ **Raggiunto e verificato end-to-end.**

---

## ☑ Fase 1 — Bigram per conteggio (nessun training)  ✅ COMPLETATA (2026-07-18)

> **Esito**: `BigramCount` in `ronklm/models/bigram_count.py`, 6 test verdi (28
> totali). Numeri reali: **NLL uniforme 4.2341**, **NLL bigram train 2.3340 / val
> 2.3455 nats**, perplexity val **10.44**. Verifica qualitativa: "dopo `q` → `u`"
> al **93.3%**. Generazione: pseudo-italiano sillabico (doppie, vocali finali,
> spazi giusti) — il massimo per chi vede 1 carattere. Branch `v1.3.0`.


**Obiettivo didattico.** Costruire il language model più semplice che esista —
guarda solo l'ultimo carattere — e con esso i tre concetti che reggono tutto il
progetto: distribuzione sul prossimo token, campionamento, e **loss**. Senza
neanche un gradiente: solo conteggi. Serve a separare i concetti (cosa vuol dire
"modellare il linguaggio") dai meccanismi (come si addestra una rete): mischiarli
è il modo classico di non capire né gli uni né gli altri.

### ☑ 1.1 — La matrice dei conteggi

**Cosa**: `N`, matrice `(vocab_size, vocab_size)` di interi, dove `N[i, j]` = quante
volte, nel corpus di train, al carattere `i` segue il carattere `j`. Si costruisce
con una sola passata sul testo.

**Perché funziona come modello**: la riga `N[i]` è la fotografia empirica di "cosa
viene dopo `i`". Se dopo `q` c'è quasi sempre `u`, la riga di `q` avrà un picco su
`u`. Stiamo dicendo: *la miglior stima della probabilità del futuro è la frequenza
osservata nel passato*. È un'idea statistica antica (i modelli n-gram hanno
dominato il NLP per decenni) e vederla funzionare — e poi *fallire* per contesti
più lunghi — è esattamente la motivazione delle fasi successive: una tabella per
2 caratteri di contesto avrebbe `vocab²` righe, per 10 caratteri `vocab¹⁰` ≈ più
delle stelle nell'universo osservabile. **Il conteggio non scala; le reti neurali
sono il modo di *comprimere* questa tabella impossibile in una funzione con pochi
parametri.** Questa frase è metà del senso del deep learning.

### ☑ 1.2 — Da conteggi a probabilità (con smoothing)

**Cosa**: `P = (N + 1) / (N + 1).sum(axis=1, keepdims=True)` — ogni riga divisa per
la sua somma, dopo aver aggiunto 1 a tutti i conteggi.

**Perché normalizzare per riga**: una distribuzione di probabilità deve sommare a 1.
La riga `P[i]` risponde a "dato `i`, con che probabilità ciascun carattere?" — è
la prima incarnazione concreta della definizione di LM data in I.1.

**Perché il `+1` (smoothing di Laplace)**: se una coppia non appare mai nel train,
`P` le darebbe probabilità *zero*. Ma "mai visto nel campione" non significa
"impossibile": se poi quella coppia appare nel validation set, il modello le
assegnerebbe probabilità 0 → logaritmo di 0 → loss infinita → tutto rotto da un
singolo evento raro. Il +1 dice: "fingiamo di aver visto tutto almeno una volta".
È la prima apparizione di un tema eterno del ML: **mai fidarsi ciecamente dei dati
osservati; le stime vanno regolarizzate**. (Le reti neurali smussano da sole, per
come è fatta softmax — lo noteremo in Fase 2.)

**Dettaglio NumPy da capire bene**: `keepdims=True` fa sì che la somma per riga
abbia shape `(vocab, 1)` e non `(vocab,)`, così la divisione si propaga
correttamente riga per riga. È il primo incontro col **broadcasting**, la regola
con cui NumPy allinea shape diverse: va capito ora perché in Fase 3 dovremo
calcolarci i gradienti *attraverso* il broadcasting, ed è il punto tecnicamente più
insidioso dell'intero progetto.

### ☑ 1.3 — Campionamento: la prima generazione

**Cosa**: partire da un carattere, leggere la sua riga di `P`, estrarre il
successivo con `rng.choice(vocab_size, p=P[i])`, ripetere. Generare qualche
centinaio di caratteri.

**Perché si campiona invece di scegliere sempre il più probabile**: scegliendo
sempre il massimo (greedy), da uno stesso carattere uscirebbe sempre la stessa
catena → testo ciclico e degenere. Campionare secondo la distribuzione mantiene la
*varietà* del linguaggio: l'output sarà diverso a ogni run ma statisticamente
fedele al corpus. La tensione tra "probabile" e "vario" tornerà in Fase 7 con
temperature e top-k: qui ne vediamo la forma pura.

**Cosa aspettarsi**: pseudo-italiano sillabico — `"e po la co si il mant"` — con
già le doppie, le vocali finali, spazi a frequenza giusta. Per un modello che vede
1 carattere è il massimo teorico, e vederlo tara le aspettative per tutto il resto.

### ☑ 1.4 — La loss: negative log-likelihood (NLL)

**Cosa**: per ogni coppia consecutiva `(i → j)` del validation set, accumulare
`-log(P[i, j])`; riportare la media. Calcolare anche i due riferimenti: modello
uniforme (`-log(1/vocab) = log(vocab)`, ~4.6 nats con vocab≈100) e la NLL su train.

**Perché proprio questa formula — derivazione completa, da capire davvero.**
Vogliamo un numero che dica "quanto è buono il modello". Criterio naturale: un buon
modello assegna probabilità **alta al testo che è realmente accaduto**. La
probabilità dell'intero testo è il prodotto delle probabilità dei singoli passi
(regola della catena della probabilità):
`P(testo) = P(c₂|c₁) · P(c₃|c₂) · …` — vorremmo *massimizzarla*. Tre ritocchi la
rendono maneggevole:
1. **Logaritmo**: il prodotto di migliaia di numeri < 1 è un numero
   microscopico che i float non rappresentano (underflow). Il log trasforma il
   prodotto in *somma* di log-probabilità, numericamente stabile. E siccome il log
   è crescente, massimizzare il log equivale a massimizzare l'originale.
2. **Segno meno**: per convenzione si *minimizzano* le loss. `-log(p)` è perfetta:
   vale 0 se il modello era certo del carattere giusto (p=1), cresce all'infinito
   se gli aveva dato probabilità ~0. Punisce la sicurezza malriposta più di ogni
   altra cosa — che è esattamente ciò che vogliamo da un modello onesto.
3. **Media** (invece di somma): rende il numero indipendente dalla lunghezza del
   testo → confrontabile tra dataset e tra fasi.

Questa è la **cross-entropy**, la stessa identica loss con cui è addestrato GPT-4.
Da qui in poi ogni modello del progetto sarà giudicato da questo numero. Intuizione
da portarsi dietro: NLL media ≈ "sorpresa media per carattere"; e-elevato-alla-NLL ≈
"tra quanti caratteri il modello sta effettivamente esitando" (perplexity).

**Verifica**: NLL(bigram) nettamente sotto log(vocab) — se non lo fosse, c'è un bug
— e NLL(train) ≈ NLL(val), perché una tabella di bigrammi è troppo povera per
overfittare: prima osservazione sperimentale del rapporto capacità/overfitting.

### ☑ 1.5 — Creazione del `codebase_reference.md`

Prima stesura dell'atlante secondo i criteri delle istruzioni globali (indice
dove-sta-cosa, firme complete, tabelle, cosa NON esiste ancora). Da qui in poi si
aggiorna a ogni fase. *Perché nasce ora e non in Fase 0*: ora esiste il primo
modello funzionante — c'è qualcosa di sostanzioso da mappare.

**Deliverable di fase**: RonkLM v0 conta, genera e — soprattutto — **si misura**.
Concetti acquisiti: distribuzione sul prossimo token, campionamento, smoothing,
cross-entropy/NLL.

---

## ☐ Fase 2 — Bigram neurale (il gradiente a mano)

**Obiettivo didattico.** Ottenere *lo stesso identico risultato* della Fase 1 ma
per una strada opposta: invece di **contare**, **imparare per tentativi
corretti dal gradiente**. Siccome sappiamo già dove si deve arrivare (la matrice
`P` della Fase 1), ogni pezzo del meccanismo di training è verificabile contro una
verità nota. È il "hello world" della backpropagation: un solo strato, e ogni
derivata fatta a mano su carta prima che in codice. **Questa è la fase più
importante del progetto per la comprensione**: tutto ciò che viene dopo è questo
stesso ciclo, ripetuto su funzioni più ricche.

### ☐ 2.1 — Input one-hot e la matrice dei pesi `W`

**Cosa**: rappresentare il carattere `i` come vettore one-hot (tutti 0, un 1 in
posizione `i`); il modello è una sola matrice di pesi `W (vocab, vocab)`
inizializzata con piccoli numeri gaussiani casuali; i logits sono `x_onehot @ W`.

**Perché one-hot**: è il modo più onesto di dare un simbolo discreto a una macchina
che fa solo aritmetica, *senza introdurre ordinamenti fasulli*: dare al modello
l'indice grezzo (12 per `m`, 13 per `n`) suggerirebbe che `n` = `m`+1 in qualche
senso numerico — falso e dannoso. One-hot rende tutti i caratteri equidistanti.

**Osservazione che pagherà in Fase 4**: `one_hot(i) @ W` non fa alcun vero calcolo —
*seleziona la riga i-esima di `W`*. Quindi "moltiplicare un one-hot per una
matrice" e "usare la riga i-esima come rappresentazione del carattere i" sono la
stessa cosa. Quando in Fase 4 introdurremo gli **embedding**, non saranno un'idea
nuova: saranno questa stessa selezione di riga, implementata efficientemente e con
righe più corte del vocabolario.

**Perché inizializzazione casuale piccola e non zeri**: con `W=0` tutti i logits
sono uguali → il modello parte dalla distribuzione uniforme, il che va anche bene
qui; ma nelle reti a più strati pesi tutti uguali producono neuroni che ricevono
gradienti identici e non si differenziano mai ("symmetry breaking"). Prendiamo
subito l'abitudine giusta. Piccoli, perché logits grandi a caso = modello
inizialmente *sicurissimo e a caso* = loss iniziale enorme e primi passi violenti.
Sanity check che faremo sempre: **alla partenza la loss deve valere ≈ log(vocab)**
(il modello non sa nulla → deve essere ≈ uniforme). Se non lo è, l'inizializzazione
è sbagliata. Questo controllo da 30 secondi cattura una quantità sorprendente di bug.

### ☐ 2.2 — Softmax: da logits a probabilità

**Cosa**: `softmax(z)_j = exp(z_j) / Σ_k exp(z_k)`, implementata con il trucco di
stabilità `z - max(z)`.

**Perché serve e perché proprio lei**: la rete produce logits — reali qualsiasi —
ma la loss della Fase 1 vuole probabilità. Softmax è il ponte, e ha esattamente le
proprietà giuste: (a) `exp` rende tutto positivo; (b) la divisione per la somma
normalizza a 1; (c) è *derivabile ovunque* — indispensabile, perché il gradiente
dovrà attraversarla; (d) preserva l'ordine dei logits ed è invariante per
traslazione (aggiungere una costante a tutti i logits non cambia nulla — le
*differenze* tra logits contano, non i valori assoluti); (e) il rapporto tra due
probabilità dipende esponenzialmente dalla differenza dei logits: pochi punti di
logit = dominio quasi totale. Il nome viene da qui: è una versione "morbida"
(soft) e derivabile della funzione argmax.

**Perché il trucco `z - max(z)`**: `exp(800)` supera il massimo float64 → `inf` →
`NaN` a cascata. Per l'invarianza per traslazione appena detta, sottrarre il
massimo non cambia *matematicamente* il risultato, ma porta il logit più grande a
0 e rende ogni `exp` calcolabile. Prima lezione di una verità permanente: **la
matematica su carta e la matematica in float sono due discipline diverse**, e i
NaN d'addestramento nascono quasi sempre in punti come questo.

### ☐ 2.3 — La derivazione del gradiente (su carta, poi in codice)

**Cosa**: derivare a mano `∂L/∂W` per la cross-entropy su softmax, documentando la
derivazione completa passo-passo nel `codebase_reference.md`, e implementarla.

**Perché farlo a mano una volta nella vita**: il risultato finale è di una
semplicità sconcertante —

```
∂L/∂logits = probs − y_onehot
```

cioè: il gradiente sui logits è *la probabilità che il modello ha dato a ogni
carattere, meno 1 sul carattere giusto*. Se il modello dava 0.9 al carattere vero,
il gradiente è piccolo (0.9−1 = −0.1): quasi niente da correggere. Se gli dava
0.01, il gradiente è −0.99: strattone violento. **La correzione è proporzionale
all'errore, automaticamente.** Poi `∂L/∂W = xᵀ @ (probs − y)`: per input one-hot
significa che si aggiorna *solo la riga del carattere visto* — il che ha
perfettamente senso: vedere `q→u` non insegna nulla su cosa segue la `z`.
Sapere che dentro `loss.backward()` di ogni framework, per l'ultimo strato di ogni
LLM, c'è *esattamente questa sottrazione*, toglie la magia dal training una volta
per tutte. La derivazione richiede la regola della catena e la derivata di softmax
(due casi: j uguale o diverso dalla classe vera) — la faremo per esteso nel
reference, è mezz'ora di algebra e ripaga per sempre.

### ☐ 2.4 — Il training loop: la discesa del gradiente

**Cosa**: il ciclo canonico, scritto esplicito:

```
per step in range(n_steps):
    X, Y = get_batch(...)          # 1. dati
    probs = forward(X)             # 2. previsione
    loss  = cross_entropy(probs,Y) # 3. quanto sbaglio?
    dW    = backward(...)          # 4. in che direzione correggere ogni peso?
    W    -= lr * dW                # 5. piccolo passo in quella direzione
```

**Perché questo ciclo è "il deep learning"**: dalla regressione logistica a GPT-4,
*tutto* il campo è questo ciclo; cambia solo cosa c'è dentro `forward`. Le fasi
4–7 non toccheranno mai più questi 5 passi: arricchiranno solo il punto 2.

**Perché il learning rate (`lr`) e perché "piccolo passo"**: il gradiente è
un'informazione *locale* — dice la direzione di discesa *nel punto in cui siamo*,
come la pendenza sotto i piedi nella nebbia. Un passo enorme in quella direzione
può scavalcare la valle e finire più in alto di prima (loss che oscilla o
diverge); un passo microscopico impiega ere. Il `lr` è il compromesso, ed è
l'iperparametro più importante del machine learning: sbagliarlo di un fattore 10
distrugge qualsiasi training. Qui lo scopriremo *sperimentalmente*: proveremo
lr troppo alto, troppo basso e giusto, e guarderemo le tre curve di loss. (In
Fase 8 impareremo a variarlo durante il training, e in Fase 4 AdamW lo adatterà
peso per peso.)

**Perché su batch e non su tutto il dataset**: il gradiente su *tutto* il corpus
sarebbe il più accurato, ma costa una passata intera per un solo passo. Il
gradiente su un batch casuale è una *stima rumorosa* di quello vero — e va bene
così: mille passi rumorosi ma economici battono un passo perfetto e costosissimo.
(Il rumore ha perfino effetti benefici sulla generalizzazione.) Si chiama
**Stochastic Gradient Descent**, la S è il batch casuale.

### ☐ 2.5 — La verifica di convergenza: neurale ≡ conteggi

**Cosa**: confrontare (a) NLL finale del bigram neurale vs quella del bigram
contato; (b) `softmax(W)` riga per riga vs la `P` della Fase 1.

**Perché è la verifica perfetta**: per questo problema, la soluzione che minimizza
la cross-entropy È la distribuzione empirica dei conteggi (risultato standard di
maximum likelihood). Quindi il training, partito da pesi casuali, deve
*riscoprire da solo* la tabella della Fase 1, per pura discesa del gradiente. Le
NLL devono coincidere a meno di rumore (~±0.01), e le righe di `softmax(W)`
somigliare alle righe di `P`. Se accade, abbiamo la prova sperimentale che
l'intero meccanismo forward→loss→backward→update funziona. Poche verifiche in
tutto il progetto sono così nette.

### ☐ 2.6 — Il gradient check numerico

**Cosa**: per un piccolo sottoinsieme di pesi, stimare il gradiente "alla bruta"
con le differenze finite centrali — `(L(w+h) − L(w−h)) / 2h` con `h ≈ 1e-5` — e
confrontarlo col nostro gradiente analitico tramite errore relativo (< 1e-6:
ottimo; > 1e-3: bug quasi certo). In `tests/test_models.py`.

**Perché è il test più importante che si possa scrivere**: un backward sbagliato è
il bug più subdolo del deep learning, perché **non rompe niente di visibile** — il
training parte, la loss magari perfino scende (male), e si perdono giornate a
incolpare gli iperparametri. Il gradient check usa *solo il forward* (che è facile
da scrivere giusto) per verificare il backward (che è facile sbagliare): due
strade indipendenti che devono dare lo stesso numero. **Perché la differenza
centrale** e non la semplice `(L(w+h)−L(w))/h`: l'espansione di Taylor mostra che
l'errore della versione centrale va come h² invece che come h — enormemente più
precisa a parità di h. **Perché non si usa per addestrare**: richiede 2 forward
*per ogni singolo peso* — per 1M di parametri, 2M di forward per un solo passo.
La backpropagation dà lo stesso risultato al costo di ~2 forward *in totale*: è
questa efficienza, non altro, ad aver reso possibile il deep learning. Qui questo
test è artigianale; in Fase 3 diventerà sistematico su ogni operazione.

**Deliverable di fase**: primo modello *addestrato* del progetto, con gradiente
derivato a mano, convergenza dimostrata contro una verità nota e gradient check.
Il ciclo forward→loss→backward→update non è più una frase: è codice che hai visto
convergere.

---

## ☐ Fase 3 — `ronkgrad`: il micro-motore di autograd

**Obiettivo didattico.** In Fase 2 la derivazione a mano era mezz'ora di algebra
per **un** layer. Il GPT ne avrà decine, annidati. Derivare a mano non scala —
serve automatizzare la regola della catena. Costruiamo `ronkgrad`: un motore di
**differenziazione automatica** tensoriale (~250 righe), concettualmente identico
al cuore di PyTorch. Capito questo, PyTorch non avrà più segreti strutturali.
(Riferimento spirituale: micrograd di Karpathy, che però è scalare; il nostro è
tensoriale, e la differenza — il broadcasting — è il vero contenuto della fase.)

### ☐ 3.1 — L'idea: il grafo computazionale

**Cosa**: classe `Tensor` che avvolge un `np.ndarray` (`.data`) e in più ricorda:
`.grad` (l'accumulatore del gradiente), i tensori da cui è stato prodotto
(`._parents`) e una piccola funzione `._backward` che sa propagare il gradiente
dai figli ai genitori.

**Perché questa architettura — l'intuizione completa.** Qualsiasi calcolo, anche
`loss = mean((x @ W1).tanh() @ W2 − y)²`, è una catena di operazioni *elementari*
(matmul, tanh, sottrazione, quadrato, media). Mentre il forward le esegue,
possiamo *registrare chi ha prodotto cosa*: ne risulta un grafo (aciclico e
diretto) dove i nodi sono tensori e gli archi dicono "prodotto da". La regola
della catena del calcolo differenziale dice che la derivata di una composizione è
il prodotto delle derivate dei pezzi: quindi, se ogni operazione elementare sa
calcolare il *proprio* pezzettino di derivata, il gradiente della loss rispetto a
**qualsiasi** tensore del grafo si ottiene camminando il grafo all'indietro e
moltiplicando/accumulando i pezzi. Nessuno deve mai derivare la formula composta:
la composizione è automatica. Questo — costruire il grafo durante il forward,
percorrerlo a ritroso nel backward — è *esattamente* ciò che fa PyTorch quando
chiami `loss.backward()`; la differenza è ingegneria (C++, GPU, fusione di
kernel), non concetto.

**Perché `.grad` si *accumula* (`+=`) invece di assegnarsi**: un tensore può
essere usato in più punti del grafo (es. la stessa `W` usata due volte, o un
tensore che si dirama). Il gradiente totale è la *somma* dei contributi da ogni
uso (è la regola della derivata totale). Assegnare invece di sommare
sovrascriverebbe il primo contributo — uno dei bug classici di chi scrive
un autograd. Corollario che ogni utente PyTorch conosce: prima di ogni backward,
i gradienti vanno azzerati (`zero_grad`), altrimenti si sommano a quelli del passo
precedente.

### ☐ 3.2 — Le operazioni di base e i loro backward

**Cosa**: implementare, con forward e backward: `+`, `-`, `*` (elemento per
elemento), `@` (matmul), `sum`, `mean`, e la meccanica del broadcasting.

**Perché ogni backward è breve ma va capito, uno per uno.** Il pattern è sempre:
*ricevo `out.grad` (quanto la loss dipende dal mio output) e devo dire quanto la
loss dipende dai miei input*. I tre casi fondamentali:

- **Somma** `c = a + b`: la derivata passa invariata a entrambi
  (`a.grad += out.grad`, `b.grad += out.grad`). La somma è un "distributore di
  gradiente". Ricordare questo renderà ovvie le *residual connections* in Fase 6 —
  sono importanti *precisamente perché* la somma distribuisce il gradiente intatto.
- **Prodotto** `c = a * b`: ognuno riceve il gradiente moltiplicato per *l'altro*
  (`a.grad += b.data * out.grad` e viceversa): quanto conta un fattore dipende da
  quanto vale l'altro.
- **Matmul** `C = A @ B`: le formule sono `A.grad += out.grad @ Bᵀ` e
  `B.grad += Aᵀ @ out.grad`. Le *deriveremo* nel reference (bastano gli indici
  delle somme), ma c'è un controllo mnemonico che vale oro: **le shape devono
  tornare** — `A.grad` deve avere la shape di `A`, e c'è un solo modo di
  combinare `out.grad`, `Aᵀ`, `Bᵀ` perché i conti delle dimensioni quadrino.
  Metà dei bug di backward si trovano solo guardando le shape.

**Il broadcasting nel backward — il punto più insidioso del progetto, spiegato.**
Nel forward, NumPy permette `(32, 100) + (100,)`: il vettore viene *virtualmente
copiato* su 32 righe. Ma se un valore è stato usato 32 volte, nel backward deve
ricevere 32 contributi di gradiente: quindi il gradiente di un tensore
broadcastato si ottiene **sommando `out.grad` lungo le dimensioni che erano state
espanse**, per tornare alla shape originale. Dimenticarlo produce gradienti di
shape sbagliata (crash: bug fortunato) o silenziosamente errati (bug tragico).
Scriveremo una funzione di servizio `unbroadcast(grad, shape)` usata da tutti i
backward, e la tortureremo nei test. Questo è il motivo per cui il nostro motore è
"tensoriale, non scalare": micrograd non ha questo problema perché non ha shape;
noi sì, ed è il pezzo di comprensione in più che ci portiamo a casa.

### ☐ 3.3 — Le non-linearità e le funzioni composte

**Cosa**: `tanh`, `relu`, `gelu`, `exp`, `log`, e le composte di uso costante:
`softmax` e `cross_entropy` come operazioni dedicate.

**Perché le non-linearità sono *obbligatorie* in una rete** (non un dettaglio): la
composizione di sole operazioni lineari è ancora lineare — dieci matmul in fila
equivalgono matematicamente a *una* matmul. Senza una funzione non lineare in
mezzo, una rete profonda ha esattamente la stessa espressività di un singolo
strato: tutta la profondità sarebbe finta. La non-linearità è ciò che permette
alla rete di rappresentare funzioni arbitrariamente complicate (teorema di
approssimazione universale). Implementiamo: `tanh` (classica, comprimere in
[−1,1], derivata elegante `1 − tanh²`); `relu` (`max(0, x)`, lo standard moderno:
gradiente 1 o 0, semplicissima e per questo amatissima); `gelu` (la variante
smussata di relu usata dai GPT reali — la useremo nel transformer per fedeltà,
spiegandola come "relu probabilistica").

**Perché `cross_entropy` come operazione unica e non composta da
log+softmax+indicizzazione**: due ragioni. *Stabilità*: la composizione ingenua
`log(softmax(z))` può produrre `log(0) = −inf`; la forma fusa (log-sum-exp con
sottrazione del massimo) è stabile per costruzione. *Efficienza e semplicità del
gradiente*: come scoperto in Fase 2, il gradiente della coppia
softmax+cross-entropy è la sottrazione `probs − y`, molto più semplice dei due
gradienti separati moltiplicati. Anche PyTorch la fonde
(`F.cross_entropy`) per gli stessi identici motivi. La nostra Fase 2 è stata,
retroattivamente, la derivazione di questa operazione.

### ☐ 3.4 — `backward()`: l'ordinamento topologico

**Cosa**: il metodo `loss.backward()` che: (1) costruisce l'ordine topologico del
grafo con una DFS; (2) inizializza `loss.grad = 1`; (3) chiama i `._backward` dei
nodi in ordine inverso.

**Perché serve l'ordine topologico**: il backward di un nodo può eseguirsi solo
quando il suo `out.grad` è *completo* — cioè quando tutti i nodi a valle che lo
usano hanno già contribuito. L'ordinamento topologico (ogni nodo dopo tutti i suoi
discendenti, poi si percorre al contrario) garantisce esattamente questo. Con un
ordine sbagliato, un nodo propagherebbe un gradiente parziale: risultati
silenziosamente errati, di nuovo. **Perché `loss.grad = 1`**: la derivata della
loss rispetto a sé stessa è 1 — è il seme da cui la catena parte.

### ☐ 3.5 — La batteria di test (`test_autograd.py`)

**Cosa**: per **ogni** operazione del motore, un gradient check numerico (Fase 2.6,
ora sistematizzato in una utility riusabile), con casi mirati sul broadcasting
(shape `(3,1)+(1,4)`, bias su batch, ecc.), sull'uso ripetuto dello stesso tensore,
e su composizioni tipo mini-MLP.

**Perché la fase non è finita finché questi test non passano tutti**: da qui in
avanti, *ogni* modello del progetto si fiderà ciecamente di `ronkgrad`. Un bug qui
avvelenerebbe silenziosamente le Fasi 4–8, che sono esattamente le fasi in cui i
sintomi ("il modello impara poco") hanno mille altre spiegazioni possibili. La
regola non negoziabile II.2.1 nasce qui: nessun gradiente non testato. In cambio,
d'ora in poi scrivere un modello nuovo richiederà *solo il forward* — il backward
sarà gratis e già verificato. Questo cambio di produttività è lo stesso che i
framework hanno regalato al mondo della ricerca; ce lo saremo guadagnato da soli.

**Deliverable di fase**: `y = (x @ W1).tanh() @ W2; loss.backward()` — e i
gradienti compaiono, giusti, testati. Possediamo il nostro mini-PyTorch e sappiamo
esattamente cosa fa ogni sua riga.

---

## ☐ Fase 4 — MLP language model

**Obiettivo didattico.** Rompere finalmente il limite del bigram: un contesto di
**più caratteri**. Introduciamo i due mattoni che i transformer usano ovunque —
**embedding** e **strati nascosti** — e la pratica del training vero: minibatch,
curva train/val, overfitting, AdamW. Architettura di riferimento: il language
model di Bengio et al. 2003, il paper che ha dato inizio ai LM neurali.

### ☐ 4.1 — La libreria di layer (`nn.py`)

**Cosa**: sopra `ronkgrad`, i mattoni riusabili: classe base `Module` (che sa
elencare i propri parametri, ricorsivamente), `Linear(n_in, n_out)`,
`Embedding(vocab, n_embd)`, `Sequential`.

**Perché l'astrazione `Module`**: il GPT finale avrà decine di sotto-componenti
annidati, ognuno con i suoi pesi. L'ottimizzatore ha bisogno della *lista completa*
dei parametri; raccoglierla a mano è il modo di dimenticarne uno (che quindi non si
addestra mai — bug silenzioso, di nuovo). `Module.parameters()` ricorsivo risolve
il problema una volta per tutte. È esattamente il ruolo di `nn.Module` in PyTorch.

**Perché l'inizializzazione dei pesi merita attenzione** (e non è un dettaglio):
i pesi di `Linear` vanno scalati come `1/sqrt(n_in)`. Motivo: l'output di un
neurone è la somma di `n_in` termini; se ogni peso ha varianza fissa, la varianza
della somma cresce linearmente con `n_in` → con 500 input, attivazioni ~22 volte
più larghe del dovuto → tanh satura ai suoi estremi, dove la derivata è ~0 → il
gradiente muore al primo passaggio. Con lo scaling `1/sqrt(n_in)` la varianza
dell'output resta ~1 indipendentemente dalla larghezza. Vederemo l'effetto con un
esperimento: istogramma delle attivazioni con e senza scaling corretto.

### ☐ 4.2 — L'architettura MLP (`mlp.py`)

**Cosa**: contesto di `block_size` caratteri (partiamo con 8) →
`Embedding(vocab, n_embd)` per ciascuno → concatenazione dei vettori →
`Linear` + `tanh` (strato nascosto) → `Linear` finale verso i logits del vocab.

**Perché gli embedding, spiegato fino in fondo.** In Fase 2 abbiamo visto che
one-hot @ matrice = selezione di riga. L'`Embedding(vocab, n_embd)` rende la cosa
ufficiale: una tabella dove **ogni carattere possiede un vettore denso di
`n_embd` numeri, e quei numeri sono parametri addestrabili**. Due conseguenze
profonde: (1) *compressione* — 100 caratteri descritti da vettori di 24 numeri
invece che da vettori di 100; (2) *geometria della somiglianza* — il training è
libero di avvicinare tra loro i vettori di caratteri che si comportano in modo
simile (le vocali, le cifre…), così ciò che il modello impara su `a` si trasferisce
in parte a `e`. Il conteggio della Fase 1 non poteva farlo per costruzione: ogni
riga della tabella era indipendente. **Questa è la risposta alla domanda lasciata
aperta in 1.1**: come si comprime la tabella impossibile da `vocab^contesto` righe?
Rappresentando i simboli come vettori e *calcolando* sulle rappresentazioni.
(Nei LLM veri: "king − man + woman ≈ queen" è la stessa idea a scala parola.)

**Perché lo strato nascosto**: la concatenazione degli embedding da sola,
proiettata linearmente sui logits, potrebbe solo *sommare contributi indipendenti
per posizione* (di nuovo un modello quasi-tabellare). Lo strato nascosto con
non-linearità permette di rilevare **combinazioni**: "c'è una `q` in penultima
posizione E una `u` in ultima" — feature composite che nessuna tabella lineare
esprime. È qui che la rete smette di essere una tabella compressa e comincia a
*calcolare*.

**Il limite strutturale, dichiarato subito** (motiverà la Fase 5): il contesto è
**rigido**. 8 caratteri, sempre, tutti concatenati in posizioni fisse: la rete deve
imparare *da capo* per ogni posizione come usare l'informazione lì contenuta
("cosa fare se in terzultima posizione c'è una vocale" non si trasferisce alla
quartultima); allargare il contesto fa crescere linearmente i pesi del primo
strato; e caratteri lontani devono comunque passare tutti dallo stesso collo di
bottiglia. Il transformer nasce per rompere esattamente questa rigidità.

### ☐ 4.3 — Il training vero: minibatch, train/val, overfitting

**Cosa**: training con `ronkgrad`, batch da 32–64, valutazione periodica della
loss su train E val, grafico delle due curve.

**Perché ora l'overfitting diventa reale**: a differenza del bigram (2.5: troppo
povero per overfittare), l'MLP ha decine di migliaia di parametri e *può*
memorizzare pezzi di Pinocchio. Vedremo per la prima volta le due curve separarsi:
train che continua a scendere, val che rallenta e si ferma (o risale). Discussione
esperienziale, sui nostri numeri, dei tre rimedi classici: più dati, modello più
piccolo, regolarizzazione (il weight decay arriva in 4.4). La regola pratica da
interiorizzare: **la sola loss che conta è quella di validazione** — quella di
train si può sempre abbassare barando (memorizzando).

### ☐ 4.4 — AdamW (`optim.py`)

**Cosa**: implementare `SGD` come classe pulita, e poi `AdamW` completo:
momento del primo ordine (media mobile dei gradienti), momento del secondo ordine
(media mobile dei gradienti al quadrato), bias-correction, weight decay
*disaccoppiato*.

**Perché SGD puro non basta, e cosa aggiunge Adam — pezzo per pezzo:**
- **Momentum** (media mobile dei gradienti): i gradienti da minibatch sono rumorosi
  (2.4); mediarli nel tempo filtra il rumore e accumula velocità nelle direzioni
  costanti — come una palla che rotola invece di un escursionista che riparte da
  fermo a ogni passo.
- **Secondo momento / scaling adattivo**: parametri diversi ricevono gradienti su
  scale diversissime (l'embedding di una lettera rara riceve segnale raramente; i
  bias dell'ultimo layer, sempre). Un unico lr per tutti è per forza sbagliato per
  qualcuno. Adam divide il passo di ogni peso per la radice della media mobile dei
  suoi gradienti² → ogni peso ha, di fatto, il suo learning rate auto-tarato.
- **Bias-correction**: le due medie mobili partono da 0 → nei primi step sono
  sottostimate → senza correzione, i primi passi sarebbero distorti. La correzione
  `/(1−β^t)` compensa esattamente questo transitorio.
- **Weight decay disaccoppiato** (la W di AdamW): il weight decay spinge tutti i
  pesi dolcemente verso 0 a ogni passo — regolarizzazione: pesi piccoli =
  funzioni più semplici = meno overfitting (rimedio promesso in 4.3). Il punto
  sottile: in Adam "classico" il decay finiva dentro il gradiente e veniva quindi
  ri-scalato dal meccanismo adattivo, indebolendolo proprio dove serviva; AdamW lo
  applica *fuori* dal meccanismo adattivo, direttamente ai pesi. Per questo
  l'industria usa AdamW e non Adam.

Averlo scritto a mano significa che quando leggerai `torch.optim.AdamW(params,
lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)` in un repo vero, ogni argomento
sarà un numero di cui conosci il meccanismo dall'interno.

### ☐ 4.5 — Confronto sperimentale e generazione

**Cosa**: tabella NLL val — bigram contato / bigram neurale / MLP (che deve vincere
nettamente); campioni di testo generato a confronto; mini-studio dell'effetto di
`block_size` (2 vs 8) e `n_embd`.

**Perché il confronto è il punto**: è la prima volta nel progetto che *aggiungere
capacità al modello* compra qualità misurabile. La tabella delle NLL diventa la
spina dorsale sperimentale del progetto e ogni fase successiva vi aggiungerà una
riga. Nel testo generato la differenza si deve *vedere*: parole quasi-italiane più
lunghe e stabili, doppie giuste, morfologia abbozzata.

**Deliverable di fase**: un LM neurale vero, addestrato con l'ottimizzatore
dell'industria scritto da noi, misurabilmente migliore del bigram, con overfitting
osservato e discusso. Concetti acquisiti: embedding, strato nascosto,
inizializzazione, minibatch, train/val, AdamW.

---

## ☐ Fase 5 — Self-attention (una testa, causale)

**Obiettivo didattico.** Il cuore del transformer, costruito da zero e capito riga
per riga. L'idea in una frase: invece di un contesto rigido e concatenato (MLP),
**ogni posizione della sequenza decide da sola, dinamicamente, a quali posizioni
precedenti prestare attenzione e quanto**, con pesi di attenzione *calcolati dai
dati stessi* a ogni forward.

### ☐ 5.1 — L'intuizione Q/K/V prima del codice

**Cosa**: sezione scritta (qui e nel reference) + le tre proiezioni lineari
`query`, `key`, `value` in `attention.py`.

**Perché tre proiezioni e cosa significano — la metafora onesta.** Ogni posizione
della sequenza emette tre vettori, ottenuti dallo stesso embedding con tre matrici
diverse (tutte apprese):
- **query** = "che cosa sto cercando" (es., implicitamente: *sono una vocale
  finale, cerco l'ultima consonante forte*);
- **key** = "che cosa offro, come mi faccio trovare" (es.: *sono una `r` a inizio
  sillaba*);
- **value** = "l'informazione che consegno *se* qualcuno mi sceglie" (che può
  essere diversa da come mi faccio trovare: chiave e contenuto sono ruoli diversi).

L'affinità tra la query della posizione `t` e la key della posizione `s` (un
prodotto scalare: alto = compatibili) decide quanto `t` attinge dal value di `s`.
Tutto — cosa cercare, come farsi trovare, cosa consegnare — è *appreso dal
gradiente*: noi forniamo solo il meccanismo. **Perché è la soluzione ai limiti
dell'MLP (4.2)**: i pesi di attenzione sono ricalcolati a ogni input (contesto
*dinamico*, non rigido), le stesse matrici Q/K/V servono per tutte le posizioni
(ciò che si impara si trasferisce ovunque nella sequenza), e una posizione lontana
è raggiungibile in un passo, non attraverso un collo di bottiglia.

### ☐ 5.2 — I punteggi e lo scaling `1/√d`

**Cosa**: `scores = q @ kᵀ / sqrt(head_size)` — shape `(B, T, T)`: per ogni
elemento del batch, una matrice T×T dove la cella `(t, s)` è l'affinità
posizione-t→posizione-s.

**Perché lo scaling, derivato per bene** (è la domanda d'esame classica): il
prodotto scalare di due vettori di dimensione `d` con componenti indipendenti a
varianza 1 ha varianza `d` — i punteggi crescono con la radice della dimensione
della testa per pura statistica, non perché le affinità siano più nette. Punteggi
grandi in ingresso a una softmax la **saturano**: quasi tutta la probabilità su
una posizione, quasi zero altrove; e nelle zone sature la derivata della softmax è
~0 → **il gradiente non passa** → le matrici Q/K smettono di imparare proprio
all'inizio, quando dovrebbero imparare di più. Dividere per `√d` riporta la
varianza dei punteggi a ~1 e la softmax nella sua zona viva. Una singola costante,
messa lì per far fluire il gradiente: il transformer è pieno di scelte così, e
questa è la più pulita da capire fino in fondo.

### ☐ 5.3 — La maschera causale

**Cosa**: prima della softmax, porre a `−inf` tutte le celle `(t, s)` con `s > t`
(matrice triangolare); dopo la softmax quelle celle valgono esattamente 0.

**Perché è *necessaria* e non un'opzione**: il nostro compito è predire il
carattere successivo. Se la posizione `t` potesse attingere dalle posizioni
future, durante il training vedrebbe *la risposta* dentro l'input: loss
bassissima, modello inutilizzabile in generazione (dove il futuro non esiste
ancora). La maschera impone la freccia del tempo. **Perché `−inf` prima della
softmax** invece di azzerare dopo: `exp(−inf) = 0` e la normalizzazione della
softmax si ridistribuisce automaticamente e correttamente sulle sole posizioni
lecite — azzerare *dopo* lascerebbe le righe con somma < 1 (non più una
distribuzione di probabilità). **Il legame con 0.4**: è questa maschera che
permette di addestrare *tutte* le T posizioni della finestra in un solo forward,
ciascuna col proprio contesto legale — il trucco "T esempi al prezzo di uno"
promesso allora. ("Decoder-only" nel gergo = un transformer fatto solo di blocchi
con questa maschera; i modelli GPT sono tutti così.)

### ☐ 5.4 — Softmax e aggregazione dei value

**Cosa**: `att = softmax(scores, axis=-1)` → `out = att @ v`, shape `(B, T, head_size)`.

**Perché l'output è una *media pesata* e cosa significa**: ogni riga di `att` è
una distribuzione di probabilità sulle posizioni passate (somma 1); `att @ v` è
quindi, per ogni posizione, una **miscela convessa dei value del passato**, dosata
dalle affinità. La softmax qui non produce una "probabilità di essere giusto" come
in Fase 2, ma un **meccanismo di indirizzamento morbido e derivabile**: un po' come
leggere da una memoria dove invece di scegliere *una* cella si legge un mix pesato
di tutte — ed è proprio la morbidezza a rendere il meccanismo addestrabile per
gradiente (un indirizzamento rigido non sarebbe derivabile). Verifica meccanica in
questa sottofase: inseguire le shape dell'intera pipeline a mano,
`(B,T,C) → q,k,v (B,T,H) → scores (B,T,T) → att (B,T,T) → out (B,T,H)`, e
controllare che le righe di `att` sommino a 1 e rispettino la causalità.

### ☐ 5.5 — Esperimento e ispezione

**Cosa**: (a) mini-LM con la sola testa di attention + proiezione ai logits,
confrontato con l'MLP a parità di budget; (b) **visualizzazione delle matrici di
attenzione** su frasi vere del corpus (heatmap T×T).

**Perché ispezionare `att` è irrinunciabile**: è una delle poche finestre *dirette*
sull'interno di una rete: si vede — letteralmente, in un'immagine — a cosa guarda
il modello. Su testo italiano ci si aspetta di veder emergere teste che guardano
il carattere precedente, gli inizi di parola (dopo uno spazio), le vocali
precedenti. Rende concreto tutto il discorso Q/K/V come nessuna formula può fare.

**Deliverable di fase**: una testa di self-attention causale, funzionante,
verificata da `ronkgrad` (backward gratis e già testato — il dividendo della
Fase 3), ispezionata visivamente. Concetti acquisiti: Q/K/V, scaling, causalità,
attenzione come media pesata derivabile.

---

## ☐ Fase 6 — Il blocco Transformer

**Obiettivo didattico.** Una testa di attention da sola non basta: serve il
**blocco** — l'unità che i GPT ripetono N volte — e soprattutto serve capire i tre
componenti che *rendono possibile* impilare blocchi in profondità: multi-head,
residual connections, LayerNorm. Questa fase è dove si impara *perché le reti
profonde sono addestrabili* — che non è affatto ovvio.

### ☐ 6.1 — Multi-head attention

**Cosa**: `MultiHeadAttention`: `n_head` teste indipendenti (ognuna con le sue
Q/K/V, dimensione `n_embd / n_head`), output concatenati e riproiettati con una
`Linear` finale.

**Perché più teste invece di una testa grande — il ragionamento**: una singola
softmax produce *una* distribuzione di attenzione per posizione: un solo
"sguardo". Ma a una posizione possono servire *contemporaneamente* informazioni
diverse da posti diversi: il carattere precedente (per l'ortografia), l'inizio
della parola (per la morfologia), l'apertura di virgolette molto indietro (per
chiudere un dialogo). Teste separate = sguardi paralleli specializzabili
indipendentemente. Dividere `n_embd` per `n_head` (invece di moltiplicare i
parametri) mantiene il costo costante: la scelta standard dell'industria — a
parità di budget, più sguardi piccoli battono un solo sguardo grande. La `Linear`
finale dopo la concatenazione serve a *mescolare* i contributi delle teste, che
altrimenti resterebbero segregati in fette separate del vettore.

### ☐ 6.2 — Feed-forward network (FFN)

**Cosa**: `Linear(n_embd → 4·n_embd)` + `gelu` + `Linear(4·n_embd → n_embd)`,
applicata a ogni posizione indipendentemente.

**Perché serve, dopo l'attention — la divisione dei ruoli**: l'attention *sposta*
informazione tra posizioni ma la elabora poco (in fondo fa medie pesate). La FFN è
il suo complemento esatto: non guarda nessun'altra posizione, ma *elabora* — con
una vera non-linearità — ciò che l'attention ha raccolto. Il ritmo del transformer
è: comunica (attention) → pensa (FFN) → comunica → pensa… **Perché 4×**: il
fattore di espansione dà alla FFN uno spazio interno più largo dove computare
prima di ricomprimere; il valore 4 è la convenzione empirica di tutti i GPT (e nei
modelli reali la FFN contiene ~2/3 dei parametri totali — è lì che si ritiene
risieda gran parte della "conoscenza" memorizzata).

### ☐ 6.3 — LayerNorm, implementata a mano

**Cosa**: `LayerNorm(n_embd)`: per ogni posizione, normalizza il suo vettore di
feature a media 0 e varianza 1, poi riscala con due parametri appresi
`gamma` (guadagno) e `beta` (offset).

**Perché normalizzare**: in una rete profonda, la scala delle attivazioni di uno
strato dipende da tutti gli strati precedenti — che *stanno cambiando* durante il
training. Ogni strato insegue un bersaglio mobile e le scale possono
esplodere/collassare strada facendo. LayerNorm ristabilisce a ogni blocco un punto
di riferimento fisso (media 0, varianza 1) → training stabile e lr più alti
possibili. **Perché gamma e beta**: la normalizzazione pura è troppo autoritaria —
toglie alla rete anche la libertà di *volere* una scala diversa; i due parametri
appresi gliela restituiscono, partendo dal default sano (gamma=1, beta=0).
**Perché LayerNorm e non BatchNorm** (che magari incontrerai altrove): BatchNorm
normalizza *attraverso il batch* — accoppia esempi indipendenti tra loro, si
comporta diversamente in training e in generazione (dove il batch può essere 1) ed
è un vivaio storico di bug; LayerNorm normalizza ogni posizione *per conto suo* —
niente accoppiamenti, identica in training e inferenza. Per le sequenze non c'è
partita, e i transformer usano LayerNorm ovunque. Il backward della
normalizzazione è il più delicato che scriveremo (la media e la varianza dipendono
da tutti gli elementi del vettore: i gradienti si intrecciano) — gradient check
obbligatorio e derivazione nel reference.

### ☐ 6.4 — Il blocco: residual + pre-norm

**Cosa**: `block.py`:

```
x = x + attn(ln1(x))     # comunica  (con residual e pre-norm)
x = x + ffn(ln2(x))      # elabora   (con residual e pre-norm)
```

**Perché le residual connections sono LA ragione per cui il deep learning è
"deep"** — spiegazione completa. Il problema: il gradiente, per raggiungere i
primi strati di una rete profonda, deve attraversare in catena tutti gli strati
successivi, venendo moltiplicato a ogni passaggio; prodotti di tanti fattori < 1
svaniscono esponenzialmente (o esplodono, se > 1). Prima del 2015, reti oltre
~20 strati *peggioravano* aggiungendo strati. La soluzione (ResNet) è
imbarazzantemente semplice: invece di `x = f(x)`, scrivere `x = x + f(x)`.
Conseguenze:
1. *Per il gradiente*: ricordi il backward della somma (3.2)? — distribuisce il
   gradiente **invariato** a entrambi i rami. Il ramo `x` nudo è quindi
   un'autostrada: il gradiente della loss arriva ai primi strati *intatto*,
   qualunque cosa facciano gli strati in mezzo. (Ecco perché in 3.2 insistevamo su
   quel backward: questa riga è il motivo.)
2. *Per l'apprendimento*: ogni blocco parte dal comportamento "non faccio niente"
   (se `f(x)≈0`, il blocco è l'identità) e impara *correzioni incrementali* a un
   segnale che scorre — non deve ricostruire tutto da capo. Impilare 12 blocchi
   diventa sicuro: al peggio, i blocchi inutili restano vicini all'identità.

**Perché pre-norm** (LayerNorm *dentro* il ramo, prima di attn/ffn — a differenza
del paper originale del 2017 che la metteva dopo la somma): con la post-norm, la
LayerNorm sta *sull'autostrada* e rinormalizza il segnale a ogni blocco,
disturbando proprio il flusso pulito del gradiente; con la pre-norm l'autostrada
resta intonsa da input a output. Empiricamente: la post-norm richiede warmup
delicati per non divergere, la pre-norm è stabile. GPT-2 e tutti i successori sono
pre-norm; anche noi.

### ☐ 6.5 — Test del blocco

**Cosa**: forward su batch reale con shape verificate `(B,T,C) → (B,T,C)` (il
blocco preserva la shape: è ciò che lo rende impilabile); backward completo con
gradienti non nulli e finiti su *tutti* i parametri; gradient check sul blocco
intero; test che a inizializzazione il blocco è ~vicino all'identità.

**Perché il test "gradiente non nullo su ogni parametro"**: è il modo meccanico di
scoprire parametri disconnessi dal grafo (dimenticati in un `parameters()`, o
tagliati fuori da un bug di shape) — che altrimenti resterebbero congelati per
sempre ai valori casuali iniziali, silenziosamente.

**Deliverable di fase**: il mattone completo del GPT, impilabile e testato.
Concetti acquisiti: multi-head, FFN e divisione comunica/elabora, LayerNorm,
residual, pre-norm — ovvero: *perché le reti profonde si addestrano*.

---

## ☐ Fase 7 — RonkLM: il GPT completo

**Obiettivo didattico.** Assemblare tutto in un decoder-only funzionante, colmando
l'ultimo buco concettuale (l'attention non sa *dove* sono i token: serve la
posizione) e costruendo la generazione autoregressiva con i suoi controlli
(temperature, top-k).

### ☐ 7.1 — L'architettura (`gpt.py`)

**Cosa**:

```
indici (B,T)
  → token embedding (B,T,C)  +  positional embedding (T,C)     [somma]
  → Block × n_layer                                            (B,T,C)
  → LayerNorm finale                                           (B,T,C)
  → Linear verso il vocab                                      (B,T,vocab)  = logits
```

Config parametrica (`n_layer`, `n_head`, `n_embd`, `block_size`, `vocab_size`) in
una dataclass. Target iniziale: ~4 layer, 4 teste, `n_embd` 128, `block_size` 128
(~0.8M parametri — tarato in Fase 8 sui tempi CPU reali).

**Perché serve il positional embedding — il buco da colmare**: l'attention è, per
costruzione, un'operazione su *insiemi*: permutando i token di input (e le
maschere), i punteggi q·k non cambiano — nulla nel meccanismo sa che un token è
*prima* di un altro. Ma "ma la" ≠ "la ma": l'ordine è metà del linguaggio. La
soluzione: una seconda tabella di embedding, indicizzata dalla *posizione*
(0,1,…,block_size−1) anziché dal carattere, sommata al token embedding. Ogni
posizione acquista una firma appresa, e le teste possono imparare pattern
posizionali ("guarda 1 indietro") oltre che di contenuto ("cerca la vocale").
**Perché sommare e non concatenare**: la somma mantiene la dimensione (blocchi
identici e impilabili), e con vettori appresi in uno spazio ampio la rete ha campo
libero di allocare sottospazi quasi-ortogonali ai due tipi di informazione se le
serve — empiricamente funziona altrettanto bene della concatenazione, a costo
zero. (I LLM moderni usano schemi più sofisticati — RoPE — che citeremo nel
reference come estensione; l'embedding posizionale appreso è lo schema di GPT-2
ed è perfetto per capire il problema.)

**Perché la LayerNorm finale**: dopo l'ultimo blocco il segnale è la somma di
tutti i residual accumulati, su scala non controllata; la testa che produce i
logits lavora molto meglio su un segnale rinormalizzato. (Standard GPT-2,
coerente con la logica pre-norm.)

### ☐ 7.2 — Forward + loss su tutte le posizioni

**Cosa**: il forward restituisce logits `(B,T,vocab)`; la cross-entropy si calcola
su **tutte** le B·T predizioni contemporaneamente (reshape a `(B·T, vocab)` contro
target `(B·T,)`).

**Perché è il raccolto delle semine precedenti**: qui si incassa materialmente il
"T esempi al prezzo di uno" preparato da 0.4 (Y = X spostato) e reso legittimo da
5.3 (la maschera garantisce che la predizione in `t` non abbia sbirciato oltre
`t`). Ogni batch da `(64, 128)` = 8.192 esempi di training in un solo
forward/backward. Senza questa struttura, il training dei transformer non sarebbe
economicamente possibile.

### ☐ 7.3 — La generazione (`generate.py`): temperature e top-k

**Cosa**: loop autoregressivo — encode del prompt; finché servono caratteri:
forward sugli ultimi `block_size` token, prendi i logits *dell'ultima posizione*,
applica `temperature` e `top-k`, campiona, appendi; decode.

**Perché si tronca il contesto agli ultimi `block_size` token**: la tabella
posizionale ha `block_size` righe e le matrici di attenzione shape T×T — oltre
quella finestra il modello, semplicemente, non è definito. (È il famoso "limite di
contesto" dei LLM: ora sai da quali due tensori nasce.)

**Temperature — cosa fa davvero**: si dividono i logits per `τ` prima della
softmax. Per l'esponenziale della softmax, `τ<1` *allarga* le differenze tra
logits → distribuzione più appuntita → testo più conservativo e ripetitivo;
`τ>1` le comprime → più uniforme → più vario e più sgangherato; `τ→0` = argmax
(greedy: degenere e ciclico, come previsto in 1.3). Non cambia *l'ordine* delle
preferenze del modello: cambia quanto ci si azzarda a deviare dalla prima scelta.

**Top-k — perché serve anche con la temperature**: la coda della distribuzione
(decine di caratteri a probabilità minuscola ma non nulla) di tanto in tanto viene
comunque pescata, e un singolo carattere assurdo (`%` in mezzo a una parola) può
deragliare tutto il seguito — il modello non ha mai visto contesti col `%` lì, e
genera spazzatura da spazzatura (errore composto: il testo generato esce dalla
distribuzione su cui il modello è stato addestrato). Top-k taglia la coda:
tieni i k logits migliori, azzera (a −inf) gli altri, rinormalizza. Insieme,
`temperature` e `top-k` sono le stesse due manopole dei LLM di produzione.

### ☐ 7.4 — Salvataggio e caricamento

**Cosa**: `save`/`load` con `np.savez` (un archivio con tutti i parametri
nominati) + il vocabolario del tokenizer + la config del modello, in un unico
checkpoint.

**Perché vocabolario e config *dentro* il checkpoint**: un modello char-level
caricato con `stoi` diverso da quello di training produce spazzatura deterministica
e difficilissima da diagnosticare (i pesi sono "giusti" ma parlano un'altra mappa
di indici — vedi 0.3); una config diversa non fa nemmeno combaciare le shape. Il
checkpoint deve essere *autosufficiente*: pesi + mappa + architettura, sempre
insieme.

### ☐ 7.5 — Primo training end-to-end

**Cosa**: training completo su Pinocchio, tabella NLL aggiornata (bigram → MLP →
GPT), campioni generati a più temperature, salvataggio del primo checkpoint
ufficiale.

**Perché ci aspettiamo il salto**: rispetto all'MLP, il GPT vede 128 caratteri di
contesto (vs 8), con pesi condivisi tra posizioni e attenzione dinamica: NLL val
attesa nettamente sotto quella dell'MLP, e nel testo: parole quasi tutte reali,
punteggiatura sensata, struttura dei dialoghi collodiani (`— disse…`), nomi dei
personaggi ricombinati. Coerenza narrativa: assente, come promesso in I.5.

**Deliverable di fase**: **RonkLM v1** — un GPT char-level completo, scritto
interamente a mano sopra il nostro autograd, che genera pseudo-Collodi.

---

## ☐ Fase 8 — Training serio, esperimenti e CLI

**Obiettivo didattico.** Da "gira" a "gira bene, si misura e si usa". Gli
iperparametri si capiscono solo toccandoli: questa fase è un laboratorio
sperimentale documentato.

### ☐ 8.1 — CLI (`scripts/train_ronklm.py`)

**Cosa**: entrypoint `argparse`: sottocomandi `train` (dataset, config modello,
lr, step, seed, checkpoint di ripresa) e `generate` (checkpoint, prompt,
temperature, top-k, lunghezza).

**Perché una CLI e non notebook/script da editare**: ogni esperimento resta
riproducibile dal suo *comando* (che salveremo nel log dell'esperimento insieme al
seed — II.2.3); modificare il codice a ogni prova distrugge la confrontabilità tra
esperimenti, che in 8.4 è tutto.

### ☐ 8.2 — Loop di training robusto

**Cosa**: eval periodica su val (media su più batch), log a intervalli regolari
(step, loss, lr corrente, tempo/step), checkpoint del *best model* su val,
possibilità di riprendere un training interrotto.

**Perché la loss di eval si media su più batch**: la loss di un singolo batch è
una stima rumorosa (2.4) — su un batch fortunato può sembrare un progresso che non
esiste. Mediare su 20–50 batch fissi di val dà un numero stabile: le decisioni
(early stop, best model) si prendono su quello. **Perché salvare il best su val e
non l'ultimo**: se il modello inizia a overfittare (4.3), l'ultimo checkpoint è
*peggiore* di uno intermedio; tenere il migliore su val è la forma più semplice di
early stopping.

### ☐ 8.3 — Learning rate schedule: warmup + cosine decay

**Cosa**: lr che sale linearmente da ~0 al valore pieno nei primi ~2–5% degli step
(warmup), poi scende seguendo un coseno fino a ~1/10 del picco.

**Perché il warmup — legato ad Adam (4.4)**: le medie mobili di Adam nei
primissimi step sono stime basate su pochissimi campioni; in particolare il
secondo momento (che sta al *denominatore* del passo) può essere sottostimato →
passi enormi in direzioni rumorose, su una rete appena inizializzata che è nel suo
punto più fragile. Il warmup tiene i passi piccoli finché le stime non maturano.
**Perché il decay finale**: a fine training si è vicini a un minimo; passi grandi
*orbitano* attorno al minimo senza entrarci (l'ampiezza dell'oscillazione è
proporzionale al lr). Ridurre il lr permette di *depositarsi*. Il coseno è la
forma dolce standard (nessun salto brusco); l'effetto del solo schedule sulla NLL
finale è visibile e lo misureremo (esperimento in 8.4).

### ☐ 8.4 — Il laboratorio: esperimenti documentati

**Cosa**: griglia di esperimenti one-factor-at-a-time, ognuno con comando, seed,
curva e NLL finale, raccolti in una tabella nel reference: profondità
(`n_layer` 1/2/4/8 a parità di parametri totali), `block_size` (32/128), con/senza
schedule, con/senza weight decay, `temperature` e `top-k` in generazione (stesso
checkpoint, campioni a confronto). Se i tempi CPU lo richiedono, la griglia si
riduce — ma ogni esperimento tagliato viene elencato come "non fatto".

**Perché one-factor-at-a-time**: cambiando due cose insieme non si sa a cosa
attribuire la differenza. Meno efficiente di una ricerca su griglia completa, ma
qui l'obiettivo è *capire il contributo di ogni idea*, non trovare l'ottimo.
È il metodo sperimentale applicato al nostro stesso progetto, e la tabella finale
è la risposta empirica alla domanda "cosa compra ciascun pezzo del transformer?".

### ☐ 8.5 — README e galleria

**Cosa**: README con quickstart (installare, addestrare, generare in 3 comandi),
la tabella NLL completa bigram→GPT, campioni di testo per fase (la "galleria
dell'evoluzione"), e la lista delle estensioni possibili (BPE, RoPE, weight
tying, dropout, riscrittura PyTorch) ognuna con una riga sul perché sarebbe il
passo giusto.

**Perché la galleria**: mettere in fila la generazione del bigram, dell'MLP e del
GPT sullo stesso prompt è la dimostrazione più eloquente dell'intero progetto —
si *vede* la scala di modelli promessa in I.1.

### ☐ 8.6 — Verifica finale dell'atlante

**Cosa**: passata meccanica di verifica del `codebase_reference.md` contro il
codice reale (estrazione firme via grep/AST e confronto, come da istruzioni
globali), correzione di ogni divergenza.

**Perché meccanica e non a memoria**: dopo 8 fasi di modifiche, la probabilità che
l'atlante sia perfettamente allineato "a occhio" è zero; e un atlante sbagliato è
peggio di nessun atlante (istruzioni globali, con ragione).

**Deliverable di fase**: RonkLM addestrabile e usabile da CLI, con esperimenti
documentati e documentazione completa. **Fine del Percorso A.**

---

<a name="parte-iii-b"></a>
# Parte III-B — Percorso B: la scalata a 50M

**La promessa di questo percorso**: un RonkLM da ~50M di parametri che scrive
**italiano sensato a livello di frase e paragrafo**. E il suo prerequisito
assoluto: il Percorso A completato — ogni fase di questo percorso *riusa come
verità di riferimento* qualcosa costruito nel Percorso A (il motore per il test di
equivalenza, la pipeline dati come modello mentale, la tabella NLL come metro).

**Perché 50M è il numero giusto per "sensato ma addestrabile da una persona"**:
è la scala di GPT-2 small diviso 2 (124M) e di progetti come TinyStories, che
hanno dimostrato che decine di milioni di parametri **bastano per grammatica
solida e coerenza locale** se i dati sono buoni; ed è ancora dentro i limiti di
una singola GPU consumer (il modello in mixed precision + AdamW + attivazioni sta
in ~8–12 GB di VRAM con batch accumulato). 10× più piccolo non sarebbe mai
"sensato"; 10× più grande non sarebbe più un progetto personale.

---

## ☐ Fase 9 — Port a PyTorch, con equivalenza dimostrata

**Obiettivo didattico.** Tradurre RonkLM dal nostro motore a PyTorch e — questo è
il punto — **dimostrare con un test numerico che i due dicono le stesse cose**.
Non è una fase di abbandono del lavoro fatto: è la fase in cui il lavoro fatto
diventa lo strumento di verifica di tutto ciò che verrà.

### ☐ 9.1 — Traduzione dell'architettura (`ronklm_torch/model.py`)

**Cosa**: riscrivere il GPT della Fase 7 in PyTorch, mantenendo *identica*
l'architettura e la nomenclatura dei parametri. Tabella di corrispondenza nel
reference: `ronkgrad.Tensor` ↔ `torch.Tensor(requires_grad=True)`, il nostro
`Module/parameters()` ↔ `nn.Module`, la nostra `Linear` ↔ `nn.Linear`, la nostra
fusione softmax+NLL ↔ `F.cross_entropy`, il nostro AdamW ↔ `torch.optim.AdamW`.

**Perché la tabella di corrispondenza è il vero contenuto della sottofase**: è la
dimostrazione materiale che PyTorch non contiene *nessun concetto* che non abbiamo
già implementato — solo ingegneria migliore (kernel C++/CUDA, fusione di
operazioni, gestione memoria). Ogni riga della tabella rimanda alla fase del
Percorso A dove quel pezzo è stato costruito e derivato. Chi legge il codice
PyTorch di RonkLM con questa tabella in mano non incontra mai magia.

### ☐ 9.2 — Il test di equivalenza (`tests/test_equivalence.py`)

**Cosa**: (a) *forward*: stessi pesi (esportati dal checkpoint NumPy della Fase 7,
importati nel modello torch), stesso input → stessi logits entro tolleranza float
(~1e-5 relativa); (b) *backward*: stessa loss sullo stesso batch → stessi
gradienti sui parametri, entro tolleranza; (c) *training*: pochi step con stesso
seed e stessi batch → stesse loss step per step (entro deriva float).

**Perché è la sottofase irrinunciabile del percorso**: un port "a occhio" che
diverge silenziosamente dal riferimento è il modo classico di portarsi dietro un
bug sottile per mesi (una trasposizione, un `1/√d` dimenticato, una LayerNorm su
un asse sbagliato — tutte cose che *non crashano*, degradano soltanto). Il
Percorso A ci ha regalato un lusso raro nel deep learning: **una implementazione
di riferimento di cui ci fidiamo fino all'ultima riga, perché l'abbiamo derivata
noi con i gradient check**. Questo test la spende. In più, è il gradient check
della Fase 2.6 riproposto un'ottava sopra: due strade indipendenti, stesso numero,
oppure c'è un bug. È lo stesso principio, ed è l'ultima volta che potremo
permettercelo — da qui in poi il riferimento sarà RonkLM-torch stesso.

### ☐ 9.3 — Benchmark e primi assaggi di GPU

**Cosa**: misurare token/secondo di training: NumPy-CPU vs torch-CPU vs torch-GPU
(se disponibile), sulla config della Fase 7; documentare i fattori di speedup
reali. Introdurre `device`, `torch.compile` e la mixed precision (bf16/fp16) con
una spiegazione di cosa sono e perché la GPU li rende possibili.

**Perché misurare invece di citare**: i "3–4 ordini di grandezza" promessi in I.2
diventano un numero *nostro*, misurato sul *nostro* modello. E il benchmark decide
la questione hardware della Fase 12 con dati alla mano (quanti giorni per N
miliardi di token sulla GPU X). **Mixed precision, il perché in breve**: le GPU
moderne hanno unità dedicate (tensor core) che macinano fp16/bf16 a velocità
multiple del fp32; i pesi master restano in fp32 per non accumulare errori di
arrotondamento negli update piccoli. Dettagli operativi al momento dell'uso.

**Deliverable di fase**: RonkLM-torch, *dimostrato* equivalente al riferimento,
con benchmark. Il Percorso A è ufficialmente diventato la suite di test del
Percorso B.

---

## ☐ Fase 10 — Tokenizer BPE scritto a mano

**Obiettivo didattico.** Il pezzo del Percorso A volutamente rimandato (I.3):
adesso serve davvero, e lo costruiamo da zero come tutto il resto (`ronklm/bpe.py`,
~200 righe). È lo stesso algoritmo (byte-pair encoding) dei tokenizer GPT reali.

### ☐ 10.1 — Perché il char-level non può arrivare a "sensato"

**Cosa**: sezione scritta + esperimento di conteggio sul corpus.

**Il perché, quantitativo.** Tre costi del char-level che a 50M diventano
proibitivi: (1) *contesto effettivo* — 512 posizioni char-level ≈ 80 parole
italiane: troppo poche per la coerenza di un paragrafo; con BPE (~3.5–4 caratteri
per token in italiano) le stesse 512 posizioni ≈ 300+ parole; e siccome
l'attention costa O(T²), comprare contesto allungando T è la strada cara, mentre
comprarlo *densificando i token* è quasi gratis. (2) *capacità sprecata* — un
modello char-level spende una fetta dei suoi strati solo per ricomporre
l'ortografia ("m-a-n-g-i-a-r-e è una parola") prima ancora di poter modellare la
sintassi; il BPE gli consegna le parole frequenti già intere. (3) *profondità di
predizione* — predire il prossimo *token* è un compito semanticamente più ricco
che predire la prossima *lettera* (spesso ovvia), quindi ogni step di training
insegna di più.

### ☐ 10.2 — L'algoritmo BPE, da zero

**Cosa**: training del tokenizer: si parte dai byte/caratteri; si conta la coppia
adiacente più frequente nel corpus; la si fonde in un nuovo simbolo; si ripete
fino a `vocab_size` (target: **~16.000**). Poi `encode` (applicare le fusioni in
ordine di apprendimento) e `decode` (concatenare). Serializzazione delle fusioni
su file (il tokenizer è a tutti gli effetti un modello: si addestra e si salva).

**Perché l'algoritmo è fatto così**: è compressione guidata dai dati — le
sequenze frequenti *meritano* un simbolo dedicato, quelle rare restano scomposte
in pezzi. Il risultato è un vocabolario adattivo: `"di"`, `"che"`, `"mente"`
diventano token singoli, `"Ranocchiaio"` resta spezzato ma *rappresentabile* —
nessuna parola è mai fuori vocabolario, perché nel caso peggiore si scende ai
caratteri (la proprietà che il word-level non poteva dare, I.3). **Perché ~16k e
non i ~50–100k dei GPT commerciali**: quelli servono decine di lingue più codice;
noi una lingua sola — un vocabolario troppo grande su un corpus solo italiano
riempirebbe la coda di token rarissimi (visti poche volte → embedding mai
addestrati bene) e ingrosserebbe embedding e testa finale, che a vocab 16k e
`n_embd` 512 pesano già ~8M di parametri l'uno. 16k è il punto di equilibrio
misurabile: lo verificheremo guardando rapporto di compressione e frequenze di
coda (10.3).

### ☐ 10.3 — Test e misure

**Cosa**: round-trip `decode(encode(s)) == s` su testo arbitrario (inclusi
accenti e simboli rari); rapporto di compressione (caratteri/token) sul corpus;
istogramma delle frequenze dei token (la coda rara deve essere piccola);
ispezione a occhio dei primi 200 merge appresi (devono "sembrare italiano":
`ch`, `di`, `re`, `zione`…).

**Perché l'ispezione a occhio dei merge**: è l'equivalente per il tokenizer delle
heatmap di attention (5.5) — una finestra diretta su cosa l'algoritmo ha imparato,
e il modo più rapido di accorgersi di un corpus sporco (se tra i primi merge
compaiono artefatti HTML o spazi doppi, la pulizia della Fase 11 ha un buco).

**Deliverable di fase**: BPE nostro, testato, addestrato sull'italiano. RonkLM
smette di sillabare.

---

## ☐ Fase 11 — Il corpus grande

**Obiettivo didattico.** La lezione che i ranking dei LLM hanno insegnato al
mondo: **i dati sono metà del modello**. Costruiamo la pipeline che porta da
"file scaricati da internet" a "miliardi di token puliti, deduplicati,
pre-tokenizzati e streammabili".

### ☐ 11.1 — Quanti dati servono: le leggi di scala

**Cosa**: sezione scritta con i conti, che fissa il target dati del progetto.

**Il perché, coi numeri.** La ricerca sulle leggi di scala (Chinchilla, 2022) ha
misurato che a parità di budget di calcolo il rapporto ottimale è **~20 token di
training per parametro**: per 50M parametri, ~**1 miliardo di token** (≈ 4 GB di
testo italiano). Sotto quella soglia il modello è sotto-nutrito (i parametri in
più non rendono); si può andare *oltre* (over-training: più token per parametro)
e per noi è anzi desiderabile, perché il nostro vincolo è la dimensione del
modello, non il calcolo. Target pragmatico: **1–2 miliardi di token**. Fonti
candidate, in ordine di qualità: Wikipedia italiana (~1.2 GB di testo, ~350–400M
token: pulita, enciclopedica), libri italiani di pubblico dominio da
LiberLiber/Gutenberg (~centinaia di MB: prosa di qualità), un sottoinsieme
italiano di un corpus web curato (CulturaX/OSCAR: volume, ma da filtrare con
cura). La scelta finale della miscela è una decisione registrata di questa fase.
**Nota strategica sull'alternativa TinyStories**: è dimostrato che un corpus
*sintetico, semplice e curato* (storie brevi con lessico controllato) produce
coerenza percepita molto superiore a parità di parametri; la strada "RonkLM
narratore di storie semplici" resta documentata come piano B se la miscela
naturale desse risultati deludenti — con il trade-off dichiarato: più coerenza,
meno copertura della lingua reale.

### ☐ 11.2 — La pipeline: pulizia e deduplicazione

**Cosa**: `data/corpus_b/`: script di download riproducibili (come 0.2, in
grande); estrazione testo (per Wikipedia: da dump ufficiale, via estrattore
standard); filtri di qualità (lunghezza minima, proporzione di caratteri
alfabetici, rimozione boilerplate); **deduplicazione** a livello di documento
(hash) e quasi-duplicati; normalizzazione coerente con 0.2.

**Perché la deduplicazione è il filtro più importante**: il web è pieno di testo
ripetuto (mirror, citazioni, template). I duplicati (a) fanno *memorizzare* il
modello invece di generalizzare — lo stesso testo visto 50 volte è 50 volte più
memorizzato; (b) **contaminano la validazione**: se lo stesso documento finisce
in train e in val, la loss di val mente — e siccome (regola 4.3) è l'unico numero
di cui ci fidiamo, un val contaminato rompe la bussola dell'intero percorso. Lo
split train/val qui si fa *per documento*, mai per riga (stessa logica dello
split contiguo di 0.4, un'ottava sopra).

### ☐ 11.3 — Pre-tokenizzazione e storage binario

**Cosa**: tokenizzare *una volta* l'intero corpus col BPE della Fase 10; salvare
gli id in file binari (`uint16` — basta, con vocab 16k < 65.536) in shard;
caricamento in training via `np.memmap`; `get_batch` che estrae finestre casuali
dagli shard (la 0.4, ri-implementata per dati che non stanno in RAM).

**Perché pre-tokenizzare e perché il binario**: tokenizzare al volo costerebbe
CPU a ogni epoca ripetendo sempre lo stesso identico lavoro, e terrebbe la GPU
affamata (il collo di bottiglia di un training ben fatto dev'essere la GPU, mai il
caricamento dati). `uint16` = 2 byte/token → 1.5 miliardi di token ≈ 3 GB su
disco: con `memmap` il sistema operativo pagina in RAM solo le finestre lette,
quindi il training parte in un secondo e usa memoria costante. È esattamente lo
schema di nanoGPT, e ora abbiamo tutto il background per capirne ogni scelta.

**Deliverable di fase**: 1–2 miliardi di token italiani puliti, deduplicati,
binarizzati, streammabili. La "tabella impossibile" della Fase 1.1 ha finalmente
il suo avversario alla giusta scala.

---

## ☐ Fase 12 — RonkLM-50M: architettura, training run, valutazione

**Obiettivo didattico e traguardo del progetto.** Dimensionare, addestrare e
valutare il modello da 50M. Qui si raccoglie tutto: l'architettura della Fase 7,
il motore della Fase 9, i token della Fase 10, i dati della Fase 11.

### ☐ 12.1 — Dimensionamento, con l'aritmetica esplicita

**Cosa**: config di riferimento (da raffinare col benchmark 9.3):
`n_layer=8, n_head=8, n_embd=512, block_size=512, vocab=16k`. Conti nel
reference, riga per riga:

```
Embedding token:      16.384 × 512                  ≈  8,4M
Embedding posizioni:  512 × 512                     ≈  0,3M
Per blocco:  attn (4 × 512²) + ffn (8 × 512²)       ≈  3,1M
8 blocchi:                                          ≈ 25,2M
Testa finale:         512 × 16.384                  ≈  8,4M
                                          totale    ≈ 42–50M (secondo varianti)
```

**Perché mostrare l'aritmetica**: per rendere tangibile *dove abitano* i
parametri — sorpresa istruttiva: a questa scala **un terzo del modello sta negli
embedding e nella testa**, ed è il motivo per cui esiste il **weight tying**
(usare la *stessa* matrice per embedding di ingresso e testa di uscita: i due
oggetti mappano lo stesso vocabolario nello stesso spazio semantico, in direzioni
opposte — condividerli risparmia ~8M di parametri e in pratica *migliora* la
loss). Rimandato apposta fin qui: è la prima scala a cui il beneficio si vede.
Decisione registrata in questa sottofase dopo un confronto A/B piccolo.

### ☐ 12.2 — Il training run

**Cosa**: training completo con la CLI (estesa da 8.1): mixed precision,
**gradient accumulation**, warmup+cosine (8.3), gradient clipping, checkpoint
periodici e ripresa robusta (un run di giorni *verrà* interrotto), log su CSV +
grafici. Prima un **run pilota** su ~1/10 dei dati e un modello ~10M per validare
pipeline e stime di tempo, *poi* il run vero.

**Perché le tecniche nuove, una per una**: *gradient accumulation* — il batch
"giusto" a questa scala (~0.5M token per update, standard GPT-2) non entra in
VRAM; si sommano i gradienti di K micro-batch prima di ogni update: matematicamente
identico a un batch K volte più grande (il backward della somma, ancora lui),
VRAM costante. *Gradient clipping* — su miliardi di batch qualcuno produce un
gradiente anomalo (documento strano, coincidenza numerica); un singolo passo
gigante può buttare il modello in una zona da cui non si riprende (loss spike);
il clipping taglia la *norma* del gradiente a una soglia: assicurazione a costo
zero. *Run pilota prima del run vero* — un errore scoperto al giorno 3 di un run
da 5 giorni costa 3 giorni; lo stesso errore nel pilota costa un'ora. È il
gradient check della gestione di progetto.

**Hardware, decisione da chiudere qui**: GPU propria (≥12 GB VRAM: run di giorni)
oppure noleggio cloud (una consumer-class in cloud costa ~0.3–0.5 €/h: un run
completo nell'ordine delle decine di euro). Il benchmark 9.3 fornisce i numeri
per decidere; il piano non presuppone nessuna delle due.

### ☐ 12.3 — Valutazione e chiusura

**Cosa**: (a) NLL/perplexity su val — nota metodologica: i numeri BPE **non sono
confrontabili** con la tabella char-level del Percorso A (unità diverse: nats per
token ≠ nats per carattere; convertiremo in *bits per byte* per avere un metro
unico attraverso l'intero progetto); (b) galleria di generazioni con prompt
fissi a più temperature; (c) valutazione qualitativa strutturata su una griglia
(grammatica / coerenza nel paragrafo / coerenza oltre il paragrafo / fatti); (d)
aggiornamento finale di atlante e README.

**Aspettative oneste, per l'ultima volta**: RonkLM-50M scriverà **frasi italiane
corrette e paragrafi che stanno in piedi** — registro coerente, anafora corretta
("Maria… lei…"), argomento mantenuto per qualche frase. Andrà **ancora** alla
deriva su testi lunghi, inventerà fatti con disinvoltura e **non** eseguirà
istruzioni ("riassumi questo testo" non funzionerà: è un modello di
*completamento*, non di *istruzioni* — trasformarlo in assistente richiederebbe
fine-tuning supervisionato e RLHF, che documenteremo in chiusura come "il
capitolo successivo che questo progetto sceglie di non fare"). Questo è "sensato,
non GPT-5.6", per contratto.

**Deliverable di fase e di progetto**: **RonkLM-50M** — un modello addestrato da
noi, su dati preparati da noi, con un tokenizer scritto da noi, con
un'architettura di cui possediamo ogni derivata. Fine del Percorso B.

---

<a name="parte-iv"></a>
# Parte IV — Milestone, rituale, stato

## IV.1 Milestone e criteri di "fatto"

| Milestone | Fasi | Criterio oggettivo di completamento |
|---|---|---|
| M1 — Dati & intuizione | 0–2 | Bigram neurale converge alla NLL del bigram contato (±0.01); gradient check ok |
| M2 — Motore | 3–4 | `test_autograd.py` verde su ogni op; NLL(MLP) < NLL(bigram) su val |
| M3 — Attenzione | 5–6 | Blocco transformer con gradient check ok e gradienti presenti su tutti i parametri |
| M4 — GPT (NumPy) | 7 | NLL(GPT) < NLL(MLP) su val; generazione con temperature/top-k; checkpoint autosufficiente |
| M5 — Prodotto A | 8 | Training da CLI riproducibile; tabella esperimenti; atlante verificato meccanicamente |
| **M6 — Equivalenza** | 9 | `test_equivalence.py` verde: torch e NumPy danno stessi logits/gradienti/loss entro tolleranza |
| **M7 — BPE + dati** | 10–11 | BPE round-trip ok su italiano; 1–2 mld di token puliti, deduplicati, binarizzati |
| **M8 — RonkLM-50M** | 12 | Run da ~50M completato; italiano corretto a livello di frase/paragrafo; valutazione qualitativa documentata |

## IV.2 Rituale di fine fase (da istruzioni globali — obbligatorio)

Alla fine di **ogni** fase, senza attendere richiesta:
1. Aggiornare **questo** piano: checkbox, note su deviazioni e decisioni prese.
2. Aggiornare `memory/codebase_reference.md` (atlante tecnico: firme, tabelle,
   dove-sta-cosa).
3. Generare/aggiornare `explain.md` alla radice (spiegazione didattica **profonda**,
   dai primi principi, ben indicizzata: come funziona ogni cosa e *perché* è stata
   scelta — vedi Parte IV.5).
4. Messaggio **estremamente dettagliato**: stato, checkbox di fase e sottofasi,
   commento sullo stato generale e su quello della fase.
5. Commit e push su branch versionato (`v1.0.0` → incrementi per entità), **su
   entrambi i remote** (`github` e `gitea`).

## IV.5 I tre documenti del progetto (ruoli distinti)

Per non duplicare e non confondere, ogni documento ha un ruolo netto:

| Documento | Ruolo | Taglio |
|---|---|---|
| `memory/plan_ronklm_system.md` | **il piano**: cosa faremo, in che ordine, con quale *perché strategico* | roadmap a fasi, checkbox, decisioni |
| `memory/codebase_reference.md` | **l'atlante**: dove sta cosa, firme, tabelle, trappole | riferimento tecnico secco |
| `explain.md` (radice) | **il libro di testo**: come funziona ogni cosa dai primi principi | didattica profonda, esempi, analogie |

`explain.md` è pensato per un lettore che parte da zero: spiega non solo *cosa* fa il
codice ma *come funzionano* i concetti sottostanti (tokenizzazione, tensori, PRNG,
entropia, gradienti…), con esempi sui numeri reali del nostro corpus. Cresce di una
sezione a ogni fase e mantiene un indice navigabile.

## IV.3 Stato attuale

- ☑ **Kickoff**: cartella `memory/` creata; piano v1 redatto; piano v2
  (riscrittura approfondita) e piano v2.1 (aggiunto Percorso B fino a 50M)
  completati il 2026-07-18.
- ☑ **Fase 0 — Fondamenta** ✅ (2026-07-18): git+remote, corpus Pinocchio pulito
  (240.920 char, vocab 69), `CharTokenizer`, `Dataset`+batching, 16 test verdi.
  `codebase_reference.md` creato. Branch di rilascio: `v1.1.0`.
- ☐ **Fase 1 — Bigram per conteggio** *(prossima)*.
- ☐ Fasi 2–8 (Percorso A, NumPy, ~1M) — non iniziate.
- ☐ Fasi 9–12 (Percorso B, PyTorch, 50M) — non iniziate; **sbloccate solo a
  Percorso A completato** (il Percorso A è la suite di test del Percorso B).

## IV.4 Decisioni prese e decisioni aperte

**Prese** (con riferimento alla motivazione):
- Obiettivo finale: **~50M parametri**, italiano *sensato* a livello di frase/paragrafo (→ intro, Fase 12).
- Struttura a **due percorsi**: A = capire (NumPy, ~1M); B = scalare (PyTorch, 50M) (→ I.2).
- Percorso A stack: Python + NumPy puro (→ I.2); tokenizer char-level (→ I.3).
- Percorso B stack: PyTorch dopo test di equivalenza (→ Fase 9); tokenizer BPE a mano (→ Fase 10).
- Percorso: a fasi, bigram→GPT→scala (→ I.4).
- Corpus A: italiano, *Pinocchio* (→ I.5). Corpus B: 1–2 mld token italiani (→ Fase 11).
- Ottimizzatore finale: AdamW scritto a mano (→ 4.4). Architettura: pre-norm (→ 6.4).
- **Remote git**: `github` (Frostmoore/ronklm) + `gitea` (smp-webmaster/ronklm); push su **entrambi** a ogni fine fase.

**Aperte** (da chiudere nella fase indicata):
- Fonte esatta e regole di pulizia del corpus A → 0.2.
- Dimensioni del modello A (dipendono dai tempi CPU misurati) → 7.1/8.4.
- Miscela esatta del corpus B (Wikipedia / libri PD / web curato / eventuale TinyStories-IT) → 11.1.
- Config esatta del 50M e weight tying → 12.1, guidati dal benchmark 9.3.
- **Hardware per il run da 50M**: GPU propria vs noleggio cloud → deciso in 12.2 coi numeri di 9.3.
- Post-progetto (RoPE, dropout, fine-tuning istruzioni/RLHF) → fuori scope, elencati in 12.3.
