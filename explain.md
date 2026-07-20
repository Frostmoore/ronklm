# explain.md — RonkLM spiegato per filo e per segno

> **Cos'è questo documento.** È il *libro di testo* di RonkLM. Mentre il
> `codebase_reference.md` è l'atlante tecnico (firme, tabelle, "dove sta cosa") e il
> `plan_ronklm_system.md` è la roadmap, **questo file spiega come funziona ogni
> cosa, dai primi principi**, per un lettore che parte da zero sul deep learning.
> Non do per scontato nulla: quando serve un concetto (cos'è un tensore, come
> funziona un generatore di numeri casuali, cos'è l'entropia) lo spiego qui, con
> esempi presi dai **numeri veri** del nostro corpus.
>
> **Come leggerlo.** È indicizzato: usa l'indice qui sotto (in un lettore Markdown i
> link saltano alla sezione). Cresce di un capitolo a ogni fase del progetto. I
> riquadri **📖 Concetto** introducono un'idea generale; i riquadri **🔧 Nel codice**
> la collegano a ciò che abbiamo scritto; i riquadri **⚠️ Trappola** raccontano gli
> errori veri in cui siamo inciampati e perché.

---

## Indice

- [Fase 0 — Fondamenta: dal testo ai batch](#fase-0)
  - [0.0 Il quadro generale: cos'è davvero un LLM](#sec-0-0)
  - [0.1 Perché un modello vede numeri, non lettere](#sec-0-1)
  - [0.2 Caratteri, Unicode e UTF-8: cosa sono davvero i "simboli"](#sec-0-2)
  - [0.3 La tokenizzazione a caratteri (il `CharTokenizer`)](#sec-0-3)
  - [0.4 Il corpus: da dove viene e come l'abbiamo ripulito](#sec-0-4)
  - [0.5 Anatomia del vocabolario e delle frequenze](#sec-0-5)
  - [0.6 NumPy e i tensori: il minimo indispensabile](#sec-0-6)
  - [0.7 Il dataset: perché si divide in train e validation](#sec-0-7)
  - [0.8 Il trucco X/Y: T esempi al prezzo di uno](#sec-0-8)
  - [0.9 Batch, casualità e riproducibilità (PRNG e seed)](#sec-0-9)
  - [0.10 La disciplina dei test](#sec-0-10)
  - [0.11 Git, versioni e i due remote](#sec-0-11)
  - [0.12 Le tre trappole, spiegate con l'autopsia](#sec-0-12)
  - [0.13 Glossario dei termini introdotti](#sec-0-13)
  - [0.14 Cosa sappiamo fare ora, e cosa arriva in Fase 1](#sec-0-14)
- [Fase 1 — Il bigram: il primo language model](#fase-1)
  - [1.0 L'idea: predire guardando un solo carattere](#sec-1-0)
  - [1.1 La matrice dei conteggi](#sec-1-1)
  - [1.2 Da conteggi a probabilità: normalizzare e lo smoothing](#sec-1-2)
  - [1.3 Broadcasting: la regola che allinea le shape](#sec-1-3)
  - [1.4 Generare testo: il campionamento autoregressivo](#sec-1-4)
  - [1.5 La loss spiegata a fondo: negative log-likelihood](#sec-1-5)
  - [1.6 I nostri numeri, letti uno per uno](#sec-1-6)
  - [1.7 Il limite del bigram e perché è il punto di partenza giusto](#sec-1-7)
  - [1.8 Glossario Fase 1 / cosa arriva in Fase 2](#sec-1-8)
- [Fase 2 — Imparare invece di contare: la backpropagation](#fase-2)
  - [2.0 L'idea: la stessa meta, per la strada opposta](#sec-2-0)
  - [2.1 One-hot e la matrice dei pesi W](#sec-2-1)
  - [2.2 La softmax, spiegata a fondo](#sec-2-2)
  - [2.3 La derivazione del gradiente, a mano, passo per passo](#sec-2-3)
  - [2.4 Il training loop: i 5 passi del deep learning](#sec-2-4)
  - [2.5 Il learning rate: l'arte del passo giusto](#sec-2-5)
  - [2.6 La prova del nove: neurale ≡ conteggi](#sec-2-6)
  - [2.7 Il gradient check: come si verifica un gradiente](#sec-2-7)
  - [2.8 Un bonus elegante: allenare dai conteggi](#sec-2-8)
  - [2.9 Glossario Fase 2 / cosa arriva in Fase 3](#sec-2-9)
- [Fase 3 — ronkgrad: costruire un mini-PyTorch](#fase-3)
  - [3.0 Il problema: derivare a mano non scala](#sec-3-0)
  - [3.1 L'idea: il grafo computazionale](#sec-3-1)
  - [3.2 La classe Tensor: dati, gradiente, e "come tornare indietro"](#sec-3-2)
  - [3.3 Le operazioni e i loro backward](#sec-3-3)
  - [3.4 Il broadcasting all'indietro: il punto più insidioso](#sec-3-4)
  - [3.5 Perché i gradienti si accumulano](#sec-3-5)
  - [3.6 backward(): l'ordinamento topologico](#sec-3-6)
  - [3.7 Le non-linearità e perché sono obbligatorie](#sec-3-7)
  - [3.8 cross_entropy fusa: stabilità ed eleganza](#sec-3-8)
  - [3.9 La prova: 18 gradient check e la precisione macchina](#sec-3-9)
  - [3.10 Glossario Fase 3 / cosa arriva in Fase 4](#sec-3-10)
- [Fase 4 — L'MLP: contesto, embedding e AdamW](#fase-4)
  - [4.0 L'idea: rompere il limite di un solo carattere](#sec-4-0)
  - [4.1 L'infrastruttura: Module, Linear, Embedding](#sec-4-1)
  - [4.2 Gli embedding: comprimere la tabella impossibile](#sec-4-2)
  - [4.3 Lo strato nascosto: rilevare combinazioni](#sec-4-3)
  - [4.4 L'inizializzazione dei pesi: 1/√n](#sec-4-4)
  - [4.5 AdamW, l'ottimizzatore dei GPT veri](#sec-4-5)
  - [4.6 L'overfitting, dal vivo](#sec-4-6)
  - [4.7 I numeri e il testo generato](#sec-4-7)
  - [4.8 Il limite dell'MLP e cosa arriva in Fase 5](#sec-4-8)
- [Fase 5 — Self-attention: il cuore del transformer](#fase-5)
  - [5.0 L'idea in una frase](#sec-5-0)
  - [5.1 Query, Key, Value: la metafora onesta](#sec-5-1)
  - [5.2 I punteggi di affinità e lo scaling 1/√d](#sec-5-2)
  - [5.3 La maschera causale: la freccia del tempo](#sec-5-3)
  - [5.4 La media pesata dei value](#sec-5-4)
  - [5.5 Guardare dentro: la heatmap di attenzione](#sec-5-5)
  - [5.6 Perché batte i limiti dell'MLP (e cosa manca ancora)](#sec-5-6)
  - [5.7 Glossario Fase 5 / cosa arriva in Fase 6](#sec-5-7)
- [Fase 6 — Il blocco Transformer: perché le reti profonde funzionano](#fase-6)
  - [6.0 L'unità che i GPT ripetono](#sec-6-0)
  - [6.1 Multi-head: più sguardi in parallelo](#sec-6-1)
  - [6.2 Feed-forward: comunicare e poi pensare](#sec-6-2)
  - [6.3 LayerNorm: un riferimento fisso a ogni strato](#sec-6-3)
  - [6.4 Le connessioni residue: l'autostrada del gradiente](#sec-6-4)
  - [6.5 Pre-norm vs post-norm](#sec-6-5)
  - [6.6 Il blocco è impilabile](#sec-6-6)
  - [6.7 Glossario Fase 6 / cosa arriva in Fase 7](#sec-6-7)
- [Fase 7 — RonkLM: il GPT completo](#fase-7)
  - [7.0 Assemblare tutti i pezzi](#sec-7-0)
  - [7.1 Il positional embedding: dare un ordine alla sequenza](#sec-7-1)
  - [7.2 Lo stack di blocchi e la testa finale](#sec-7-2)
  - [7.3 La generazione: temperature e top-k](#sec-7-3)
  - [7.4 Il checkpoint autosufficiente](#sec-7-4)
  - [7.5 RonkLM v1: i numeri e il testo](#sec-7-5)
  - [7.6 Glossario Fase 7 / cosa arriva in Fase 8](#sec-7-6)
- [Fase 8 — Training serio, CLI ed esperimenti](#fase-8)
  - [8.0 Da "gira" a "gira bene e si usa"](#sec-8-0)
  - [8.1 La CLI: esperimenti riproducibili dal comando](#sec-8-1)
  - [8.2 Il loop robusto: eval, best-checkpoint, clipping](#sec-8-2)
  - [8.3 Il learning-rate schedule: warmup e cosine decay](#sec-8-3)
  - [8.4 Il laboratorio: cosa compra ogni iperparametro](#sec-8-4)
  - [8.5 Chiusura del Percorso A](#sec-8-5)
- [Fase 9 — Il port a PyTorch, con equivalenza dimostrata](#fase-9)
  - [9.0 Perché adesso, e perché non prima](#sec-9-0)
  - [9.1 La tabella di corrispondenza: PyTorch non ha concetti nuovi](#sec-9-1)
  - [9.2 La trappola della trasposizione](#sec-9-2)
  - [9.3 La prova di equivalenza: i numeri](#sec-9-3)
  - [9.4 Il benchmark: quanto costa capire](#sec-9-4)
  - [9.5 Anatomia della memoria: dove finiscono 43 GB](#sec-9-5)
  - [9.6 Mixed precision e GPU: cosa cambia davvero](#sec-9-6)
  - [9.7 Glossario Fase 9 / cosa arriva in Fase 10](#sec-9-7)
- [Fase 10 — Il tokenizer BPE, scritto a mano](#fase-10)
  - [10.0 Perché il char-level non basta più](#sec-10-0)
  - [10.1 L'algoritmo BPE in una pagina](#sec-10-1)
  - [10.2 Perché partire dai byte](#sec-10-2)
  - [10.3 La pre-tokenizzazione: non fondere attraverso le parole](#sec-10-3)
  - [10.4 Il problema di velocità e come l'abbiamo risolto](#sec-10-4)
  - [10.5 Cosa ha imparato, guardato da vicino](#sec-10-5)
- [Fase 11 — Il corpus grande: da Wikipedia ai token](#fase-11)
  - [11.0 La pipeline in quattro passi](#sec-11-0)
  - [11.1 Guardare i dati: tre bug trovati a occhio](#sec-11-1)
  - [11.2 I filtri di qualità e perché sono severi](#sec-11-2)
  - [11.3 Binarizzazione e memmap](#sec-11-3)

---

<a name="fase-0"></a>
# Fase 0 — Fondamenta: dal testo ai batch

Alla fine di questa fase **non esiste ancora nessun modello**. Ed è giusto così. Un
modello di linguaggio è una funzione che mangia numeri e sputa numeri; prima di
costruirla dobbiamo costruire il *tubo* che porta il testo di Pinocchio dentro quella
funzione nella forma esatta che le serve. La Fase 0 è tutto quel tubo.

---

<a name="sec-0-0"></a>
## 0.0 Il quadro generale: cos'è davvero un LLM

> **📖 Concetto.** Un Language Model (LM) risponde a **una sola domanda**, ripetuta
> all'infinito: *"dato il testo visto finora, quanto è probabile ciascun possibile
> prossimo pezzo di testo?"*

Sembra poco, ma da questa unica capacità nasce tutto. Facciamo un esempio concreto
con l'italiano. Se il testo finora è:

```
Il gatto è sul tett_
```

un buon modello italiano, alla posizione del trattino basso, produce qualcosa come:

```
'o' -> 91%      'i' -> 4%       'a' -> 3%      'e' -> 1%     ...     'q' -> 0.0001%
```

Cioè una **distribuzione di probabilità** su *tutti* i possibili prossimi caratteri:
tanti numeri positivi che sommano a 1. Il modello non "sa" l'italiano nel senso in
cui lo sai tu; ha soltanto imparato, da tantissimo testo, che dopo `tett` la lettera
`o` è quasi certa.

Da questa capacità derivano due cose, e sono i due usi di *ogni* LLM esistente:

1. **Generare testo.** Se so dire quanto è probabile ogni prossimo carattere, posso
   *estrarne uno a caso* rispettando quelle probabilità, aggiungerlo al testo, e
   ripetere. Un carattere alla volta. Questo si chiama **generazione
   autoregressiva** ("auto" = da sé, "regressiva" = si rialimenta col proprio
   output) ed è *esattamente* ciò che fa ChatGPT quando scrive: predice il prossimo
   pezzo, lo aggiunge, ripredice.

2. **Misurarsi.** Se il modello assegna probabilità *alta* al testo che è davvero
   accaduto, è un buon modello. Questo ci darà (in Fase 1) un **numero** che misura
   la bravura senza bisogno di giudizio umano — ed è il numero che il training userà
   per migliorarsi.

> **Il filo rosso di tutto il progetto.** Costruiremo una scala di modelli sempre
> più bravi a rispondere a quell'unica domanda. Il bigram (Fase 1) la risponde
> guardando *solo l'ultimo carattere*. L'MLP (Fase 4) guarda gli ultimi N. Il GPT
> (Fase 7) guarda tutto il contesto e decide *da solo* quali parti del passato
> contano. Ma la domanda non cambia mai. Tienilo a mente: quando arriveremo
> all'attention e sembrerà magia, sarà sempre e solo questo.

E la Fase 0? La Fase 0 costruisce il modo di **dare al modello il testo** e di
**dirgli qual era la risposta giusta**, così che possa imparare. Vediamo come.

---

<a name="sec-0-1"></a>
## 0.1 Perché un modello vede numeri, non lettere

Un modello di linguaggio, sotto sotto, è **aritmetica**: moltiplicazioni di matrici,
somme, poche funzioni non lineari. L'aritmetica lavora su numeri, non su lettere. La
lettera `g` non si può moltiplicare per 0.7. Quindi il primissimo problema, prima di
qualsiasi rete neurale, è: **come trasformo del testo in numeri, e i numeri di
ritorno in testo?**

Questa traduzione ha un nome: **tokenizzazione**. Il traduttore si chiama
**tokenizer**. Un *token* è l'unità atomica di testo che il modello tratta come un
singolo simbolo. Nel nostro progetto, per scelta (spiegata tra poco), **un token = un
carattere**.

Il tokenizer fa due operazioni inverse:

- **encode**: testo → lista di numeri interi. `"ciao"` → `[28, 45, 32, 42]`
- **decode**: lista di numeri interi → testo. `[28, 45, 32, 42]` → `"ciao"`

I numeri non hanno alcun significato "matematico" (il fatto che `c`=28 e `d`=29 non
vuol dire che `d` sia "uno più di" `c`): sono solo **etichette**, indirizzi in una
rubrica. La rubrica si chiama **vocabolario**.

---

<a name="sec-0-2"></a>
## 0.2 Caratteri, Unicode e UTF-8: cosa sono davvero i "simboli"

Prima di parlare del nostro tokenizer, chiariamo cosa intendiamo per "carattere",
perché nasconde una sottigliezza che ci ha morso (vedi [trappole](#sec-0-12)).

> **📖 Concetto: Unicode.** Il mondo si è messo d'accordo su una tabella gigantesca
> che assegna a ogni simbolo di ogni lingua un numero univoco, detto **codepoint**.
> `a` è il codepoint U+0061 (97 in decimale), `à` è U+00E0 (224), il trattino lungo
> `—` è U+2014 (8212), l'emoji 🍕 è U+1F355. Unicode è la *tabella*: dice quale
> numero corrisponde a quale simbolo, punto.

> **📖 Concetto: UTF-8.** Unicode dice *quale numero*, ma non *come scriverlo su
> disco in byte*. Questo lo dice una **codifica**. UTF-8 è la codifica dominante:
> rappresenta i codepoint bassi (ASCII: lettere inglesi, cifre, punteggiatura) con
> **1 byte**, e quelli più alti (lettere accentate, `—`, emoji) con **2, 3 o 4
> byte**. Per questo `à` "pesa" più di `a`: due byte contro uno.

Perché ci importa? Perché quando leggiamo un file dobbiamo dire a Python *con quale
codifica* interpretarlo. Se sbagliamo, `à` diventa due caratteri strani (`Ã ` o
simili). In tutto il progetto leggiamo e scriviamo **sempre in UTF-8 esplicito**
(`encoding="utf-8"`), così `à`, `«`, `—` restano un carattere solo, quello giusto.

Nel nostro `CharTokenizer`, un "carattere" è **un codepoint Python** (`str` di
lunghezza 1). Il nostro corpus contiene 69 codepoint distinti — il nostro
vocabolario.

> **⚠️ Nota di onestà.** "Un codepoint = un carattere" non è vero al 100% nel mondo
> reale (alcuni simboli visibili sono composti da più codepoint, es. certe emoji con
> tono della pelle). Ma nel testo di un romanzo italiano dell'800 la coincidenza è
> totale, quindi per noi la semplificazione è perfetta e la sfruttiamo senza sensi
> di colpa.

---

<a name="sec-0-3"></a>
## 0.3 La tokenizzazione a caratteri (il `CharTokenizer`)

📁 File: [`ronklm/tokenizer.py`](ronklm/tokenizer.py)

### 0.3.1 Perché a *caratteri* e non a parole o "pezzi di parola"

Esistono tre grandi famiglie di tokenizer:

| Famiglia | Un token è… | Esempio su `"mangiando"` | Vocabolario |
|---|---|---|---|
| **char-level** (la nostra) | un carattere | `m a n g i a n d o` | piccolo (~decine/centinaia) |
| **word-level** | una parola | `mangiando` | enorme (centinaia di migliaia) |
| **subword / BPE** (i GPT veri) | un pezzo frequente | `mangi` + `ando` | medio (decine di migliaia) |

Abbiamo scelto **char-level** per il Percorso A, per ragioni che valgono la pena:

- **Il tokenizer smette di essere un problema.** Il BPE (che vedremo scritto a mano
  in Fase 10) è un algoritmo interessante ma *ortogonale* a come funziona la rete
  neurale: è pre-processing statistico. Metterlo ora ci farebbe spendere una fase
  intera su un tema che non tocca gradienti né attention. Il char-level si scrive in
  ~20 righe (due dizionari) e ci leva il pensiero.

- **Vocabolario piccolo = modello piccolo = addestrabile su CPU.** L'ultimo strato
  di ogni LM produce un punteggio *per ogni token del vocabolario*. Con ~69 token
  quello strato è minuscolo; con 50.000 token (BPE dei GPT veri) sarebbe da solo più
  grande di tutto il nostro modellino.

- **Si *vede* il modello imparare.** A livello di carattere osserveremo la macchina
  progredire per gradi: prima spazzatura casuale, poi rispetta le frequenze delle
  lettere italiane, poi inventa parole pronunciabili, poi parole vere, poi sintassi.
  È oro didattico: colleghi un numero che scende (la loss) a qualcosa che leggi.

- **Il costo:** le sequenze diventano lunghe (una frase = ~60 caratteri, non ~12
  pezzi-di-parola), quindi a parità di finestra il modello "vede" meno testo. Per
  capire è un costo trascurabile.

### 0.3.2 Come funziona, dentro

Il cuore del `CharTokenizer` sono **due dizionari** costruiti dal vocabolario:

```
chars = ['\n', ' ', '!', "'", ..., 'a', 'b', 'c', ..., '—']   # 69 caratteri, ORDINATI
stoi  = {'\n':0, ' ':1, '!':2, ..., 'a':30, ...}   # string-to-int: carattere -> indice
itos  = {0:'\n', 1:' ', 2:'!', ..., 30:'a', ...}   # int-to-string: indice -> carattere
```

- **encode** guarda ogni carattere nel dizionario `stoi` e restituisce il suo
  indice. `"ba"` → `[stoi['b'], stoi['a']]` → `[31, 30]`.
- **decode** fa l'inverso con `itos`: `[31, 30]` → `"ba"`.

> **📖 Concetto: perché il dizionario è velocissimo.** Un dizionario Python (una
> *hash map*) trova il valore associato a una chiave in **tempo costante**: non
> scorre la lista dei 69 caratteri per trovare quello giusto, calcola direttamente
> "dove" sta. Quindi codificare un milione di caratteri costa un milione di
> operazioni costanti — lineare, velocissimo. È il motivo per cui usiamo due
> dizionari e non, poniamo, una ricerca in una lista.

### 0.3.3 Le due scelte che sembrano dettagli ma non lo sono

**(1) Il vocabolario è ordinato (`sorted`).** Perché? Perché così è
**deterministico**: lo stesso corpus produce *sempre* lo stesso vocabolario, con gli
stessi indici. `a` sarà sempre l'indice 30, in ogni esecuzione, su ogni computer. Se
non ordinassimo (usando l'ordine "a caso" in cui i caratteri appaiono nel testo o in
un `set`), due esecuzioni potrebbero assegnare indici diversi. E allora un modello
addestrato oggi, salvato, e ricaricato domani, troverebbe che l'indice 30 ora è `q`
invece di `a`: **produrrebbe spazzatura**, pur avendo i pesi "giusti". Ordinare è la
polizza assicurativa che rende pesi e vocabolario compatibili nel tempo.

**(2) Fallisce rumorosamente (`fail-loud`).** Se chiedi a `encode` di codificare un
carattere che non è nel vocabolario (poniamo una `k`, che in Pinocchio non c'è), non
inventa un numero e non mette un valore di ripiego: **solleva un errore esplicito**.

> **📖 Concetto: perché "fallire rumorosamente" è una virtù.** In programmazione il
> bug peggiore non è quello che fa crashare: è quello che *non si vede* e produce
> risultati sottilmente sbagliati per settimane. Un carattere sconosciuto mappato
> silenziosamente a un indice qualsiasi avvelenerebbe i dati senza un solo segnale
> d'allarme. Meglio un errore forte e chiaro *subito*, nel punto esatto del
> problema, che un disastro silenzioso a valle. Questa filosofia ("meglio rompersi
> forte che sbagliare piano") tornerà ovunque nel progetto.

### 0.3.4 Salvare e ricaricare il vocabolario

Abbiamo aggiunto `save()`/`load()` (in formato JSON) già ora, con un anticipo
strategico: in Fase 7, quando salveremo un modello addestrato, il file dovrà
contenere *anche* il vocabolario. Pesi e mappa degli indici devono viaggiare
**sempre insieme**, per la stessa ragione del punto (1): separarli produce
spazzatura deterministica.

---

<a name="sec-0-4"></a>
## 0.4 Il corpus: da dove viene e come l'abbiamo ripulito

📁 File: [`data/prepare_corpus.py`](data/prepare_corpus.py) → produce `data/input.txt`

### 0.4.1 La scelta del testo

Serviva un corpus (1) in italiano, (2) di dimensione giusta — abbastanza grande da
non essere memorizzato subito, abbastanza piccolo da addestrarci su CPU: ideale
~0.3–1.5 MB —, (3) di pubblico dominio, (4) di stile omogeneo (un solo autore rende i
risultati di un modello piccolo molto più riconoscibili).

Scelta: ***Le avventure di Pinocchio* di Carlo Collodi** (1883). Lessico ricco ma
ripetitivo il giusto (Geppetto, la Fata, il Grillo ricorrono e il modello li
imparerà in modo visibile), tanti dialoghi (impara la struttura `— disse…`), prosa
ottocentesca ma semplice. ~240 KB dopo la pulizia: perfetto.

### 0.4.2 Perché uno *script* e non un file scaricato a mano

Il file `input.txt` non lo abbiamo creato a mano: lo produce lo script
`prepare_corpus.py`. La ragione è **la riproducibilità**: lo script documenta
*esattamente* da dove vengono i dati e come sono stati trasformati. Se domani il file
si corrompe, o vogliamo cambiare edizione, la provenienza non è persa. Nei progetti
di machine learning veri, "da dove vengono e come sono stati puliti i dati" è la
prima cosa che si perde e la più costosa da ricostruire.

### 0.4.3 Isolare la storia dal contorno

Il file scaricato da Project Gutenberg (ebook **#52484**) contiene, oltre alla
storia: un frontespizio (titolo, editore, anno), un indice dei capitoli (in fondo,
con numeri di pagina) e note del trascrittore. Tutto questo **non è prosa di
Collodi**: è contorno editoriale che, se lasciato, insegnerebbe al modello cose
sbagliate (che dopo "Firenze" viene una tabella di numeri di pagina, per esempio).

Lo ritagliamo con **ancore testuali** invece che con posizioni numeriche:

```python
STORY_START = "I.\n\nCome andò che Maestro Ciliegia"   # inizio del Capitolo I
STORY_END   = "FINE."                                   # chiusura della storia
```

> **📖 Concetto: perché ancore e non offset.** Un *offset* numerico ("parti dal
> carattere 840") si rompe alla prima ri-edizione del file: basta un byte in più
> all'inizio e tutto scala. Un'*ancora di contenuto* ("parti da dove trovi questa
> frase") è robusta: la frase resta la frase anche se il resto cambia. È una piccola
> lezione di ingegneria che vale ben oltre questo progetto.

### 0.4.4 La normalizzazione: togliere rumore senza toccare la lingua

> **📖 Concetto.** *Normalizzare* significa ridurre le varianti equivalenti di uno
> stesso simbolo a una forma sola. Perché conta per un modello? Perché **ogni
> variante è un token in più nel vocabolario**, e ogni token in più diluisce le
> statistiche: se metà degli apostrofi fosse "dritta" (`'`) e metà "curva" (`'`), il
> modello dovrebbe imparare *due volte* la stessa regola. Concentrare le varianti
> concentra il segnale.

Nel nostro caso il testo era già quasi pulito (apostrofi e puntini già ASCII).
Abbiamo preso queste decisioni, tenendo ciò che è lingua e togliendo ciò che è
rumore:

| Elemento | Occorrenze | Decisione | Perché |
|---|---|---|---|
| `—` (trattino lungo) | 1929 | **tenuto** | è il marcatore dei dialoghi italiani: il modello *deve* impararlo |
| `à è ì ò ù È` | ~2900 | **tenuti** | sono la lingua italiana |
| `« »` (caporali) | 100 | **tenuti** | virgolette italiane legittime |
| a-capo, maiuscole | — | **tenuti** | il modello imparerà "dopo il punto, maiuscola" e i paragrafi |
| `\xa0` (spazio unificatore) | 214 | → spazio normale | è uno spazio "invisibile diverso": sarebbe un token-fantasma |
| `[Illustrazione: …]` | 79 blocchi | **rimossi** | didascalie editoriali, non prosa: spezzano la frase |
| `_corsivo_` (underscore) | 76 | underscore rimossi | marcatore di corsivo di Gutenberg: la parola è vera, il segno no |
| `" „` | 2 | → `"` ASCII | 2 refusi isolati: sarebbero 2 token-fantasma |

Abbiamo anche tolto gli spazi a fine riga e collassato 3+ righe vuote consecutive in
una sola: la struttura dei paragrafi resta, ma senza buchi giganti.

> **⚠️ Come le abbiamo scoperte.** Non abbiamo indovinato: dopo una prima passata
> abbiamo *stampato l'elenco completo del vocabolario e i contesti* dei caratteri
> sospetti (`[`, `]`, `_`, `"`). Vedere `[Illustrazione: Un vecchietto tutto…]` in
> mezzo a una frase ha reso ovvio che erano didascalie da togliere. **Guardare i
> dati con i propri occhi** prima di darli in pasto a un modello è una delle
> abitudini più preziose del mestiere.

Risultato finale: **240.920 caratteri, vocabolario di 69 simboli**, tutto LF puro.

---

<a name="sec-0-5"></a>
## 0.5 Anatomia del vocabolario e delle frequenze

I 69 caratteri del nostro vocabolario:

```
\n  (spazio)  !  '  (  )  ,  -  .  1  4  8  :  ;  ?
A B C D E F G H I J L M N O P Q R S T U V X Z
a b c d e f g h i j l m n o p q r s t u v z
à è ì ò ù È  « »  —
```

Un paio di osservazioni istruttive:

- **Mancano K, W, Y, k, w, x, y.** Non sono un errore: semplicemente non compaiono
  nel testo italiano di Collodi. Il vocabolario è *quello del corpus*, non
  dell'alfabeto astratto.
- **Le cifre presenti sono solo `1 4 8`** (da "avevano 14 anni", "ne avevano 8",
  "alle ore 11"). Un insieme di cifre incompleto è un po' strano a vedersi, ma è
  onesto: sono le uniche presenti. Il modello le tratterà come caratteri rari.

### Le frequenze non sono uniformi — e questo è il punto

Ecco i caratteri più frequenti nel nostro corpus:

```
spazio  37.804      a  20.844      i  19.194      e  19.029      o  18.233
n       12.199      r  11.456      t  10.954      l  10.262      c   9.477
```

Lo spazio da solo è il 16% del testo. Le vocali dominano. La `q` compare pochissimo,
sempre seguita da `u`. Questa **disparità** è esattamente ciò che un language model
sfrutta.

> **📖 Concetto (anteprima di Fase 1): informazione e sorpresa.** Se ogni carattere
> fosse equiprobabile (1/69 ciascuno), non ci sarebbe niente da predire: puro caso.
> Ma non è così: dopo `q` arriva quasi sempre `u`, dopo uno spazio è probabile una
> consonante, ecc. **Predire bene = sfruttare queste regolarità.** In Fase 1
> misureremo quanto un modello è bravo con un numero legato alla "sorpresa media per
> carattere": bassa sorpresa (il modello sapeva cosa veniva) = buon modello. Questo
> numero si chiama *cross-entropy*, e la disparità delle frequenze che vedi qui è la
> materia prima che lo rende migliore del puro caso. Il primo modello, il bigram,
> farà esattamente questo: contare "dopo X, cosa viene?".

---

<a name="sec-0-6"></a>
## 0.6 NumPy e i tensori: il minimo indispensabile

📁 Usato in: [`ronklm/dataset.py`](ronklm/dataset.py)

Tutto il calcolo del progetto (dal Percorso A) poggia su **NumPy**. Vale la pena
capire i tre concetti che useremo ovunque.

> **📖 Concetto: il tensore (`ndarray`).** Un *tensore* è una griglia di numeri a N
> dimensioni. Un singolo numero è 0-D (uno *scalare*); una fila di numeri è 1-D (un
> *vettore*); una tabella è 2-D (una *matrice*); un cubo è 3-D; e così via. In NumPy
> si chiama `ndarray`. Il nostro corpus codificato è un vettore 1-D di 240.920
> interi; un batch sarà una matrice 2-D.

> **📖 Concetto: shape.** La *shape* sono le dimensioni del tensore. Una matrice di
> 4 righe e 32 colonne ha shape `(4, 32)`. **Il debugging delle reti neurali è per
> l'80% ragionamento sulle shape**: "questo tensore ha la forma giusta per essere
> moltiplicato con quello?". Prendi confidenza ora, ci accompagnerà fino alla fine.

> **📖 Concetto: dtype.** Il *tipo* dei numeri dentro il tensore. Noi usiamo
> `int64`: interi con segno da 64 bit. Per 69 valori è sovrabbondante (basterebbe 1
> byte), ma è il tipo naturale per *indicizzare* (che è ciò che gli indici dei
> caratteri fanno) e la memoria sprecata a questa scala è irrilevante. In Fase 11,
> col corpus da gigabyte, passeremo a `uint16` proprio per non sprecare.

> **📖 Concetto: perché NumPy e non liste Python.** Una lista Python di numeri è
> flessibile ma lenta: ogni numero è un oggetto separato in memoria. Un `ndarray`
> tiene i numeri **compatti e contigui**, come una fila ordinata di caselle, e le
> operazioni su di essi sono scritte in C e ottimizzate. Moltiplicare due matrici
> 1000×1000 in NumPy è migliaia di volte più veloce che farlo con cicli Python. È il
> motore aritmetico su cui costruiremo tutto.

---

<a name="sec-0-7"></a>
## 0.7 Il dataset: perché si divide in train e validation

📁 File: [`ronklm/dataset.py`](ronklm/dataset.py) → classe `Dataset`

La classe `Dataset` prende il testo, lo **codifica una volta sola** in un vettore di
interi (via il tokenizer), e lo divide in due parti:

- **train** (90%): il testo su cui il modello si allenerà;
- **validation** (10%, o "val"): un pezzo tenuto *nascosto* dal training.

Perché tenerne una parte nascosta? Per una ragione profonda che è la malattia numero
uno del machine learning.

> **📖 Concetto: overfitting (memorizzare invece di imparare).** Il modello vedrà il
> testo di training migliaia di volte. Un modello con abbastanza "capacità" (tanti
> parametri) può, invece di imparare *regole generali* dell'italiano,
> semplicemente **memorizzare** il testo che ha visto — come uno studente che impara
> a pappagallo il libro senza capirlo. Sul testo di training sembrerebbe bravissimo,
> ma sarebbe inutile su testo nuovo.

Il validation set è la difesa contro questo inganno. Misuriamo la bravura del modello
sul val (testo *mai visto durante il training*): quel numero risponde alla domanda
giusta — "ha imparato l'italiano, o ha imparato Pinocchio a memoria?". Quando la
bravura su train continua a salire ma quella su val si ferma o peggiora, il modello
sta memorizzando. **La sola metrica di cui ci fideremo, in tutto il progetto, è
quella di validation.**

> **📖 Concetto: perché lo split è contiguo (l'ultimo 10%) e non a caratteri
> sparsi.** Il testo è sequenziale. Se prendessimo caratteri a caso qua e là per il
> val, ognuno avrebbe i suoi vicini nel train, e la "verifica su testo mai visto"
> sarebbe una finzione: il modello avrebbe di fatto già visto il contorno di ogni
> pezzo di val. Un blocco *contiguo* tenuto da parte (l'ultima fetta del libro) è
> testo davvero non visto. Questo principio — "il set di test non deve contaminarsi
> col training" — è così importante che tornerà, ingigantito, in Fase 11 con la
> deduplicazione del corpus grande.

---

<a name="sec-0-8"></a>
## 0.8 Il trucco X/Y: T esempi al prezzo di uno

Questo è il concetto più bello della Fase 0, e il motore segreto dell'efficienza dei
transformer. Va capito bene.

Un esempio di addestramento, per un language model, è una coppia:

```
(contesto, carattere-che-viene-dopo)
```

cioè "dato questo pezzo di testo, il prossimo carattere giusto è questo". Ora, il
trucco: da una finestra di `block_size + 1` caratteri consecutivi si estraggono
**`block_size` esempi tutti insieme**. Vediamolo con la finestra `"ciao m"` e
`block_size = 5`:

```
finestra:   c   i   a   o  (spazio)  m

X = "ciao "      (i primi 5)
Y = "iao m"      (gli stessi, spostati di 1 a sinistra)

posizione 0:  visto "c"        ->  predici "i"
posizione 1:  visto "ci"       ->  predici "a"
posizione 2:  visto "cia"      ->  predici "o"
posizione 3:  visto "ciao"     ->  predici " "
posizione 4:  visto "ciao "    ->  predici "m"
```

Guarda la relazione tra `X` e `Y`: **`Y` è semplicemente `X` spostato di un
carattere**. `Y[t]` è il carattere che segue `X[t]`. Ogni posizione della finestra è
un esempio di addestramento indipendente, con un contesto di lunghezza diversa (1
carattere, 2, 3, …).

Nel codice questa relazione è realizzata così:

```python
X = data[i     : i + block_size]        # la finestra
Y = data[i + 1 : i + 1 + block_size]    # la stessa, spostata di 1
```

E l'abbiamo bloccata con un test, come invariante matematica:

```python
X[:, 1:] == Y[:, :-1]     # la coda di X coincide con la testa di Y
```

> **Perché questo è enorme.** Una finestra da 256 caratteri dà **256 predizioni da
> imparare in un colpo solo**, non una. Quando in Fase 5 introdurremo la *maschera
> causale* nell'attention, potremo calcolare tutte queste predizioni in *un solo
> passaggio* della rete, garantendo al tempo stesso che la predizione alla posizione
> `t` non "sbirci" i caratteri futuri. Questo è il motivo per cui i transformer, pur
> enormi, si addestrano in tempi umani: ogni singolo passaggio insegna moltissime
> cose contemporaneamente. La Fase 0 pianta il seme; la Fase 7 raccoglie.

---

<a name="sec-0-9"></a>
## 0.9 Batch, casualità e riproducibilità (PRNG e seed)

### 0.9.1 Cos'è un batch e perché

Non diamo al modello un esempio alla volta né tutto il corpus in una volta: gli diamo
un **batch** — un gruppetto di finestre (per esempio 32 o 64) processate insieme. Il
metodo `get_batch` estrae `batch_size` finestre da posizioni **casuali** del testo e
le impila in una matrice `(batch_size, block_size)`.

Perché posizioni casuali e non in ordine? Perché l'addestramento (lo vedremo in Fase
2) migliora i pesi un batch alla volta, e funziona meglio quando gli esempi di un
batch sono **il più possibile scorrelati** tra loro. Finestre consecutive del libro
sarebbero quasi identiche (stessa scena, stesse parole): il modello finirebbe per
"seguire la trama" invece di imparare la lingua. Pescare a caso da tutto il corpus dà
a ogni batch un campione vario.

### 0.9.2 Cos'è un PRNG e perché il seed rende tutto riproducibile

> **📖 Concetto: numeri pseudo-casuali.** Un computer non sa generare vero caso.
> Genera numeri *pseudo*-casuali: una sequenza prodotta da una formula
> deterministica che *sembra* casuale (supera i test statistici di casualità) ma è
> in realtà completamente determinata da un valore iniziale, il **seed** (seme). Il
> componente che li produce è un **PRNG** (Pseudo-Random Number Generator).

Il fatto cruciale: **stesso seed → stessa identica sequenza di numeri**. Questo ci dà
una proprietà d'oro, la **riproducibilità**. Nel nostro codice, `get_batch` non usa
il caso "globale": riceve dall'esterno un generatore creato con un seed esplicito:

```python
rng = np.random.default_rng(42)   # seed = 42
X, Y = ds.get_batch("train", block_size=32, batch_size=8, rng=rng)
```

Con lo stesso seed, otterrai *sempre* lo stesso batch. Perché è importante?

- **Debugging:** se un training si comporta male, puoi rieseguirlo *identico* e
  isolare il problema. Senza riproducibilità, ogni esecuzione è diversa e i bug sono
  fantasmi.
- **Confronti onesti:** per confrontare due modelli devi allenarli sugli stessi dati
  nello stesso ordine. Il seed lo garantisce.

Lo abbiamo persino verificato con due test: stesso seed → batch identico; seed diverso
→ batch diverso. È una regola non negoziabile del progetto: **niente caso globale,
sempre un generatore con seed esplicito.**

### 0.9.3 Un dettaglio da non sbagliare: i bordi

Quando peschiamo una posizione di partenza `i` a caso, dobbiamo garantire che
`i + block_size` esista ancora nel testo (perché `Y` ha bisogno di un carattere in
più di `X`). Nel codice:

```python
ix = rng.integers(0, len(data) - block_size, size=batch_size)
```

`integers(0, N)` estrae numeri in `[0, N)` (estremo destro **escluso**). Con
`N = len(data) - block_size`, la partenza massima possibile fa sì che
`i + block_size` sia al più l'ultimo indice valido: **mai fuori dai bordi**. Sono i
dettagli come questo, un `-block_size` messo o dimenticato, che fanno la differenza
tra codice che gira e codice che crasha a un batch a caso dopo mezz'ora.

---

<a name="sec-0-10"></a>
## 0.10 La disciplina dei test

📁 File: [`tests/`](tests/), eseguibili con `python run_tests.py`

Abbiamo scritto **16 test** già in Fase 0, prima ancora di avere un modello. Non è
pignoleria: è la fondamenta della fiducia.

> **📖 Concetto: perché testare, e testare *presto*.** Il `CharTokenizer` e il
> `Dataset` sono il **contratto** tra il testo e ogni modello futuro: *ogni*
> esperimento di *ogni* fase ci passerà attraverso. Un bug qui non darebbe un errore
> chiaro in Fase 5: darebbe un modello che "impara male", e passeresti giorni a
> incolpare l'architettura mentre la colpa era nei dati. Testare il fondamento
> *adesso* significa che, quando qualcosa andrà storto più avanti, potremo escludere
> con certezza che sia colpa del tubo dei dati.

Ogni test dimostra qualcosa di preciso. Alcuni esempi:

- `test_roundtrip_on_full_corpus`: codificare e poi decodificare l'*intero*
  Pinocchio restituisce il testo identico → il tokenizer non perde informazione.
- `test_Y_is_X_shifted_by_one`: verifica l'invariante `X[:, 1:] == Y[:, :-1]` → il
  trucco della [sezione 0.8](#sec-0-8) è davvero rispettato.
- `test_reproducibility_same_seed_same_batch`: stesso seed → stesso batch → la
  riproducibilità della [sezione 0.9](#sec-0-9) funziona.
- `test_encode_raises_on_unknown_char`: un carattere ignoto solleva un errore → il
  `fail-loud` è attivo.

Nota tecnica: nell'ambiente non c'è `pytest`, quindi abbiamo scritto un mini-runner
(`tests/_runner.py`) che esegue tutte le funzioni `test_*` e riporta il risultato. I
test sono comunque scritti in stile `pytest`, quindi se un giorno lo installeremo
funzioneranno senza modifiche.

---

<a name="sec-0-11"></a>
## 0.11 Git, versioni e i due remote

Il progetto è versionato con **git** dal primo giorno, con una convenzione precisa:
i **branch si chiamano come le versioni** (`v1.0.0`, `v1.1.0`, …) e la numerazione
avanza a ogni fine fase secondo l'entità delle modifiche (piccola `+0.0.1`, media
`+0.1.0`, grande `+1.0.0`).

Perché tutta questa disciplina su un progetto didattico? Per tracciabilità: a colpo
d'occhio, dalla storia git, si vede cosa è stato fatto e quando, e si può sempre
tornare a uno stato funzionante — una rete di sicurezza preziosa in un progetto dove
"si sbaglia per imparare".

Ogni fine fase viene inoltre pushato su **due remote** (GitHub e un'istanza Gitea
privata), così il lavoro è ridondato su due server indipendenti.

> **⚠️ Trappola git disinnescata (fine-riga).** Vedi la [sezione trappole](#sec-0-12):
> abbiamo aggiunto un `.gitattributes` per impedire a git di convertire i fine-riga
> del corpus, cosa che avrebbe reintrodotto un carattere invisibile nel vocabolario
> su un altro computer.

---

<a name="sec-0-12"></a>
## 0.12 Le tre trappole, spiegate con l'autopsia

Le trappole in cui siamo inciampati sono materiale didattico prezioso: raccontano
*come si sbaglia davvero*. Le registriamo con la causa tecnica per non farci mordere
due volte.

### Trappola 1 — L'ebook sbagliato

Cercando "Pinocchio" su Gutenberg, il primo risultato (ebook #19517) aveva il titolo
giusto ma era di soli 23 KB. Aprendolo: era una **lettura audio**, cioè solo un
elenco di capitoli con i timestamp (`# Chapter 01 - 00:05:35`), non il testo. Il
testo integrale era un altro ebook, il **#52484** (256 KB). *Lezione: verifica sempre
che i dati siano ciò che pensi, guardandoli, prima di usarli.*

### Trappola 2 — La doppia conversione dei fine-riga su Windows

Questa è sottile e istruttiva.

> **📖 Concetto: `\n` contro `\r\n`.** I sistemi operativi non concordano su come si
> va a capo in un file di testo. Unix/Linux/Mac usano un carattere solo, "line feed"
> (`\n`). Windows storicamente ne usa due, "carriage return + line feed" (`\r\n`),
> un'eredità delle macchine da scrivere. Questa differenza è la fonte di infiniti
> bug cross-platform.

Cosa è successo: quando salvavamo la cache del download con la funzione "scrivi
testo" di Python, su Windows ogni `\n` veniva **automaticamente** trasformato in
`\r\n`. Ma il file scaricato aveva *già* `\r\n`, che diventava così `\r\r\n`.
Rileggendolo, questa sequenza corrotta faceva sballare le nostre ancore testuali
(`"I.\n\nCome andò…"` non combaciava più), e lo script si fermava con "ancora non
trovata".

La cura: leggere e scrivere la cache **in binario** (`read_bytes`/`write_bytes`), che
non applica nessuna conversione magica dei fine-riga. Idem per il file finale
`input.txt`, così ciò che sta su disco combacia *al byte* con ciò che abbiamo
analizzato.

### Trappola 3 — Gli artefatti nel vocabolario, e il `.gitattributes`

Alla prima passata, il vocabolario conteneva `[`, `]`, `_`, `"`. Ispezionando i
contesti abbiamo scoperto che venivano da `[Illustrazione: …]` (didascalie) e
`_corsivo_` (marcatori di corsivo di Gutenberg). Li abbiamo rimossi nella
normalizzazione (vedi [0.4.4](#sec-0-4)): il vocabolario è sceso a 69.

Ma c'era un colpo di coda: **git**, su Windows, poteva riconvertire `input.txt` a
`\r\n` al momento del checkout su un altro computer — reintroducendo il carattere
`\r` nel vocabolario (69 → 70) e riaprendo la trappola 2 in un altro modo. La cura
definitiva: un file `.gitattributes` che forza i fine-riga a `\n` (`eol=lf`) ovunque
e per sempre, su ogni piattaforma. L'abbiamo verificato: il file committato non
contiene nemmeno un `\r`.

---

<a name="sec-0-13"></a>
## 0.13 Glossario dei termini introdotti in Fase 0

- **Token**: l'unità atomica di testo per il modello. Per noi, un carattere.
- **Tokenizer**: il traduttore testo ⇄ numeri.
- **Vocabolario (vocab)**: l'insieme dei token possibili; `vocab_size` = quanti sono
  (per noi, 69).
- **encode / decode**: testo → indici / indici → testo.
- **Codepoint**: il numero univoco che Unicode assegna a un simbolo.
- **UTF-8**: la codifica che scrive i codepoint in byte (1–4 byte l'uno).
- **Tensore / `ndarray`**: griglia di numeri a N dimensioni (NumPy).
- **Shape**: le dimensioni di un tensore, es. `(4, 32)`.
- **dtype**: il tipo dei numeri in un tensore, es. `int64`.
- **Train / Validation split**: la parte di dati per allenarsi / la parte nascosta
  per misurare la generalizzazione.
- **Overfitting**: quando il modello memorizza il training invece di generalizzare.
- **Batch**: un gruppo di esempi processati insieme.
- **block_size**: la lunghezza della finestra di contesto (in caratteri).
- **X / Y**: contesto / target; `Y` è `X` spostato di un carattere.
- **PRNG**: generatore di numeri pseudo-casuali.
- **Seed**: il valore iniziale che determina la sequenza di un PRNG → riproducibilità.
- **fail-loud**: filosofia di sollevare errori espliciti invece di sbagliare in
  silenzio.
- **`\n` / `\r\n`**: i due modi (Unix / Windows) di andare a capo in un file.

---

<a name="sec-0-14"></a>
## 0.14 Cosa sappiamo fare ora, e cosa arriva in Fase 1

**Ora** possiamo prendere Pinocchio e produrre, a comando, batch di tensori interi
`(X, Y)` pronti per essere dati in pasto a qualsiasi modello, con la garanzia che `Y`
è la risposta giusta per `X` e che tutto è riproducibile. Il *tubo dei dati* è
completo, testato e documentato. **Non esiste ancora nessun modello** — ed è
esattamente il punto: prima si capisce cosa mangia la macchina.

**In Fase 1** costruiremo il primo, semplicissimo language model: il **bigram per
conteggio**. Guarderà *solo l'ultimo carattere* e predirà il prossimo basandosi su
una semplice tabella: "nel corpus, dopo `q`, quante volte è venuta `u`? e `a`? e
`z`?". Nessun gradiente, nessun addestramento: solo conteggi. Ma con esso nasceranno
i tre concetti che reggono tutto il resto: la **distribuzione di probabilità sul
prossimo token**, il **campionamento** (generare testo), e soprattutto la **loss** —
il numero, anticipato in [0.5](#sec-0-5), che misura la "sorpresa media" e che diventerà
la bussola di ogni addestramento futuro.

---

<a name="fase-1"></a>
# Fase 1 — Il bigram: il primo language model

Finalmente costruiamo un modello. Il più semplice che esista: guarda **solo l'ultimo
carattere** e predice il prossimo. E lo fa **contando**, non addestrando — nessun
gradiente, nessuna rete. Perché partire da qualcosa di così stupido? Perché ci
permette di incontrare, in isolamento e nella loro forma più pura, i tre concetti su
cui poggia *tutto* il resto: la **distribuzione sul prossimo carattere**, il
**campionamento** (generare testo) e soprattutto la **loss** (la misura di bravura).
Separare i concetti dai meccanismi (il training verrà in Fase 2) è il modo di capire
davvero entrambi.

📁 File: [`ronklm/models/bigram_count.py`](ronklm/models/bigram_count.py)

---

<a name="sec-1-0"></a>
## 1.0 L'idea: predire guardando un solo carattere

"Bigram" significa "coppia di caratteri". L'assunzione del modello è brutalmente
semplice:

> *La probabilità del prossimo carattere dipende **solo** dal carattere corrente,
> non da tutto ciò che c'è prima.*

Cioè: per predire cosa viene dopo la `q`, il bigram guarda *solo* la `q` e ignora
tutto il resto della frase. È un'assunzione falsa (il contesto conta eccome!), ma è
un punto di partenza onesto e sorprendentemente informativo — l'italiano ha
regolarità locali fortissime (dopo `q` viene quasi sempre `u`) che perfino questo
modellino cattura.

Come impara queste probabilità? Nel modo più diretto immaginabile: **le conta** nel
testo. "Nel corpus, dopo `q`, quante volte è venuta `u`? E `a`? E `z`?"

---

<a name="sec-1-1"></a>
## 1.1 La matrice dei conteggi

Il cuore del modello è una tabella quadrata `N` di dimensione `(vocab × vocab)`,
cioè 69×69 nel nostro caso. La regola:

```
N[i, j] = quante volte, in tutto il testo di training, al carattere i è seguito j
```

La riga `N[i]` è quindi la "fotografia" di cosa viene dopo il carattere `i`. La riga
della `q` avrà un picco enorme sulla colonna della `u` e quasi zero altrove.

Nel codice la costruiamo con una singola operazione vettorizzata:

```python
a = data[:-1]   # tutti i caratteri "da"  (tutti tranne l'ultimo)
b = data[1:]    # tutti i caratteri "a"   (tutti tranne il primo)
np.add.at(self.N, (a, b), 1)
```

> **🔧 Nel codice: cos'è `np.add.at`.** `a` e `b` sono due lunghi vettori allineati:
> `(a[k], b[k])` è la k-esima coppia consecutiva del testo. `np.add.at(N, (a, b), 1)`
> fa `N[a[k], b[k]] += 1` per ogni `k`, ma tutto in C, in un colpo solo. Perché non
> un normale `N[a, b] += 1`? Perché quella forma "ingenua" sbaglia quando la stessa
> coppia compare più volte (un difetto sottile di NumPy con indici ripetuti):
> conterebbe una volta sola. `np.add.at` è la versione che accumula correttamente
> gli indici ripetuti. Contare 240.000 coppie così è istantaneo.

> **📖 Concetto: perché il conteggio non scala (e perché ci servono le reti).**
> Funziona benissimo per 1 carattere di contesto: la tabella ha 69×69 ≈ 4.700 celle.
> Ma se volessimo guardare 2 caratteri di contesto, servirebbe una tabella 69×69×69;
> per 10 caratteri, 69¹⁰ celle — **più delle stelle nell'universo osservabile**. Il
> conteggio esplode. Le reti neurali sono, in un certo senso, il modo di
> *comprimere* questa tabella impossibile in una funzione con pochi parametri che la
> *approssima*. Questa frase è metà del senso del deep learning: tienila da parte,
> tornerà in Fase 4 quando gli embedding faranno esattamente questa compressione.

---

<a name="sec-1-2"></a>
## 1.2 Da conteggi a probabilità: normalizzare e lo smoothing

I conteggi grezzi non sono ancora probabilità. Per trasformarli, **dividiamo ogni
riga per la sua somma**: così ogni riga diventa una distribuzione che somma a 1.

```python
smoothed = self.N.astype(np.float64) + smoothing         # +1 a tutte le celle
self.P = smoothed / smoothed.sum(axis=1, keepdims=True)   # dividi ogni riga per la sua somma
```

La riga `P[i]` risponde finalmente alla domanda-chiave di un LM: *"dato il carattere
`i`, con che probabilità arriva ciascun carattere?"*. È la prima incarnazione
concreta della "domanda unica" della [sezione 0.0](#sec-0-0).

### Perché il `+1` (smoothing di Laplace)

Guarda quel `+ smoothing` (di default 1). Serve a risolvere un problema serio: se una
coppia non compare **mai** nel training, il suo conteggio è 0, e la sua probabilità
sarebbe 0. Ma "mai visto nel campione" ≠ "impossibile". Se poi quella coppia
comparisse nel testo di validation, il modello le assegnerebbe probabilità 0 →
`log(0) = −∞` → **loss infinita**, tutto rotto da un singolo evento raro.

Il `+1` dice: *"fingiamo di aver visto ogni coppia almeno una volta"*. Nessuna
probabilità è più esattamente zero, e la catastrofe è evitata.

> **📖 Concetto: la regolarizzazione, prima apparizione.** Lo smoothing è un esempio
> di un principio eterno del machine learning: **non fidarti ciecamente dei dati
> osservati; le stime vanno "ammorbidite"**. Un campione finito non contiene tutto
> ciò che è possibile; assumere che ciò che non hai visto sia impossibile è
> l'errore. Vedremo la stessa idea, in altre vesti, nel *weight decay* di Fase 4.
> (Curiosità: le reti neurali "ammorbidiscono" da sole, per come è fatta la funzione
> softmax — non producono mai esattamente 0 — e lo noteremo in Fase 2.)

---

<a name="sec-1-3"></a>
## 1.3 Broadcasting: la regola che allinea le shape

In quella riga di normalizzazione c'è un dettaglio NumPy che va capito ora perché
tornerà, cruciale, quando in Fase 3 dovremo calcolare i gradienti *attraverso* di
esso.

```python
smoothed.sum(axis=1, keepdims=True)   # shape (69, 1), non (69,)
```

`smoothed` ha shape `(69, 69)`. Sommando lungo `axis=1` (le colonne) otteniamo una
somma per ogni riga: 69 numeri. Con `keepdims=True` questi 69 numeri hanno shape
`(69, 1)` — una colonna — invece di `(69,)` — una fila.

Poi facciamo `smoothed / somma`, dividendo una matrice `(69, 69)` per una colonna
`(69, 1)`. Come fa NumPy? Con il **broadcasting**.

> **📖 Concetto: il broadcasting.** Quando operi tra due tensori di shape diverse,
> NumPy prova ad "allargarli" a una forma comune replicando *virtualmente* le
> dimensioni di taglia 1. Una colonna `(69, 1)` divisa in una matrice `(69, 69)`:
> NumPy immagina di replicare quella colonna 69 volte in orizzontale, così ogni
> elemento della riga `i` viene diviso per la somma della riga `i`. È esattamente
> ciò che vogliamo: normalizzare riga per riga. **Se avessimo usato `keepdims=False`
> ottenendo shape `(69,)`, il broadcasting avrebbe allineato quei 69 numeri alle
> *colonne* invece che alle *righe*, e avremmo normalizzato nel verso sbagliato** —
> un bug silenzioso classico. Per questo `keepdims=True` è importante.

Il broadcasting è comodissimo in avanti (nel forward), ma in Fase 3 dovremo
insegnare al nostro motore di gradienti a "disfarlo" all'indietro (sommare i
gradienti lungo le dimensioni che erano state replicate). È, ti anticipo, il punto
tecnicamente più insidioso di tutto il progetto. Averlo incontrato qui, in un
contesto semplice, ci prepara.

---

<a name="sec-1-4"></a>
## 1.4 Generare testo: il campionamento autoregressivo

Avere `P` ci permette di *generare*. L'algoritmo (metodo `generate`):

1. Parti da un carattere (noi partiamo da un a-capo `\n`, che spesso inizia una riga).
2. Leggi la sua riga di probabilità `P[carattere_corrente]`.
3. **Estrai** il prossimo carattere a caso, rispettando quelle probabilità.
4. Aggiungilo, e ripeti dal punto 2 col nuovo carattere.

```python
cur = int(rng.choice(self.vocab_size, p=self.P[cur]))
```

> **📖 Concetto: perché si campiona e non si prende il massimo.** Verrebbe la
> tentazione di scegliere sempre il carattere *più* probabile (la scelta "greedy",
> avida). Ma da uno stesso carattere uscirebbe *sempre* la stessa catena, e il testo
> collasserebbe in un ciclo degenere (es. `"e le le le le…"`). Estrarre a caso
> *secondo* la distribuzione mantiene la varietà naturale del linguaggio: l'output è
> diverso a ogni esecuzione ma statisticamente fedele al corpus. Questa tensione tra
> "probabile" (conservativo) e "vario" (creativo) tornerà, con manopole esplicite
> (temperature, top-k), in Fase 7. Qui ne vediamo la forma pura.

`rng.choice(V, p=P[cur])` fa esattamente questo: estrae un indice tra 0 e V−1 con
probabilità date dal vettore `P[cur]`. E, essendo `rng` un generatore con seed
esplicito ([sezione 0.9](#sec-0-9)), la generazione è riproducibile.

---

<a name="sec-1-5"></a>
## 1.5 La loss spiegata a fondo: negative log-likelihood

Ecco il concetto più importante della Fase 1, e forse dell'intero progetto: come si
misura *con un numero* quanto è buono un modello probabilistico. Ci arriviamo per
gradi, perché ogni pezzo della formula ha una ragione.

**Punto di partenza.** Un buon modello assegna probabilità **alta al testo che è
realmente accaduto**. Quindi, come misura di bravura, prendiamo la probabilità che
il modello assegna all'intero testo di validation. La probabilità di una sequenza è
il **prodotto** delle probabilità dei singoli passi (regola della catena della
probabilità):

```
P(testo) = P(c₂|c₁) · P(c₃|c₂) · P(c₄|c₃) · … · P(c_n|c_{n-1})
```

Vorremmo *massimizzare* questo prodotto. Ma ha tre problemi pratici, e li risolviamo
con tre ritocchi.

**Ritocco 1 — il logaritmo.** Quel prodotto è composto da decine di migliaia di
numeri, tutti minori di 1. Il risultato è un numero *microscopico*, così piccolo che
il computer non riesce a rappresentarlo (diventa 0 per arrotondamento:
"underflow"). Applicando il **logaritmo**, il prodotto diventa una **somma**:

```
log P(testo) = log P(c₂|c₁) + log P(c₃|c₂) + …
```

Le somme di numeri gestibili non danno underflow. E siccome il logaritmo è una
funzione *crescente*, massimizzare `log P` equivale a massimizzare `P`: non abbiamo
cambiato il problema, solo reso i conti stabili.

> **📖 Concetto: perché il logaritmo trasforma prodotti in somme.** È la sua
> proprietà fondante: `log(a·b) = log(a) + log(b)`. Applicata a mille fattori,
> trasforma un prodotto ingestibile in una somma comoda. È il motivo per cui i
> logaritmi compaiono ovunque quando si maneggiano probabilità.

**Ritocco 2 — il segno meno.** Per convenzione, negli algoritmi si *minimizzano* le
funzioni di costo, non si massimizzano. Basta cambiare segno: minimizzare `−log P`
equivale a massimizzare `log P`. E `−log(p)` ha un'interpretazione bellissima:

- se il modello era **certo** del carattere giusto (`p ≈ 1`), allora `−log(1) = 0`:
  nessuna penalità;
- se gli aveva dato probabilità **bassa** (`p ≈ 0.01`), allora `−log(0.01) ≈ 4.6`:
  penalità grande;
- se gli aveva dato `p ≈ 0`, la penalità tende a **infinito**.

In altre parole, `−log(p)` **punisce la sicurezza malriposta**: sbagliare essendo
sicuri costa carissimo. È esattamente il comportamento che vogliamo da un modello
onesto.

**Ritocco 3 — la media.** Invece della somma totale, prendiamo la **media** per
carattere. Così il numero non dipende dalla lunghezza del testo (un libro lungo non
ha "più loss" di uno corto solo perché è lungo): diventa confrontabile tra dataset
diversi e — cosa cruciale per noi — **tra le fasi del progetto**.

Il risultato di questi tre ritocchi ha un nome: **cross-entropy**, o **negative
log-likelihood (NLL)**. Nel codice:

```python
probs = self.P[a, b]                # probabilità assegnata a ogni coppia realizzata
return float(-np.log(probs).mean())  # media di -log: la cross-entropy
```

> **📖 Concetto: cosa "sente" la loss — sorpresa e perplexity.** La NLL media si può
> leggere come la **"sorpresa media per carattere"**, misurata in *nats* (l'unità
> quando si usa il logaritmo naturale). Bassa sorpresa = il modello quasi sempre
> "se lo aspettava". Un'altra lettura, ancora più intuitiva, è la **perplexity** =
> `e^NLL`: approssima *"tra quanti caratteri, in media, il modello sta davvero
> esitando"*. Perplexity 10 significa: è come se, a ogni passo, il modello fosse
> indeciso tra ~10 caratteri equiprobabili. Per un vocabolario di 69, esitare tra 10
> invece che tra 69 è un bel progresso.

**Questa è la stessa identica loss con cui è addestrato GPT-4.** Da qui in avanti,
ogni modello del progetto sarà giudicato da questo numero. La cosa straordinaria è
che l'abbiamo capito su un modello che *conta*: nella prossima fase useremo la
*stessa* loss come bussola per un modello che *impara*.

---

<a name="sec-1-6"></a>
## 1.6 I nostri numeri, letti uno per uno

Ecco cosa produce il nostro bigram sul corpus Pinocchio:

```
NLL uniforme (log 69)   = 4.2341 nats     <- il modello che tira a caso
NLL bigram TRAIN        = 2.3340 nats
NLL bigram VAL          = 2.3455 nats
perplexity VAL (e^NLL)  = 10.44
```

Leggiamoli come un ricercatore leggerebbe una tabella di risultati:

- **4.2341 è il riferimento da battere.** È `log(69)`: la sorpresa di un modello che
  non sa nulla e assegna 1/69 a ogni carattere. Qualsiasi modello utile deve stare
  *sotto* questo numero.
- **2.3455 sul validation** è nettamente sotto 4.2341: il bigram ha imparato
  qualcosa di reale sull'italiano. Quasi *dimezza* la sorpresa rispetto al caso.
- **Train (2.3340) ≈ Val (2.3455).** I due numeri sono quasi identici. Questo ci dice
  una cosa importante: **il bigram non fa overfitting**. È troppo "povero" (una
  tabella fissa) per memorizzare il training; quel poco che impara generalizza
  perfettamente. È la prima osservazione sperimentale del rapporto tra capacità del
  modello e overfitting — un tema che esploderemo in Fase 4, dove per la prima volta
  vedremo train e val *separarsi*.
- **Verifica qualitativa:** abbiamo controllato la riga della `q`, e il modello
  prevede `u` al **93.3%**. Ha imparato, contando, una regola ortografica
  dell'italiano. Nessuno gliel'ha detta: era nei dati.

E il testo generato? Qualcosa come:

```
lona s... de filì; — bi E griede
pe  sccocoll lesil e arevemin facavancomasuto cogespiaccatogin d; vialo! ...
```

È **pseudo-italiano sillabico**: non ci sono parole vere, ma ci sono le doppie
(`cc`, `ll`), le vocali finali, gli spazi alla frequenza giusta, perfino i trattini
dei dialoghi e i punti di sospensione. Per un modello che vede **un solo carattere**
alla volta, questo è il massimo teorico. E vederlo tara le nostre aspettative: non
possiamo pretendere parole coerenti da chi non ricorda nemmeno la lettera di due
posizioni fa.

---

<a name="sec-1-7"></a>
## 1.7 Il limite del bigram e perché è il punto di partenza giusto

Il bigram ha un limite strutturale, ed è *esattamente* il motivo per cui esistono le
fasi successive: **dimentica tutto tranne l'ultimo carattere**. Non può sapere che
dopo `"Pinocchi"` viene quasi certamente `o`, perché guarda solo la `i`. Non ha
memoria del contesto.

Come si supera? Le prossime fasi sono, letteralmente, la lotta contro questo limite:

- **Fase 2** — impareremo le *stesse* probabilità del bigram, ma con la discesa del
  gradiente invece che coi conteggi. Non migliora la qualità (stesso modello!), ma
  ci insegna *il meccanismo dell'apprendimento*, che è ciò che poi scaleremo.
- **Fase 4** — l'MLP guarderà gli ultimi N caratteri (non uno solo), e comprimerà la
  "tabella impossibile" ([sezione 1.1](#sec-1-1)) usando gli embedding.
- **Fasi 5–7** — l'attention permetterà al modello di guardare *tutto* il contesto e
  decidere da solo quali parti contano.

Il bigram non è un modello "sbagliato": è il gradino zero, quello che rende visibili
i concetti prima che i meccanismi li nascondano. Ora che sappiamo cos'è una
distribuzione sul prossimo carattere, cos'è il campionamento e — soprattutto — cos'è
la loss, siamo pronti a far *imparare* una macchina.

---

<a name="sec-1-8"></a>
## 1.8 Glossario Fase 1 / cosa arriva in Fase 2

Nuovi termini:

- **Bigram**: coppia di caratteri consecutivi; il modello che predice il prossimo
  carattere dal solo precedente.
- **Smoothing (di Laplace)**: aggiungere un conteggio fittizio (`+1`) per evitare
  probabilità nulle.
- **Broadcasting**: la regola con cui NumPy allinea tensori di shape diverse
  replicando virtualmente le dimensioni di taglia 1.
- **Campionamento autoregressivo**: generare testo estraendo un carattere alla volta
  dalla distribuzione, rialimentando l'output.
- **Loss / Negative Log-Likelihood (NLL) / cross-entropy**: la media di `−log(p)`,
  la misura di "sorpresa" del modello. Più bassa = meglio.
- **Nats**: l'unità della NLL quando si usa il logaritmo naturale.
- **Perplexity**: `e^NLL`; "tra quanti caratteri il modello sta esitando".
- **Overfitting** (osservato *non* accadere qui): memorizzare invece di generalizzare.

**In Fase 2** faremo una cosa che sembra un giro a vuoto ma è illuminante: otterremo
lo *stesso* modello del bigram — le stesse probabilità — ma invece di *contarle* le
faremo **imparare** a una piccola rete, per discesa del gradiente. Deriveremo il
gradiente **a mano** (mezz'ora di algebra che ripaga per sempre) e scopriremo che il
cuore dell'apprendimento di *ogni* LLM è una sottrazione sorprendentemente semplice:
`probabilità_predette − verità`. È il vero "hello world" della backpropagation.

---

*Fine del capitolo Fase 1.*

---

<a name="fase-2"></a>
# Fase 2 — Imparare invece di contare: la backpropagation

Questa è, per la comprensione, **la fase più importante di tutto il progetto**. Tutto
ciò che verrà dopo — l'MLP, l'attention, il GPT — è la ripetizione, su funzioni più
ricche, del meccanismo che costruiamo qui. Se capisci fino in fondo questo capitolo,
hai capito come si addestra *qualsiasi* rete neurale, GPT-4 incluso.

📁 File: [`ronklm/models/bigram_neural.py`](ronklm/models/bigram_neural.py)

---

<a name="sec-2-0"></a>
## 2.0 L'idea: la stessa meta, per la strada opposta

In Fase 1 abbiamo ottenuto le probabilità del bigram **contando**. In Fase 2
otterremo *le stesse probabilità* in un modo completamente diverso: le faremo
**imparare** a una piccola rete, per tentativi corretti dal gradiente.

Sembra un giro a vuoto — perché imparare a fatica ciò che si può contare in un
istante? Perché **il conteggio non scala** (lo abbiamo visto: la tabella esplode con
il contesto), mentre l'apprendimento per gradiente **sì**. Il bigram è il banco di
prova perfetto: siccome sappiamo *già* dove si deve arrivare (la `P` contata), ogni
pezzo del meccanismo di apprendimento è verificabile contro una verità nota. È come
imparare a usare una bussola in un posto di cui hai già la mappa: se la bussola ti
porta dove dice la mappa, ti puoi fidare quando andrai in territori inesplorati (le
fasi successive).

---

<a name="sec-2-1"></a>
## 2.1 One-hot e la matrice dei pesi W

Il modello è **una sola matrice** `W` di dimensione `(V × V)` = 69×69. Come si usa?

Un carattere in ingresso, poniamo l'indice `i`, viene prima rappresentato come
vettore **one-hot**: un vettore lungo 69, tutto zeri tranne un `1` nella posizione
`i`. Poi i *logits* (i punteggi grezzi per il prossimo carattere) si ottengono così:

```
logits = onehot(i) @ W
```

> **📖 Concetto: perché one-hot e non il numero grezzo.** Potremmo dare alla rete
> direttamente l'indice (12 per `m`, 13 per `n`). Ma così suggeriremmo che `n` = `m`
> + 1 in qualche senso numerico — una relazione **falsa e dannosa**: le lettere non
> hanno un ordine aritmetico. Il one-hot rende tutti i caratteri *equidistanti*:
> ognuno è una direzione diversa e indipendente. È il modo onesto di dare un simbolo
> discreto a una macchina che fa solo aritmetica.

> **🔧 Nel codice: one-hot @ matrice = selezione di riga.** C'è un'osservazione che
> pagherà tantissimo in Fase 4. Moltiplicare un vettore one-hot (con l'1 in
> posizione `i`) per una matrice `W` **non fa alcun vero calcolo**: restituisce
> semplicemente la **riga `i`-esima** di `W`. Prova a immaginarlo: l'1 "pesca" la
> riga `i`, gli zeri annullano tutte le altre. Quindi nel codice non scriviamo
> davvero la moltiplicazione, scriviamo `self.W[x_idx]` — selezioniamo le righe. È
> più veloce e identico. Quando in Fase 4 arriveranno gli **embedding**, non saranno
> un'idea nuova: saranno *questa stessa selezione di riga*, con righe più corte.

**Inizializzazione.** `W` parte con numeri gaussiani **piccoli** (deviazione standard
0.01). Perché piccoli e non zero, e perché ci teniamo?

- Con `W` tutti uguali (es. zero), tutti i logit sono uguali → il modello parte dalla
  distribuzione uniforme. Qui andrebbe bene, ma nelle reti a più strati pesi tutti
  identici creano neuroni che ricevono gradienti identici e non si differenziano mai
  (un problema chiamato "rottura della simmetria" mancata). Prendiamo subito
  l'abitudine giusta: init casuale.
- **Piccoli**, perché logit grandi a caso = modello che parte *sicurissimo e a
  sproposito* = loss iniziale enorme e primi passi violenti.

> **📖 Concetto: il sanity check della loss iniziale.** Ecco un controllo che
> useremo *sempre*, in ogni fase, e che cattura una quantità sorprendente di bug in
> 5 secondi. All'inizio del training, un modello ben inizializzato non sa nulla,
> quindi deve assegnare ~la stessa probabilità a ogni carattere: la sua loss deve
> valere circa **log(V) = log(69) ≈ 4.23**. Se all'avvio la loss è molto diversa (o
> è `NaN`), qualcosa nell'inizializzazione o nel forward è rotto. Il nostro modello,
> misurato, parte a **4.2339** — perfetto.

---

<a name="sec-2-2"></a>
## 2.2 La softmax, spiegata a fondo

I logit sono numeri reali qualsiasi (anche negativi, anche 100, anche −7). Ma a noi
servono *probabilità* (positive, che sommano a 1) per poter usare la loss della Fase
1. Il ponte tra i due mondi è la funzione **softmax**:

```
softmax(z)_j = exp(z_j) / Σ_k exp(z_k)
```

In parole: prendi l'esponenziale di ogni logit (così diventano tutti positivi),
poi dividi ciascuno per la somma di tutti (così sommano a 1).

> **📖 Concetto: perché proprio la softmax.** Ha esattamente le proprietà che
> servono: (a) `exp` rende tutto positivo; (b) la divisione normalizza a 1; (c) è
> **derivabile ovunque** — cruciale, perché il gradiente dovrà attraversarla; (d)
> preserva l'ordine (il logit più grande resta la probabilità più grande); (e) il
> rapporto tra due probabilità dipende *esponenzialmente* dalla differenza dei loro
> logit — cioè pochi punti di logit di vantaggio si traducono in un dominio quasi
> totale. Il nome viene da qui: è una versione "morbida" (soft) e derivabile della
> funzione `argmax` (che sceglierebbe seccamente il massimo).

> **⚠️ Trappola numerica: il `- max`.** `exp(z)` esplode: `exp(800)` supera il più
> grande numero rappresentabile e diventa `inf`, che poi contamina tutto in `NaN`.
> La cura: prima di fare gli esponenziali, sottraiamo a tutti i logit il loro massimo
> (`z - z.max()`). Questo **non cambia il risultato** (la softmax è invariante per
> traslazione: aggiungere una costante a tutti i logit lascia le probabilità
> identiche — contano solo le *differenze* tra logit), ma porta il logit più grande a
> 0 e rende ogni `exp` calcolabile. Nel codice è la riga `z = logits - logits.max(...)`.
> Prima lezione di una verità permanente: **la matematica su carta e la matematica in
> virgola mobile sono due discipline diverse**, e i `NaN` nascono quasi sempre in
> punti come questo.

---

<a name="sec-2-3"></a>
## 2.3 La derivazione del gradiente, a mano, passo per passo

Eccoci al cuore. Vogliamo sapere: *di quanto, e in che direzione, cambiare ogni peso
di `W` per ridurre la loss?* Questa informazione è il **gradiente**. Lo deriviamo
una volta nella vita, a mano, perché il risultato è così semplice e illuminante da
togliere per sempre la magia dal training.

> **📖 Concetto: cos'è un gradiente.** Per ogni peso, il gradiente è un numero che
> dice due cose insieme: il *segno* (aumentando quel peso, la loss sale o scende?) e
> la *grandezza* (quanto è sensibile la loss a quel peso?). Muovendo ogni peso nella
> direzione **opposta** al suo gradiente, la loss scende. Questo è tutto ciò che fa
> l'addestramento.

Prepariamo i pezzi. Per un singolo esempio: input carattere `i`, risposta giusta
carattere `y`.
- logit: `z = W[i]` (un vettore di 69 numeri)
- probabilità: `p = softmax(z)` (69 numeri che sommano a 1)
- loss: `L = −log(p[y])` (quanto ci siamo sorpresi del carattere giusto)

**Passo 1 — la derivata della loss rispetto ai logit.** Questo è il calcolo centrale.
Vogliamo `∂L/∂z_j` per ogni `j`. Serve la regola della catena attraverso la softmax e
il logaritmo. Salto i dettagli algebrici (sono un'oretta di conti standard: la
derivata di `−log(softmax)`), ma il risultato è di una bellezza sorprendente:

```
∂L/∂z_j = p_j − 1{j = y}
```

dove `1{j = y}` vale 1 se `j` è il carattere giusto, 0 altrimenti. In forma
vettoriale:

```
∂L/∂z = p − onehot(y)          cioè:   probabilità_predette − verità
```

Fermiamoci a **capire cosa dice questa formula**, perché è tutta la magia:

- Se il modello aveva dato al carattere giusto probabilità **alta** (diciamo
  `p[y] = 0.9`), allora nella posizione giusta il gradiente è `0.9 − 1 = −0.1`:
  **piccolo**, poco da correggere.
- Se gli aveva dato probabilità **bassa** (`p[y] = 0.01`), il gradiente lì è
  `0.01 − 1 = −0.99`: **grande**, forte correzione.
- Nelle posizioni *sbagliate* (i caratteri che non dovevano venire), il gradiente è
  `p_j − 0 = p_j`: positivo, "abbassa questa probabilità", tanto più quanto più
  erroneamente alta era.

**La correzione è automaticamente proporzionale all'errore.** Il modello si corregge
tanto quanto ha sbagliato, senza che nessuno glielo dica esplicitamente. Questa
sottrazione, `probs − verità`, è ciò che sta dentro `loss.backward()` per l'ultimo
strato di *ogni* modello di linguaggio esistente. Non è un'esagerazione: è
letteralmente questa riga.

**Passo 2 — dalla derivata sui logit a quella su W.** I logit erano `z = W[i]`, cioè
la riga `i` di `W`. Quindi la derivata rispetto a `W` tocca **solo la riga `i`**:

```
∂L/∂W[i] = ∂L/∂z = p − onehot(y)          e   ∂L/∂W[altre righe] = 0
```

> **📖 Perché ha perfettamente senso.** Vedere l'esempio "`q` → `u`" non insegna
> nulla su cosa viene dopo la `z`. È giusto che l'esempio aggiorni *solo* la riga
> della `q` (il carattere che abbiamo effettivamente visto) e lasci intatte le altre
> 68 righe. Il one-hot in ingresso "instrada" il gradiente esattamente sulla riga
> giusta.

**Passo 3 — su un intero batch.** Con B esempi, sommiamo i contributi e dividiamo per
B (perché la loss è la *media*). Nel codice:

```python
dlogits = probs.copy()
dlogits[np.arange(B), y_idx] -= 1.0     # probs - onehot(y)
dlogits /= B                            # media sul batch
dW = np.zeros_like(self.W)
np.add.at(dW, x_idx, dlogits)           # accumula ogni riga nel posto giusto
```

Guarda com'è fedele alla derivazione: `dlogits` è `probs`, meno 1 nelle posizioni
giuste (`probs − onehot(y)`), diviso B; poi `np.add.at` accumula ogni `dlogits`
nella riga del carattere corrispondente. **Tre righe di codice, un'oretta di algebra
capita per sempre.**

---

<a name="sec-2-4"></a>
## 2.4 Il training loop: i 5 passi del deep learning

Ora che sappiamo calcolare il gradiente, il resto è un ciclo. *Questo* ciclo:

```
per ogni step:
    1. X, Y = prendi dei dati          # i caratteri e le loro risposte giuste
    2. probs = forward(X)              # il modello prevede
    3. loss  = cross_entropy(probs, Y) # quanto ha sbagliato?
    4. dW    = backward(...)           # in che direzione correggere ogni peso?
    5. W    -= lr * dW                 # fai un piccolo passo in quella direzione
```

> **Questo ciclo È il deep learning.** Dalla più semplice regressione a GPT-4,
> l'intero campo è questo ciclo di cinque righe; cambia soltanto *cosa c'è dentro il
> forward al passo 2*. Le fasi 4, 5, 6, 7 non toccheranno mai più questi cinque
> passi: arricchiranno solo il modello dentro il passo 2. Tienilo a mente: quando il
> GPT sembrerà complicato, il *modo* in cui impara sarà ancora esattamente questo.

**Perché su un batch e non su tutto insieme (o un esempio alla volta)?** Calcolare il
gradiente su *tutto* il corpus a ogni passo è il più accurato, ma costa una passata
intera per un solo aggiornamento. Calcolarlo su un piccolo **batch** casuale è una
*stima rumorosa* del gradiente vero — e va benissimo: mille passi rumorosi ed
economici battono un passo perfetto e costosissimo. Questo si chiama **discesa del
gradiente stocastica** (SGD): la "S" sta per il batch casuale. (Curiosamente, il
rumore ha perfino effetti *benefici* sulla generalizzazione, ma è un'altra storia.)

---

<a name="sec-2-5"></a>
## 2.5 Il learning rate: l'arte del passo giusto

Al passo 5 c'è un numero, `lr` (learning rate, tasso di apprendimento), che moltiplica
il gradiente. È **l'iperparametro più importante del machine learning**, e vale la
pena capirlo con un'immagine.

> **📖 Concetto: scendere in una valle nella nebbia.** Immagina la loss come un
> paesaggio di montagne e valli, e i pesi come la tua posizione. Vuoi arrivare al
> fondo di una valle (loss minima). Il gradiente ti dice la *pendenza sotto i piedi*
> — la direzione di massima salita — quindi vai nella direzione opposta. Ma di quanto
> ti sposti a ogni passo? Questo lo decide il `lr`:
>
> - **`lr` troppo grande:** fai passi enormi, scavalchi la valle e ti ritrovi più in
>   alto dall'altra parte. La loss *oscilla* o addirittura *diverge* (esplode).
> - **`lr` troppo piccolo:** fai passetti minuscoli, impieghi ere geologiche a
>   scendere. Il training è cortissimo di vista.
> - **`lr` giusto:** scendi spedito ma controllato.

Il gradiente è un'informazione *locale* (la pendenza *qui*): dice la direzione giusta
solo per un passettino. Ecco perché il `lr` esiste ed è delicato: sbagliarlo di un
fattore 10 può distruggere qualsiasi addestramento. Nelle fasi successive vedremo due
raffinamenti: l'ottimizzatore **AdamW** (Fase 4), che adatta di fatto un `lr` diverso
per ogni peso, e lo **scheduling** (Fase 8), che lo fa variare durante il training.

---

<a name="sec-2-6"></a>
## 2.6 La prova del nove: neurale ≡ conteggi

Come facciamo a sapere che tutto questo meccanismo — forward, softmax, gradiente,
update — funziona davvero? Con una verifica netta, resa possibile dal fatto che
conosciamo già la risposta.

> **📖 Concetto: c'è una sola soluzione ottima, ed è quella contata.** Un risultato
> matematico standard (la *massima verosimiglianza*) dice che la matrice `W` che
> minimizza la cross-entropy corrisponde *esattamente* alla distribuzione empirica dei
> conteggi. In altre parole: il minimo assoluto del paesaggio di loss è proprio la
> `P` che in Fase 1 avevamo ottenuto contando. Quindi il training, partito da pesi
> **casuali**, deve *riscoprire da solo*, per pura discesa del gradiente, la tabella
> della Fase 1.

E infatti accade. Ecco i numeri misurati:

```
loss iniziale (pesi casuali)  = 4.2339    (~log 69: parte ~uniforme ✓)
loss dopo 500 step            = 2.3727
NLL neurale   train / val     = 2.3726 / 2.3853
NLL conteggio train / val     = 2.3340 / 2.3455
scarto medio |P_neurale − P_conteggi| = 0.012
```

Il modello neurale, partito dal caos, è sceso fino a **sfiorare** la NLL del bigram a
conteggio, e le due matrici di probabilità differiscono in media di appena 0.012. (Il
piccolo scarto residuo è perché a 500 passi non è *ancora* perfettamente al minimo, e
perché il conteggio usa lo smoothing e il neurale no: differenze di secondo ordine.)
La bussola ci ha portato dove diceva la mappa. **Il meccanismo di apprendimento
funziona** — e ora possiamo fidarcene quando andremo dove la mappa non c'è.

---

<a name="sec-2-7"></a>
## 2.7 Il gradient check: come si verifica un gradiente

C'è un secondo test, e nel deep learning è **il più importante che si possa
scrivere**. Serve a rispondere alla domanda: "sono sicuro che il gradiente che ho
derivato a mano è giusto?".

> **📖 Concetto: perché un gradiente sbagliato è il bug peggiore.** Un backward
> sbagliato spesso **non rompe niente di visibile**: il training parte, la loss magari
> perfino scende (male), e passi giorni a incolpare il learning rate o
> l'architettura mentre la colpa era in una derivata sbagliata. Non dà errori: degrada
> in silenzio. Per questo va verificato con un metodo *indipendente*.

L'idea del **gradient check** è geniale nella sua semplicità: verifichiamo il
gradiente (difficile da scrivere giusto) usando *solo* la loss (facile da scrivere
giusta). Come? Con la definizione stessa di derivata. La derivata della loss rispetto
a un peso `w` è "di quanto cambia la loss se muovo un pochino `w`". Allora:

1. prendi un peso `w`, aumentalo di un pochino `h` (es. `h = 10⁻⁵`), misura la loss;
2. diminuiscilo di `h`, misura di nuovo la loss;
3. la stima numerica del gradiente è `(loss(w+h) − loss(w−h)) / (2h)`.

Se questa stima numerica **coincide** con il gradiente analitico (quello derivato a
mano), il gradiente è giusto. Nel nostro test, l'errore relativo massimo è risultato
**sotto 10⁻⁴**: coincidono.

> **📖 Concetto: perché la differenza *centrale* (`w+h` e `w−h`) e non solo `w+h`.**
> Un'analisi con la formula di Taylor mostra che la differenza "in avanti"
> `(L(w+h) − L(w))/h` ha un errore proporzionale a `h`, mentre quella "centrale"
> `(L(w+h) − L(w−h))/(2h)` ha un errore proporzionale a `h²` — enormemente più
> piccolo a parità di `h`. È il motivo per cui si usa sempre la forma centrale.

> **📖 Concetto: e allora perché non addestrare *così*, numericamente?** Perché il
> gradient check richiede **due valutazioni della loss per ogni singolo peso**. Per
> un modello da un milione di parametri sarebbero due milioni di forward per un solo
> passo di training. La backpropagation ottiene *tutti* i gradienti al costo di circa
> **due** forward in totale. È questa efficienza, e nient'altro, ad aver reso
> possibile il deep learning. Il gradient check è uno strumento di *verifica*, non di
> allenamento. In Fase 3 lo trasformeremo da artigianale a sistematico, su ogni
> singola operazione del nostro motore.

---

<a name="sec-2-8"></a>
## 2.8 Un bonus elegante: allenare dai conteggi

Nel codice c'è un metodo `train_from_counts` che merita una nota, perché racchiude una
piccola perla. Allenare full-batch sulle 216.000 coppie è lento (ogni passo tocca
tutte le coppie). Ma per un bigram c'è una scorciatoia *esatta*: siccome tutte le
posizioni con lo stesso carattere di partenza `i` hanno gli stessi logit `W[i]`, il
gradiente sommato sulla riga `i` si scrive in forma chiusa usando solo i conteggi:

```
dW[i] = ( (quante volte i appare come "da") · softmax(W[i]) − N[i] ) / totale
```

Cioè: non serve toccare le coppie una per una, bastano i conteggi `N` della Fase 1.
Il costo per passo crolla da "proporzionale al numero di coppie" a "proporzionale a
V²" (69² celle). **È la stessa cosa vista da due lati**: la conferma, ancora una
volta, che il modello che conta e il modello che impara sono profondamente lo stesso
oggetto. (Nel progetto usiamo questa via veloce per i test di convergenza; il ciclo
stocastico "onesto" resta disponibile e testato a parte.)

---

<a name="sec-2-9"></a>
## 2.9 Glossario Fase 2 / cosa arriva in Fase 3

Nuovi termini:

- **Logit**: il punteggio grezzo (reale qualsiasi) che il modello dà a un token
  *prima* della softmax.
- **One-hot**: rappresentazione di un simbolo come vettore di tutti 0 con un solo 1.
- **Softmax**: funzione che trasforma logit in probabilità (positive, somma 1).
- **Gradiente**: per ogni peso, direzione e intensità in cui muoverlo per far
  scendere la loss.
- **Backpropagation (backward)**: l'algoritmo che calcola i gradienti propagandoli
  dall'output all'indietro.
- **Learning rate (`lr`)**: quanto è grande il passo nella direzione del gradiente.
- **SGD (discesa del gradiente stocastica)**: aggiornare i pesi usando gradienti
  stimati su batch casuali.
- **Massima verosimiglianza**: il principio per cui il miglior modello è quello che
  rende più probabili i dati osservati (→ minimizza la cross-entropy).
- **Gradient check**: verificare il gradiente analitico confrontandolo con una stima
  numerica (differenze finite centrali).

**In Fase 3** affronteremo il problema che rende impossibile scalare oltre il bigram:
derivare i gradienti *a mano* va bene per un layer, ma un GPT ne ha decine, annidati.
Costruiremo **`ronkgrad`**, un piccolo motore di **differenziazione automatica** (~250
righe) che, come il cuore di PyTorch, calcola i gradienti *da solo* percorrendo
all'indietro un "grafo" delle operazioni. Dopo la Fase 3 non deriveremo mai più un
gradiente a mano: ci basterà scrivere il forward. E capiremo, dall'interno, cosa fa
davvero quel `loss.backward()` che nei framework sembra magia.

---

*Fine del capitolo Fase 2.*

---

<a name="fase-3"></a>
# Fase 3 — ronkgrad: costruire un mini-PyTorch

In Fase 2 abbiamo derivato un gradiente a mano: mezz'ora di algebra per **un** layer.
Il GPT ne avrà decine, annidati. Derivare a mano *non scala*. In questa fase
costruiamo **`ronkgrad`**, un motore di **differenziazione automatica** (~300 righe)
che calcola i gradienti *da solo*. È concettualmente identico al cuore di PyTorch:
dopo questa fase, quel `loss.backward()` che sembra magia non avrà più segreti, e non
dovremo mai più derivare un gradiente a mano — ci basterà scrivere il *forward*.

📁 File: [`ronklm/autograd.py`](ronklm/autograd.py)

---

<a name="sec-3-0"></a>
## 3.0 Il problema: derivare a mano non scala

Immagina di dover derivare a mano il gradiente di una rete come:

```
loss = cross_entropy( (x @ W1).tanh() @ W2 , y )
```

Dovresti applicare la regola della catena attraverso `cross_entropy`, poi `@ W2`, poi
`tanh`, poi `@ W1` — e questo è un *giocattolo* con due soli strati. Un transformer
ha embedding, decine di blocchi con attention e feed-forward, layernorm, connessioni
residue. Derivare tutto a mano sarebbe centinaia di pagine di algebra, con un errore
ogni tre righe. Serve **automatizzare la regola della catena**. È esattamente ciò che
fa un motore di *autograd* (automatic gradient).

---

<a name="sec-3-1"></a>
## 3.1 L'idea: il grafo computazionale

Ecco l'intuizione che sblocca tutto. Qualsiasi calcolo, per quanto complicato, è una
**catena di operazioni elementari**: somme, prodotti, prodotti-matrice, `tanh`,
esponenziali… La nostra `loss` qui sopra è: prendi `x`, moltiplicalo per `W1`
(matmul), applica `tanh`, moltiplica per `W2` (matmul), calcola la cross-entropy.
Cinque operazioni elementari in fila.

> **📖 Concetto: il grafo computazionale.** Mentre eseguiamo il forward (i calcoli in
> avanti), possiamo *registrare chi ha prodotto cosa*. Ne risulta un **grafo**: i
> nodi sono i tensori (i valori intermedi), e le frecce dicono "questo è stato
> prodotto da quello tramite questa operazione". Il grafo è *diretto* (le frecce
> hanno un verso) e *aciclico* (non si torna mai su sé stessi): in gergo un DAG.

Perché registrarlo? Per il teorema fondante del calcolo differenziale: la derivata di
una composizione di funzioni è il **prodotto delle derivate dei pezzi** (la regola
della catena). Quindi, se *ogni operazione elementare sa calcolare la propria piccola
derivata*, allora il gradiente della loss rispetto a **qualsiasi** tensore del grafo
si ottiene camminando il grafo **all'indietro** e moltiplicando/accumulando quei
pezzettini. Nessuno deve mai derivare la formula composta: la composizione avviene da
sola, un'operazione alla volta.

> **Questa è, letteralmente, la cosa che fa PyTorch.** Costruisce il grafo durante il
> forward, lo percorre a ritroso nel backward. La differenza tra ronkgrad e PyTorch è
> solo ingegneria (C++, GPU, fusione di operazioni), non concetto. Dopo questa fase,
> quando userai PyTorch, saprai *esattamente* cosa succede sotto.

---

<a name="sec-3-2"></a>
## 3.2 La classe Tensor: dati, gradiente, e "come tornare indietro"

Il mattone è la classe `Tensor`, che avvolge un array NumPy e in più ricorda quattro
cose:

```python
class Tensor:
    def __init__(self, data, _prev=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)  # i valori (il forward)
        self.grad = np.zeros_like(self.data)            # il gradiente accumulato (il backward)
        self._backward = lambda: None                   # come spingere il gradiente ai genitori
        self._prev = _prev                              # i tensori da cui sono nato
        self._op = _op                                  # etichetta (solo per debug)
```

- **`data`** sono i valori, quelli che calcoli in avanti.
- **`grad`** è lo spazio dove si accumulerà `∂loss/∂questo_tensore`. Parte da zero.
- **`_prev`** sono i "genitori": i tensori che hanno prodotto questo. È il grafo.
- **`_backward`** è una piccola funzione, specifica di ogni operazione, che sa
  prendere il gradiente *di questo* tensore e spingerlo ai genitori. Sui tensori
  "foglia" (i dati e i pesi, che non nascono da nulla) è un no-op.

Ogni volta che scrivi `c = a + b`, l'operazione crea il nuovo tensore `c`, ne imposta
`_prev = (a, b)` e gli attacca la funzione `_backward` giusta per la somma. Il grafo
si costruisce da solo, un'operazione alla volta, semplicemente *facendo i calcoli*.

---

<a name="sec-3-3"></a>
## 3.3 Le operazioni e i loro backward

Ogni operazione segue lo stesso schema: calcola il risultato in avanti, e definisce
come il gradiente torna indietro. Vediamone tre, che sono la spina dorsale.

**Somma** `c = a + b`. In avanti, somma elemento per elemento. All'indietro:

```python
def _backward():
    self.grad += _unbroadcast(out.grad, self.data.shape)
    other.grad += _unbroadcast(out.grad, other.data.shape)
```

Il gradiente passa **invariato** a entrambi gli addendi. La somma è un "distributore
di gradiente": qualunque gradiente arrivi a `c`, lo copia identico su `a` e su `b`.

> **📖 Ricorda questo fatto.** "La somma distribuisce il gradiente intatto" sembra
> banale ora, ma è *precisamente* la ragione per cui le **connessioni residue** (Fase
> 6) permettono di addestrare reti profonde: il ramo `x` in `x + f(x)` diventa
> un'autostrada su cui il gradiente scorre senza attenuarsi. Ci torneremo.

**Prodotto** `c = a * b`. All'indietro, ognuno riceve il gradiente moltiplicato per
*l'altro*: `a.grad += b * out.grad`, e viceversa. (Quanto conta un fattore dipende da
quanto vale l'altro: è la regola del prodotto del liceo.)

**Prodotto matriciale** `C = A @ B`. Le formule sono `dA = dC @ Bᵀ` e `dB = Aᵀ @ dC`
(dove ᵀ è la trasposta). Non le deriviamo qui (sono in gioco somme di indici), ma c'è
un **controllo mnemonico che vale oro**:

> **🔧 Nel codice: le shape devono tornare.** `dA` deve avere la stessa forma di `A`.
> C'è un solo modo di combinare `dC`, `Bᵀ`, `Aᵀ` perché i conti delle dimensioni
> quadrino — e quel modo è la formula giusta. Metà dei bug di backward si trovano
> semplicemente guardando se le shape combaciano. Ci torneremo ossessivamente da qui
> in poi.

---

<a name="sec-3-4"></a>
## 3.4 Il broadcasting all'indietro: il punto più insidioso

Questo è il pezzo tecnicamente più difficile dell'intero progetto, e il motivo per
cui il nostro motore è *tensoriale* e non scalare. Concentrati un attimo.

Ricordi il broadcasting ([sezione 1.3](#sec-1-3))? In avanti, NumPy ti lascia sommare
un vettore `(V,)` a una matrice `(B, V)`: replica *virtualmente* il vettore su tutte
le `B` righe. Comodissimo. Ma nel backward crea un problema.

> **📖 Concetto: se un valore è stato usato B volte, riceve B gradienti.** Quel
> vettore `(V,)`, nel forward, è stato sommato a *ognuna* delle B righe. Quindi
> influenza la loss attraverso B strade diverse. Per la regola della derivata totale,
> il suo gradiente è la **somma** dei B contributi. Ma il gradiente che arriva
> "dall'alto" ha forma `(B, V)` — una riga di gradiente per ognuna delle B righe. Per
> riportarlo alla forma `(V,)` del vettore originale, dobbiamo **sommare lungo la
> dimensione che era stata replicata**: `grad.sum(axis=0)`.

Se lo dimentichi, ottieni un gradiente di forma `(B, V)` dove ne serviva uno `(V,)`:
o crasha (bug fortunato, te ne accorgi) o — peggio — un broadcasting silenzioso lo
"aggiusta" in modo sbagliato e il training degrada senza un solo errore. Per gestirlo
in un posto solo, abbiamo scritto la funzione `_unbroadcast`, usata da *tutti* i
backward:

```python
def _unbroadcast(grad, shape):
    while grad.ndim > len(shape):        # elimina le dimensioni in piu' davanti
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):      # somma lungo gli assi che erano di taglia 1
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad
```

In pratica `_unbroadcast` "disfa" all'indietro esattamente ciò che il broadcasting
aveva fatto in avanti. È il pezzo che micrograd (il motore didattico *scalare* di
Karpathy) non ha bisogno di avere, perché non ha shape; noi sì, ed è la comprensione
in più che ci portiamo a casa.

---

<a name="sec-3-5"></a>
## 3.5 Perché i gradienti si accumulano

Avrai notato che in ogni `_backward` scriviamo `+=`, non `=`. Non è un dettaglio.

> **📖 Concetto: un tensore usato più volte somma i suoi gradienti.** Se lo stesso
> tensore `a` compare in più punti del grafo — per esempio in `a * a`, oppure un peso
> riusato — allora influenza la loss attraverso più strade, e il suo gradiente totale
> è la **somma** dei contributi di ogni strada (di nuovo la regola della derivata
> totale). Usare `=` invece di `+=` sovrascriverebbe il primo contributo con il
> secondo: uno dei bug classici di chi scrive un autograd.

Abbiamo un test apposta per questo (`test_reused_tensor_accumulates`): verifica che
il gradiente di `a * a + a` sia corretto, cosa che richiede l'accumulo.

> **🔧 Corollario che ogni utente PyTorch conosce.** Poiché i gradienti si accumulano,
> *prima* di ogni nuovo backward vanno azzerati (in PyTorch: `optimizer.zero_grad()`).
> Altrimenti si sommerebbero a quelli del passo precedente. Nel nostro motore, siccome
> creiamo tensori freschi a ogni forward, gli intermedi partono già da zero; per i
> pesi che riusiamo tra un passo e l'altro dovremo azzerare esplicitamente (lo faremo
> nel training loop di Fase 4). Ora sai *perché* quel `zero_grad` esiste.

---

<a name="sec-3-6"></a>
## 3.6 backward(): l'ordinamento topologico

Come si orchestrano tutti i `_backward` nell'ordine giusto? Con il metodo
`backward()`, che fa tre cose:

```python
def backward(self):
    topo = []; visited = set()
    def build(v):
        if id(v) not in visited:
            visited.add(id(v))
            for parent in v._prev:
                build(parent)
            topo.append(v)
    build(self)
    self.grad = np.ones_like(self.data)   # dL/dL = 1: il seme
    for v in reversed(topo):
        v._backward()
```

1. **Costruisce l'ordinamento topologico** del grafo con una visita in profondità
   (DFS). L'ordinamento topologico ha una proprietà cruciale: ogni nodo compare
   *dopo* tutti i suoi genitori.
2. **Semina** il gradiente della radice (la loss) a 1. Perché 1? Perché la derivata
   della loss rispetto a sé stessa è 1: è il punto da cui parte tutta la catena.
3. **Percorre l'ordinamento al contrario**, chiamando ogni `_backward`.

> **📖 Concetto: perché serve l'ordinamento topologico.** Il `_backward` di un nodo
> può eseguirsi correttamente solo quando il gradiente di quel nodo (`out.grad`) è
> *completo* — cioè quando tutti i nodi che lo usano hanno già versato il loro
> contributo. Percorrere il grafo in ordine topologico inverso garantisce esattamente
> questo: quando tocca a un nodo, tutti i suoi "figli" (a valle) hanno già propagato.
> Con un ordine sbagliato, un nodo propagherebbe un gradiente parziale, e il risultato
> sarebbe silenziosamente errato.

---

<a name="sec-3-7"></a>
## 3.7 Le non-linearità e perché sono obbligatorie

Il motore include `relu`, `tanh`, `exp`, `log`, `gelu`. Le prime servono come
"funzioni di attivazione": le non-linearità che stanno tra uno strato e l'altro. Non
sono un ornamento — sono **obbligatorie**, e vale la pena capire perché.

> **📖 Concetto: senza non-linearità, la profondità è finta.** Comporre solo
> operazioni lineari (matmul, somme) dà ancora una funzione lineare: dieci matmul in
> fila equivalgono matematicamente a *una sola* matmul. Quindi una rete "profonda"
> fatta di soli strati lineari avrebbe esattamente lo stesso potere espressivo di un
> singolo strato: tutta la profondità sarebbe sprecata. Infilare una funzione **non
> lineare** tra gli strati è ciò che permette alla rete di rappresentare funzioni
> arbitrariamente complicate (è il "teorema di approssimazione universale"). La
> non-linearità è il motivo per cui il deep learning è *deep*.

Le implementiamo con backward espliciti, tranne `gelu` che costruiamo per
**composizione** di operazioni già esistenti:

```python
def gelu(self):
    c = math.sqrt(2.0/math.pi)
    inner = (self + (self**3) * 0.044715) * c
    return (self * 0.5) * (inner.tanh() + 1.0)
```

> **🔧 Nel codice: gelu non ha un backward proprio.** È scritta usando solo `+`, `*`,
> `**`, `tanh` — operazioni che *già* sanno differenziarsi. Quindi il suo gradiente è
> **automatico**, gestito dal grafo: è la prima dimostrazione concreta della potenza
> dell'autograd. Scrivi il forward, e il backward viene gratis. (`gelu` è la
> variante "morbida" di `relu` usata dai GPT reali; la useremo nel feed-forward del
> transformer in Fase 6.)

---

<a name="sec-3-8"></a>
## 3.8 cross_entropy fusa: stabilità ed eleganza

La loss della Fase 1/2 è implementata come **una singola operazione** `cross_entropy`,
non come `log(softmax(...))` composto. Due ragioni.

- **Stabilità.** La composizione ingenua `log(softmax(z))` può produrre `log(0) = −∞`.
  La forma "fusa" (log-sum-exp con la sottrazione del massimo, come il trucco della
  [sezione 2.2](#sec-2-2)) è numericamente stabile per costruzione.
- **Semplicità del gradiente.** Come scoperto in Fase 2, il gradiente della coppia
  softmax+cross-entropy è la sottrazione semplice `(softmax − onehot) / B`, molto più
  pulita dei due gradienti separati moltiplicati. Il nostro `cross_entropy._backward`
  è letteralmente quelle poche righe.

> **🔧 La Fase 2 era, retroattivamente, la derivazione di questa operazione.** Anche
> PyTorch fonde softmax e cross-entropy (`F.cross_entropy`) per gli identici motivi.
> Averla derivata a mano prima significa che ora sappiamo *esattamente* cosa c'è
> dentro.

---

<a name="sec-3-9"></a>
## 3.9 La prova: 18 gradient check e la precisione macchina

Un motore di gradienti di cui non ci si fida è inutile — anzi, dannoso, perché
avvelenerebbe *tutte* le fasi 4–8 in silenzio. Per questo la fase non è chiusa finché
non passa una batteria di **18 gradient check**, uno per ogni operazione.

Ogni test usa la tecnica della [sezione 2.7](#sec-2-7): calcola il gradiente col
nostro backward (analitico) e lo confronta con la stima numerica (differenze finite,
che usano *solo* il forward). Due strade indipendenti: se coincidono entro una
tolleranza stretta (< 10⁻⁵), il backward è giusto. Copriamo tutto: somma con
broadcast, prodotto con broadcast, divisione, potenza, matmul 2D **e a batch** (che
servirà all'attention), riduzioni, tutte le non-linearità, softmax, cross-entropy, il
caso del tensore riusato (accumulo), e una mini-MLP composita.

E c'è una validazione ancora più bella. Abbiamo preso il bigram della Fase 2 e
calcolato il suo gradiente in **due** modi: quello *manuale* (derivato a mano in Fase
2) e quello *automatico* (lasciando fare a ronkgrad). Differenza massima:

```
max |dW_manuale − dW_autograd| = 2.08e-17
```

Cioè: **zero**, a meno dell'ultimo bit di precisione dei numeri in virgola mobile. Il
motore che abbiamo scritto riproduce *esattamente* la matematica che avevamo fatto a
mano. Ora possiamo fidarci ciecamente di `ronkgrad`.

> **🔧 Il dividendo di produttività.** Da qui in avanti, scrivere un modello nuovo
> richiederà **solo il forward**: il backward sarà gratis e già verificato. È lo
> stesso salto di produttività che i framework hanno regalato alla ricerca — e ce lo
> saremo guadagnato da soli, capendone ogni riga.

---

<a name="sec-3-10"></a>
## 3.10 Glossario Fase 3 / cosa arriva in Fase 4

Nuovi termini:

- **Autograd (differenziazione automatica)**: calcolare i gradienti automaticamente
  registrando le operazioni in un grafo e percorrendolo all'indietro.
- **Grafo computazionale (DAG)**: la rete di tensori e operazioni costruita durante il
  forward; diretto e aciclico.
- **Tensore foglia**: un tensore che non nasce da altri (dati, pesi); il suo
  `_backward` è un no-op.
- **`_unbroadcast`**: riportare un gradiente alla forma originale sommando lungo le
  dimensioni broadcastate.
- **Accumulo dei gradienti (`+=`)**: sommare i contributi di ogni uso di un tensore.
- **Ordinamento topologico**: un ordine dei nodi in cui ogni nodo viene dopo i suoi
  genitori; garantisce che il backward riceva gradienti completi.
- **Funzione di attivazione / non-linearità**: la funzione non lineare tra gli strati
  (relu, tanh, gelu…), senza cui la profondità sarebbe inutile.

**In Fase 4** useremo finalmente `ronkgrad` per costruire un modello vero: un **MLP**
(rete a più strati) che guarda **più caratteri** di contesto invece di uno solo.
Introdurremo due mattoni che i transformer usano ovunque — gli **embedding** (i
caratteri diventano vettori densi, la "compressione della tabella impossibile"
promessa in [Fase 1](#sec-1-1)) e gli **strati nascosti** — più l'ottimizzatore
**AdamW**, quello con cui si addestrano i GPT veri. E per la prima volta vedremo la
loss di validation *staccarsi* da quella di training: incontreremo l'overfitting dal
vivo.

---

*Fine del capitolo Fase 3.*

---

<a name="fase-4"></a>
# Fase 4 — L'MLP: contesto, embedding e AdamW

Finalmente usiamo `ronkgrad` per costruire un modello **vero**, e per la prima volta
rompiamo il limite del bigram: guardiamo **più caratteri** di contesto, non uno solo.
Introduciamo i due mattoni che i transformer useranno ovunque — gli **embedding** e
gli **strati nascosti** — più l'ottimizzatore **AdamW**, lo stesso dei GPT reali. E
incontriamo, dal vivo, la malattia numero uno del machine learning: l'**overfitting**.

📁 File: [`ronklm/nn.py`](ronklm/nn.py), [`ronklm/optim.py`](ronklm/optim.py),
[`ronklm/models/mlp.py`](ronklm/models/mlp.py)

---

<a name="sec-4-0"></a>
## 4.0 L'idea: rompere il limite di un solo carattere

Il bigram guardava un carattere. L'MLP (Multi-Layer Perceptron, "percettrone
multistrato") guarda gli ultimi `block_size` caratteri — noi useremo 8. L'architettura,
ispirata al primo language model neurale della storia (Bengio et al., 2003):

```
contesto: 8 indici di carattere  (B, 8)
   │
   ▼  Embedding: ogni carattere diventa un vettore denso
   (B, 8, n_embd)
   │
   ▼  concatena gli 8 vettori in uno solo
   (B, 8·n_embd)
   │
   ▼  Linear + tanh   ← lo "strato nascosto"
   (B, n_hidden)
   │
   ▼  Linear          ← la "testa"
   (B, vocab)   = logits del prossimo carattere
```

Il forward è tutto qui (dal codice), ed è leggibile riga per riga:

```python
e = self.emb(x_idx)                  # (B, 8, n_embd)
flat = e.reshape(B, 8 * n_embd)      # concatena
hidden = self.h(flat).tanh()         # (B, n_hidden)
return self.head(hidden)             # (B, vocab)
```

> **🔧 Nel codice: e il backward?** Non c'è. Non lo scriviamo. Grazie a `ronkgrad`
> (Fase 3), ci basta comporre operazioni che sanno già differenziarsi (`@`, `+`,
> `tanh`, `reshape`, `gather_rows`, `cross_entropy`), e il gradiente di *tutta* la
> rete arriva gratis chiamando `loss.backward()`. È il dividendo promesso: da qui in
> poi scriviamo solo il forward.

---

<a name="sec-4-1"></a>
## 4.1 L'infrastruttura: Module, Linear, Embedding

Prima del modello, tre mattoni riusabili in [`nn.py`](ronklm/nn.py) — l'equivalente di
`torch.nn`, ma trasparente.

**`Module`** è la classe base. La sua unica magia è `parameters()`: raccoglie
*ricorsivamente* tutti i tensori-parametro del modulo e dei suoi sotto-moduli.

> **📖 Concetto: perché serve.** Il GPT finale avrà decine di sotto-componenti
> annidati, ognuno coi suoi pesi. L'ottimizzatore ha bisogno della **lista completa**
> dei parametri da aggiornare. Raccoglierla a mano è il modo garantito di
> dimenticarne uno — che quindi non verrebbe mai addestrato, restando congelato ai
> valori casuali iniziali: un bug silenzioso classico. `Module.parameters()` che
> ispeziona automaticamente gli attributi risolve il problema una volta per tutte. È
> esattamente il ruolo di `nn.Module` in PyTorch. (Abbiamo un test,
> `test_all_params_get_gradient`, che verifica che *ogni* parametro riceva gradiente:
> il modo meccanico di scoprire un peso scollegato.)

**`Linear`** è lo strato lineare `y = x @ W + b`: il mattone più comune di ogni rete.

**`Embedding`** è la tabella che trasforma indici in vettori. È il cuore concettuale
della fase, e merita la sua sezione.

---

<a name="sec-4-2"></a>
## 4.2 Gli embedding: comprimere la tabella impossibile

Ricordi la promessa lasciata in sospeso in [Fase 1.1](#sec-1-1)? Il conteggio non
scala: una tabella per 10 caratteri di contesto avrebbe più celle che stelle
nell'universo. Gli embedding sono *la risposta* a quel problema.

> **📖 Concetto: cos'è un embedding.** Invece di rappresentare un carattere come un
> one-hot lungo 69 (tutto zeri e un uno), gli diamo un **vettore denso di pochi numeri
> reali** (noi ne usiamo 24) — e quei numeri sono **parametri addestrabili**, che il
> training aggiusta. La tabella `Embedding` è una matrice `(69, 24)`: la riga `i` è il
> vettore del carattere `i`.

Due conseguenze profonde:

1. **Compressione.** 69 caratteri descritti da 24 numeri ciascuno, invece che da
   vettori lunghi 69. E il contesto di 8 caratteri diventa `8 × 24 = 192` numeri, non
   `8 × 69`. La "tabella impossibile" del conteggio è sostituita da una *funzione* con
   pochi parametri che la approssima.

2. **Geometria della somiglianza.** Siccome i vettori sono appresi, il training è
   *libero di avvicinare* tra loro i caratteri che si comportano in modo simile — per
   esempio le vocali, o le cifre. Così ciò che il modello impara su `a` si trasferisce
   in parte a `e`. Il conteggio non poteva farlo: ogni riga della sua tabella era un
   universo isolato. (Nei LLM veri, a livello di parola, è la stessa idea che produce
   il famoso "re − uomo + donna ≈ regina": la geometria appresa cattura relazioni
   semantiche.)

> **🔧 Nel codice: onehot @ W diventa gather_rows.** In [Fase 2.1](#sec-2-1) avevamo
> osservato che "moltiplicare un one-hot per una matrice = selezionare una riga".
> L'`Embedding` rende questa osservazione ufficiale ed efficiente: invece di costruire
> one-hot e moltiplicare, selezioniamo direttamente le righe con `gather_rows` (che
> abbiamo aggiunto a ronkgrad, col suo backward "scatter-add": ogni riga usata riceve
> la somma dei gradienti dei punti in cui è stata usata). Stesso risultato, molto più
> veloce. Gli embedding *non sono* un'idea nuova: sono la selezione di riga di Fase 2,
> resa protagonista.

---

<a name="sec-4-3"></a>
## 4.3 Lo strato nascosto: rilevare combinazioni

Dopo gli embedding concatenati, c'è un `Linear` seguito da `tanh`: lo **strato
nascosto**. Perché serve?

> **📖 Concetto: senza lo strato nascosto, sarebbe ancora quasi una tabella.** Se
> proiettassimo gli embedding concatenati *direttamente* sui logit (una sola
> trasformazione lineare), il modello potrebbe solo *sommare contributi indipendenti*
> di ciascuna posizione. Lo strato nascosto con la non-linearità `tanh` gli permette
> invece di rilevare **combinazioni**: "c'è una `q` in penultima posizione **E** una
> `u` in ultima", oppure "le ultime tre lettere formano `-are`". Sono *feature
> composite* che nessuna trasformazione lineare può esprimere (è di nuovo il discorso
> della [sezione 3.7](#sec-3-7): senza non-linearità la profondità è finta). È qui che
> la rete smette di essere una tabella compressa e comincia davvero a *calcolare*.

---

<a name="sec-4-4"></a>
## 4.4 L'inizializzazione dei pesi: 1/√n

Un dettaglio del `Linear` che sembra pignoleria ma non lo è: i pesi si inizializzano
con deviazione standard `1/√n_in` (dove `n_in` è il numero di ingressi).

> **📖 Concetto: perché scalare l'init con 1/√n.** L'output di un neurone è la somma
> di `n_in` prodotti. Se ogni peso avesse una varianza fissa, la varianza della somma
> crescerebbe *proporzionalmente* a `n_in` (è una proprietà delle somme di variabili
> indipendenti). Con `n_in = 192` ingressi, le attivazioni sarebbero ~14 volte più
> "larghe" del dovuto. Perché è un problema? Perché la `tanh` **satura**: per input
> grandi si appiattisce a ±1, dove la sua derivata è ~0 → **il gradiente muore** al
> primo passaggio, e la rete non impara. Scalando i pesi con `1/√n_in`, la varianza
> dell'output resta ~1 indipendentemente dalla larghezza, e la `tanh` lavora nella sua
> zona "viva". È una delle scoperte (Glorot 2010, He 2015) che hanno reso addestrabili
> le reti profonde.

---

<a name="sec-4-5"></a>
## 4.5 AdamW, l'ottimizzatore dei GPT veri

In Fase 2 aggiornavamo i pesi con la regola più semplice: `W -= lr * grad` (SGD puro).
Funziona, ma è primitivo. In [`optim.py`](ronklm/optim.py) implementiamo **AdamW**,
lo stesso ottimizzatore con cui si addestrano GPT-3, GPT-4 e praticamente ogni LLM
moderno. Vale la pena capirlo pezzo per pezzo, perché ogni pezzo risolve un problema
concreto di SGD.

> **📖 Momentum (il primo momento).** I gradienti da minibatch sono *rumorosi*
> ([sezione 2.4](#sec-2-4)): ogni batch dà una stima leggermente diversa. AdamW tiene
> una **media mobile** dei gradienti nel tempo: questo filtra il rumore e accumula
> "velocità" nelle direzioni costanti. L'immagine: invece di un escursionista che a
> ogni passo riparte da fermo, una palla che rotola giù dalla valle, che mantiene lo
> slancio.

> **📖 Scaling adattivo (il secondo momento).** Parametri diversi ricevono gradienti
> di grandezza diversissima: l'embedding di una lettera rara riceve segnale
> raramente; i pesi della testa, sempre. Un unico `lr` per tutti è per forza sbagliato
> per qualcuno. AdamW tiene anche una media mobile dei *gradienti al quadrato*, e
> divide il passo di ogni peso per la sua radice: così **ogni parametro ottiene di
> fatto il suo learning rate su misura**, grande dove i gradienti sono piccoli e
> viceversa.

> **📖 Bias-correction.** Le due medie mobili partono da zero, quindi nei primissimi
> passi sono *sottostimate*. Senza correzione, i primi passi sarebbero distorti. La
> divisione per `(1 − β^t)` compensa esattamente questo transitorio iniziale.

> **📖 Weight decay disaccoppiato (la "W" di AdamW).** A ogni passo, i pesi vengono
> anche spinti dolcemente verso zero. È **regolarizzazione**: pesi piccoli =
> funzioni più semplici = meno overfitting (è la stessa filosofia dello smoothing di
> [Fase 1.2](#sec-1-2), in altra veste). Il dettaglio sottile: in Adam "classico"
> questo decay finiva *dentro* il gradiente e veniva ri-scalato dal meccanismo
> adattivo, indebolendolo; AdamW lo applica *fuori*, direttamente ai pesi — ed è per
> questo che l'industria usa AdamW e non Adam.

Averlo scritto a mano significa che quando leggerai, in un repo vero,
`torch.optim.AdamW(params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)`, ogni
argomento sarà un numero di cui conosci il meccanismo dall'interno.

---

<a name="sec-4-6"></a>
## 4.6 L'overfitting, dal vivo

In [Fase 1.6](#sec-1-6) avevamo notato che il bigram *non* faceva overfitting: train e
val avevano la stessa NLL, perché una tabella fissa è troppo "povera" per memorizzare.
L'MLP ha decine di migliaia di parametri, e per la prima volta vediamo il fenomeno.

> **📖 Concetto: memorizzare invece di generalizzare.** Con abbastanza capacità, la
> rete può iniziare a *memorizzare* pezzi specifici di Pinocchio invece di imparare
> regole generali dell'italiano. Sul training sembra sempre più brava; ma su testo
> nuovo (la validation) il miglioramento rallenta o si ferma. Il divario tra la NLL di
> train e quella di val è la **firma dell'overfitting**.

Nei nostri numeri lo si vede nascere: **train 1.81, val 1.90**. Il modello è
leggermente più bravo sul testo che ha visto che su quello nuovo. È ancora un divario
piccolo (il modello è modesto), ma è *reale* e crescerebbe allenando più a lungo o con
un modello più grande. I rimedi classici — più dati, modello più piccolo,
regolarizzazione (il weight decay di AdamW) — sono esattamente ciò che governeremo in
Fase 8. La regola operativa da interiorizzare: **la sola loss che conta è quella di
validation**; quella di train si può sempre abbassare "barando" (memorizzando).

---

<a name="sec-4-7"></a>
## 4.7 I numeri e il testo generato

```
                     bigram    MLP
NLL validation       2.346     1.897      ← guardare 8 caratteri invece di 1 paga
NLL train            2.334     1.81       ← (il gap train/val = overfitting)
```

L'MLP abbassa nettamente la sorpresa: da ~2.35 a ~1.90 nats. E nel testo generato il
salto si *vede*:

```
luspau trome, introva ibbecio ma
cold'omestro:
— Mi fiariventi.
Maverattestro con vifendavento
i burattino e titasse questo di fuoresono un grate fuino.
```

Rispetto al bigram sillabico, ora compaiono **parole vere** (`burattino`, `questo`,
`un`), abbozzi di nomi collodiani (`Maverattestro` ≈ Maestro, `ceppetterse` ≈
Geppetto), e la struttura dei dialoghi (`— Mi...`). Non è italiano coerente — il
contesto è ancora solo 8 caratteri — ma è un balzo evidente rispetto a Fase 1.

---

<a name="sec-4-8"></a>
## 4.8 Il limite dell'MLP e cosa arriva in Fase 5

L'MLP ha un limite **strutturale**, ed è la molla che fa scattare i transformer:

> **📖 Il contesto dell'MLP è rigido.** Guarda esattamente 8 caratteri, sempre, in
> posizioni fisse concatenate. Tre conseguenze: (1) la rete deve imparare *da capo*,
> per ogni posizione, come usare l'informazione lì contenuta — ciò che impara sulla
> "terzultima posizione" non si trasferisce alla "quartultima"; (2) allargare il
> contesto fa crescere *linearmente* i pesi del primo strato; (3) tutti i caratteri,
> vicini e lontani, passano per lo stesso collo di bottiglia, indistinti.

In **Fase 5** costruiremo la **self-attention**, che nasce per rompere esattamente
questa rigidità: invece di un contesto fisso e concatenato, ogni posizione della
sequenza deciderà **da sola, dinamicamente, a quali posizioni precedenti prestare
attenzione e quanto** — con pesi calcolati dai dati stessi. È il cuore del transformer,
e lo costruiremo da zero, una testa alla volta.

---

*Fine del capitolo Fase 4.*

---

<a name="fase-5"></a>
# Fase 5 — Self-attention: il cuore del transformer

Questo è il meccanismo che ha cambiato l'intelligenza artificiale (il paper del 2017
si intitolava "Attention is All You Need"). Lo costruiamo da zero, riga per riga, una
testa alla volta. Preparati: è più semplice di quanto la sua fama suggerisca — è, alla
fine, una media pesata intelligente.

📁 File: [`ronklm/models/attention.py`](ronklm/models/attention.py)

---

<a name="sec-5-0"></a>
## 5.0 L'idea in una frase

L'MLP aveva un contesto **rigido**: 8 caratteri, sempre, concatenati in posizioni
fisse ([sezione 4.8](#sec-4-8)). La self-attention lo sostituisce con qualcosa di
radicalmente più flessibile:

> **Ogni posizione della sequenza decide da sola, dinamicamente, a quali posizioni
> precedenti prestare attenzione e quanto — con pesi calcolati dai dati stessi a ogni
> forward.**

"Dinamicamente" è la parola chiave. Nell'MLP i pesi erano fissi dopo il training. Qui,
*quali* posizioni contano viene ricalcolato a ogni singolo input, in base al contenuto.
Una posizione che è una vocale finale può "cercare" l'ultima consonante; un carattere
dopo una virgola può cercare l'inizio della frase. E le stesse regole valgono per tutte
le posizioni: ciò che si impara si trasferisce ovunque nella sequenza.

---

<a name="sec-5-1"></a>
## 5.1 Query, Key, Value: la metafora onesta

Ogni posizione emette **tre** vettori, ottenuti dallo stesso embedding con tre matrici
diverse (tutte apprese — sono i `Linear` `query`, `key`, `value` del codice):

- **query** ("che cosa sto cercando"): es. *sono una vocale finale, cerco una
  consonante forte da qualche parte prima di me*.
- **key** ("come mi faccio trovare"): es. *sono una `r` a inizio sillaba, ecco la mia
  etichetta*.
- **value** ("l'informazione che consegno se qualcuno mi sceglie"): può essere diversa
  da come mi faccio trovare — chiave e contenuto sono ruoli distinti.

> **📖 Concetto: come si "cercano" a vicenda.** L'affinità tra la query della posizione
> `t` e la key della posizione `s` si misura con un **prodotto scalare** (`q · k`):
> due vettori simili danno prodotto grande, due diversi danno prodotto piccolo. Un
> prodotto scalare grande significa "questi due si cercavano": la posizione `t`
> attingerà molto dal value di `s`. Tutto — cosa cercare, come farsi trovare, cosa
> consegnare — è **appreso dal gradiente**: noi forniamo solo il meccanismo, il
> training riempie le tre matrici.

Nel codice:

```python
k = self.key(x)      # (B, T, head_size)   "come mi faccio trovare"
q = self.query(x)    # (B, T, head_size)   "cosa cerco"
v = self.value(x)    # (B, T, head_size)   "cosa consegno"
```

---

<a name="sec-5-2"></a>
## 5.2 I punteggi di affinità e lo scaling 1/√d

Calcoliamo l'affinità tra *ogni* coppia di posizioni: la query di `t` contro la key di
ogni `s`. È un prodotto matriciale `q @ kᵀ` che produce una matrice `(T, T)` di
punteggi (la cella `(t, s)` = quanto `t` è compatibile con `s`):

```python
scores = (q @ k.transpose(1, 2)) * self.scale     # (B, T, T)
```

Nota quel `* self.scale`, dove `scale = 1/√head_size`. È la famosa domanda d'esame sui
transformer, e la spieghiamo per bene.

> **📖 Concetto: perché dividere per √d.** Il prodotto scalare di due vettori di
> dimensione `d`, con componenti indipendenti a varianza 1, ha varianza `d` — cioè i
> punteggi crescono con la radice della dimensione, per pura statistica, non perché le
> affinità siano più nette. Punteggi grandi, dati in pasto a una softmax, la
> **saturano**: quasi tutta la probabilità finisce su una sola posizione, e nelle
> zone sature la derivata della softmax è ~0 → **il gradiente non passa** → le matrici
> Q e K smettono di imparare proprio all'inizio, quando dovrebbero imparare di più.
> Dividere per `√d` riporta la varianza dei punteggi a ~1 e tiene la softmax nella sua
> zona "viva". Una singola costante, messa lì per far fluire il gradiente.

---

<a name="sec-5-3"></a>
## 5.3 La maschera causale: la freccia del tempo

C'è un problema. Il nostro compito è predire il *prossimo* carattere. Se la posizione
`t` potesse attingere dalle posizioni *future*, durante il training vedrebbe **la
risposta** dentro l'input: loss bassissima, e modello inutile in generazione (dove il
futuro ancora non esiste). Dobbiamo impedirlo.

> **📖 Concetto: la maschera causale.** Prima della softmax, mettiamo a `−∞` tutti i
> punteggi `(t, s)` con `s > t` — cioè la posizione `t` non può guardare oltre sé
> stessa. Dopo la softmax, `exp(−∞) = 0`: quelle celle pesano esattamente zero, e la
> normalizzazione si redistribuisce automaticamente sulle sole posizioni lecite (`s ≤
> t`). "Causale" perché rispetta la causa-effetto temporale: il presente dipende solo
> dal passato.

```python
scores = scores.masked_fill(self.mask[:T, :T], -1e9)   # -inf sopra la diagonale
att = scores.softmax(axis=-1)                          # righe che sommano a 1
```

> **🔧 Perché `−∞` *prima* della softmax e non azzerare *dopo*.** Se azzerassimo dopo,
> le righe non sommerebbero più a 1 (non sarebbero più distribuzioni di probabilità).
> Mettere `−∞` prima fa sì che la softmax stessa ridistribuisca correttamente il peso
> solo sulle posizioni legali.

> **🔧 Il legame con la Fase 0.** È questa maschera che permette di addestrare *tutte*
> le T posizioni della finestra in un solo forward, ciascuna col proprio contesto
> legale — il trucco "T esempi al prezzo di uno" promesso in [Fase 0.8](#sec-0-8). (In
> gergo: un transformer fatto solo di questi blocchi mascherati si chiama
> "decoder-only"; i GPT sono tutti così.)

---

<a name="sec-5-4"></a>
## 5.4 La media pesata dei value

Ultimo passo: usare i pesi di attenzione per mescolare i value.

```python
out = att @ v      # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
```

> **📖 Concetto: l'output è una media pesata convessa.** Ogni riga di `att` è una
> distribuzione di probabilità sulle posizioni passate (somma 1). Quindi `att @ v` è,
> per ogni posizione, una **miscela dosata dei value del passato**: prendi tanto del
> value di `s` quanto la posizione `t` gli ha dato attenzione. La softmax qui non
> produce "probabilità di essere giusto" come in Fase 2, ma un **meccanismo di
> indirizzamento morbido e derivabile**: un po' come leggere da una memoria dove,
> invece di scegliere *una* cella, leggi un mix pesato di tutte. Ed è proprio la
> morbidezza (invece di una scelta netta) a rendere il meccanismo addestrabile per
> gradiente: una scelta rigida non sarebbe derivabile.

---

<a name="sec-5-5"></a>
## 5.5 Guardare dentro: la heatmap di attenzione

Una delle poche finestre *dirette* sull'interno di una rete neurale è proprio la
matrice di attenzione: si può letteralmente *vedere* a cosa guarda il modello. Abbiamo
addestrato brevemente un mini-LM a una testa e stampato la sua attenzione sulla frase
`"Geppetto guarda"` (ogni riga `t` mostra quanto attinge da ogni posizione `s ≤ t`; i
puntini sono il futuro mascherato):

```
        G    e    p    p    e    t    t    o    _    g    u    a    r    d    a
  G | 1.00   .    .    .    .    .    .    .    .    .    .    .    .    .    .
  e | 0.21 0.79   .    .    .    .    .    .    .    .    .    .    .    .    .
  p | 0.01 0.00 0.99   .    .    .    .    .    .    .    .    .    .    .    .
  p | 0.01 0.00 0.50 0.50   .    .    .    .    .    .    .    .    .    .    .
  t | 0.00 0.00 0.00 0.00 0.00 1.00   .    .    .    .    .    .    .    .    .
  o | 0.02 0.18 0.00 0.00 0.18 0.00 0.00 0.62   .    .    .    .    .    .    .
  _ | 0.00 0.01 0.01 0.01 0.01 0.00 0.00 0.00 0.96   .    .    .    .    .    .
  u | 0.01 0.00 0.00 0.00 0.00 0.01 0.01 0.01 0.00 0.05 0.90   .    .    .    .
```

Due cose saltano all'occhio, e sono la prova che il meccanismo funziona:

1. **È rigorosamente triangolare.** Tutto ciò che sta a destra della diagonale (il
   futuro) è vuoto: la maschera causale funziona.
2. **Ha imparato pattern sensati.** La seconda `p` di "Geppetto" divide l'attenzione a
   metà tra le due `p` (0.50/0.50); la `u` guarda fortissimo la `g` che la precede
   (0.90, il gruppo "gu"); molte posizioni attingono al carattere immediatamente
   precedente. Nessuno gliel'ha detto: l'ha imparato dal gradiente.

Questa immagine rende concreto tutto il discorso Q/K/V come nessuna formula può fare.

---

<a name="sec-5-6"></a>
## 5.6 Perché batte i limiti dell'MLP (e cosa manca ancora)

L'attention risolve i tre limiti dell'MLP ([sezione 4.8](#sec-4-8)): il contesto è
*dinamico* (ricalcolato a ogni input, non fisso); le stesse matrici Q/K/V valgono per
tutte le posizioni (ciò che si impara si trasferisce); una posizione lontana è
raggiungibile in un passo, non attraverso un collo di bottiglia.

**Ma — e qui va detta la verità con onestà da ricercatore** — la nostra AttentionLM a
una testa raggiunge NLL val **2.328**, appena sotto il bigram (2.35) e *peggio*
dell'MLP (1.90). Come mai? Per due motivi che le prossime fasi risolveranno:

> **📖 L'attention da sola è cieca all'ordine.** Il meccanismo è, per costruzione,
> un'operazione su *insiemi*: se permuti i token di input, i prodotti `q · k` non
> cambiano. Nulla, nel meccanismo, sa che un token viene *prima* di un altro. Ma "ma
> la" ≠ "la ma": l'ordine è metà del linguaggio. Manca l'informazione di **posizione**,
> che aggiungeremo in Fase 7 (positional embedding). Senza di essa, la nostra testa può
> fare poco più che un bigram sofisticato.

> **📖 Una testa sola vede poco.** Una singola softmax produce *un* solo "sguardo" per
> posizione. Servono più sguardi paralleli (multi-head), più profondità (blocchi in
> pila), e la parte che *elabora* dopo aver raccolto (il feed-forward). Arrivano tutti
> in Fase 6.

In altre parole: in Fase 5 abbiamo costruito e *capito* il mattone fondamentale, e
verificato che funziona (causale, addestrabile, ispezionabile). La sua potenza si
libererà quando lo comporremo. È esattamente il percorso giusto: prima il pezzo, poi
la macchina.

---

<a name="sec-5-7"></a>
## 5.7 Glossario Fase 5 / cosa arriva in Fase 6

Nuovi termini:

- **Self-attention**: meccanismo per cui ogni posizione aggrega informazione dalle
  altre posizioni della *stessa* sequenza, con pesi appresi e dinamici.
- **Query / Key / Value (Q/K/V)**: i tre vettori appresi per posizione — cosa cerco /
  come mi faccio trovare / cosa consegno.
- **Punteggio di attenzione**: il prodotto scalare `q · k`, misura di affinità.
- **Scaling `1/√d`**: divisione dei punteggi per stabilizzare la softmax.
- **Maschera causale**: azzerare l'attenzione verso il futuro (`s > t`).
- **Decoder-only**: un transformer fatto solo di blocchi con maschera causale (i GPT).
- **Attention map / heatmap**: la matrice `(T, T)` dei pesi di attenzione, ispezionabile.

**In Fase 6** assembleremo il **blocco transformer**, l'unità che i GPT ripetono N
volte. Introdurremo tre cose che rendono possibile *impilare* l'attention in
profondità: **multi-head** (più sguardi paralleli), il **feed-forward** (la parte che
*elabora* dopo che l'attention ha *comunicato*), e — soprattutto — le **connessioni
residue** e il **LayerNorm**, ovvero la risposta alla domanda "perché le reti profonde
sono addestrabili?", che non è affatto ovvia.

---

*Fine del capitolo Fase 5.*

---

<a name="fase-6"></a>
# Fase 6 — Il blocco Transformer: perché le reti profonde funzionano

In Fase 5 abbiamo costruito una testa di attention. Ma una testa da sola non basta:
serve il **blocco**, l'unità che i GPT ripetono decine di volte. E soprattutto serve
capire tre ingredienti che rendono possibile *impilare* l'attention in profondità —
tra cui il più importante e meno ovvio di tutto il deep learning: **perché una rete
profonda è addestrabile**.

📁 File: [`ronklm/models/block.py`](ronklm/models/block.py),
[`ronklm/nn.py`](ronklm/nn.py) (LayerNorm)

---

<a name="sec-6-0"></a>
## 6.0 L'unità che i GPT ripetono

Un GPT è, essenzialmente, una pila di blocchi identici. Ogni blocco ha questa forma
(dal codice):

```python
x = x + self.attn(self.ln1(x))    # comunica
x = x + self.ffn(self.ln2(x))     # elabora
```

Quattro righe che contengono quattro idee: **multi-head attention** (`attn`),
**feed-forward** (`ffn`), **LayerNorm** (`ln1`, `ln2`), e le **connessioni residue**
(i due `x + ...`). Vediamole una per una — ognuna risolve un problema preciso.

---

<a name="sec-6-1"></a>
## 6.1 Multi-head: più sguardi in parallelo

In Fase 5 avevamo *una* testa. Il blocco ne usa diverse in parallelo
(`MultiHeadAttention`): ognuna con le sue Q/K/V, e con dimensione `n_embd / n_head`
(così il costo totale resta costante). Gli output delle teste si **concatenano** e si
riproiettano con un `Linear` finale.

> **📖 Concetto: perché più teste invece di una grande.** Una singola softmax produce
> *una* distribuzione di attenzione per posizione: un solo "sguardo". Ma a una
> posizione possono servire *contemporaneamente* informazioni diverse da posti diversi:
> il carattere precedente (per l'ortografia), l'inizio della parola (per la
> morfologia), l'apertura di virgolette molto indietro (per chiudere un dialogo). Teste
> separate = sguardi paralleli, ognuno specializzabile in modo indipendente. Empiricamente,
> a parità di budget, più sguardi piccoli battono un solo sguardo grande.

> **🔧 Nel codice: perché la proiezione finale dopo la concatenazione.** Le teste
> producono fette separate del vettore output. Senza un `Linear` finale, quelle fette
> resterebbero segregate. La proiezione le **mescola**, permettendo alle informazioni
> raccolte da teste diverse di combinarsi. (La concatenazione l'abbiamo aggiunta a
> ronkgrad come operazione `cat`, col suo backward che ri-spezza il gradiente sui pezzi
> — verificata col gradient check.)

---

<a name="sec-6-2"></a>
## 6.2 Feed-forward: comunicare e poi pensare

Dopo l'attention viene il **feed-forward** (`FeedForward`): due `Linear` con una
non-linearità `gelu` in mezzo, che *espande* la dimensione di 4 volte e poi la
ricomprime. Applicato a ogni posizione indipendentemente.

> **📖 Concetto: la divisione dei ruoli.** L'attention *sposta* informazione tra
> posizioni, ma la elabora poco (in fondo fa medie pesate). La FFN è il suo complemento
> esatto: non guarda nessun'altra posizione, ma **elabora** — con una vera
> non-linearità — ciò che l'attention ha raccolto. Il ritmo del transformer è:
> **comunica** (attention) → **pensa** (FFN) → comunica → pensa… A ogni blocco, le
> posizioni prima si scambiano informazione, poi ci ragionano su.

> **📖 Perché l'espansione 4×.** Il fattore 4 dà alla FFN uno spazio interno più largo
> dove computare prima di ricomprimere. È la convenzione empirica di *tutti* i GPT (e,
> curiosità, nei modelli reali la FFN contiene ~2/3 di tutti i parametri: è lì che si
> ritiene risieda gran parte della "conoscenza" memorizzata).

---

<a name="sec-6-3"></a>
## 6.3 LayerNorm: un riferimento fisso a ogni strato

Il `LayerNorm` (in [`nn.py`](ronklm/nn.py)) normalizza ogni vettore a media 0 e
varianza 1, poi lo riscala con due parametri appresi `gamma` e `beta`.

> **📖 Concetto: perché normalizzare.** In una rete profonda, la scala delle
> attivazioni di uno strato dipende da *tutti* gli strati precedenti — che stanno
> cambiando durante il training. Ogni strato insegue un bersaglio mobile, e le scale
> possono esplodere o collassare strada facendo. LayerNorm ristabilisce, a ogni blocco,
> un punto di riferimento fisso (media 0, varianza 1): il training diventa stabile e si
> possono usare learning rate più alti. (Lo verifichiamo con un test: dato un input con
> media 5 e scala 3, l'output ha media ~0 e deviazione ~1.)

> **📖 Perché gamma e beta.** La normalizzazione pura è troppo autoritaria: toglie alla
> rete anche la libertà di *volere* una scala diversa, se le serve. I due parametri
> appresi gliela restituiscono, partendo da un default sano (`gamma=1`, `beta=0`).

> **📖 Perché LayerNorm e non BatchNorm** (che magari incontrerai altrove). BatchNorm
> normalizza *attraverso il batch*: accoppia esempi indipendenti tra loro, si comporta
> diversamente in training e in generazione (dove il batch può essere 1), ed è un
> vivaio storico di bug. LayerNorm normalizza ogni posizione *per conto suo*: nessun
> accoppiamento, identica in training e inferenza. Per le sequenze non c'è partita, e i
> transformer usano LayerNorm ovunque.

Nel nostro codice LayerNorm è costruito *interamente* da operazioni primitive (`mean`,
`var`, sottrazione, radice, moltiplicazione): quindi il suo backward — che è il più
intricato che avremmo dovuto derivare a mano, perché media e varianza dipendono da
tutti gli elementi del vettore — arriva **gratis** dall'autograd. È un altro trionfo
della Fase 3.

---

<a name="sec-6-4"></a>
## 6.4 Le connessioni residue: l'autostrada del gradiente

Ecco l'idea più importante della fase, e forse la più importante del deep learning
moderno dopo la backpropagation stessa. Guarda di nuovo la struttura del blocco:

```python
x = x + self.attn(self.ln1(x))    # NON  x = attn(...)
x = x + self.ffn(self.ln2(x))     # NON  x = ffn(...)
```

Nota quel `x + `. Invece di *sostituire* `x` con l'output dello strato, lo **sommiamo**
a `x`. Questa è la **connessione residua** (o "skip connection"), e sembra un dettaglio
banale. Non lo è: è ciò che rende possibile addestrare reti profonde.

> **📖 Concetto: il problema del gradiente che svanisce.** Per raggiungere i *primi*
> strati di una rete profonda, il gradiente deve attraversare all'indietro, in catena,
> tutti gli strati successivi, venendo moltiplicato a ogni passaggio. Il prodotto di
> tanti fattori minori di 1 **svanisce esponenzialmente** (o esplode, se maggiori di 1).
> Risultato: prima del 2015, le reti oltre ~20 strati *peggioravano* aggiungendo strati,
> perché i primi non ricevevano più segnale utile.

> **📖 La soluzione, e perché funziona.** Scrivere `x = x + f(x)` invece di `x = f(x)`.
> Ricordi il backward della somma dalla [sezione 3.3](#sec-3-3)? — *distribuisce il
> gradiente invariato* a entrambi i rami. Quindi il ramo `x` "nudo" è un'**autostrada**:
> il gradiente della loss arriva ai primi strati **intatto**, qualunque cosa facciano
> gli strati `f` in mezzo. È letteralmente il motivo per cui insistevamo, in Fase 3,
> sul fatto che "la somma distribuisce il gradiente": questa riga di codice è il perché.

C'è anche un secondo beneficio, per l'*apprendimento*: ogni blocco parte dal
comportamento "non faccio niente" (se `f(x) ≈ 0`, il blocco è l'identità: l'input passa
inalterato) e impara **correzioni incrementali** a un segnale che scorre, invece di
dover ricostruire tutto da capo. Impilare 12 blocchi diventa sicuro: al peggio, i
blocchi inutili restano vicini all'identità e non fanno danni.

---

<a name="sec-6-5"></a>
## 6.5 Pre-norm vs post-norm

Un dettaglio nella posizione del LayerNorm che ha conseguenze grandi. Noi lo mettiamo
*dentro* il ramo, **prima** di attn/ffn (`x + attn(ln1(x))`): è la variante **pre-norm**.
Il paper originale del 2017 lo metteva *dopo* la somma (post-norm).

> **📖 Perché pre-norm.** Con la post-norm, il LayerNorm sta *sull'autostrada* e
> rinormalizza il segnale a ogni blocco, disturbando proprio quel flusso pulito del
> gradiente che le residual avevano costruito. Con la pre-norm, l'autostrada `x` resta
> intonsa da input a output, e il LayerNorm agisce solo *dentro* i rami di calcolo.
> Empiricamente: la post-norm richiede warmup delicati per non divergere, la pre-norm è
> molto più stabile. GPT-2 e tutti i suoi successori sono pre-norm; anche noi.

---

<a name="sec-6-6"></a>
## 6.6 Il blocco è impilabile

Una proprietà che sembra tecnica ma è il punto di tutto: il blocco prende un input di
forma `(B, T, n_embd)` e restituisce un output della **stessa forma** `(B, T, n_embd)`.

> **🔧 Perché conta.** Siccome ingresso e uscita hanno la stessa forma, i blocchi si
> possono **incastrare uno dopo l'altro** all'infinito: l'output del blocco 1 è un input
> valido per il blocco 2, e così via. È ciò che permette di costruire un GPT "profondo"
> semplicemente ripetendo `Block` N volte. Lo verifichiamo con un test
> (`test_block_preserves_shape`), e con un altro (`test_block_all_params_get_gradient`)
> controlliamo che *ogni* parametro del blocco riceva gradiente — il modo meccanico di
> scoprire un componente scollegato per errore dal grafo.

Con la Fase 6 abbiamo in mano il **mattone completo del GPT**, testato e compreso.
Manca solo assemblarlo.

---

<a name="sec-6-7"></a>
## 6.7 Glossario Fase 6 / cosa arriva in Fase 7

Nuovi termini:

- **Multi-head attention**: più teste di attention in parallelo, concatenate e
  riproiettate.
- **Feed-forward (FFN)**: la sotto-rete che elabora ogni posizione dopo l'attention
  (espansione 4×, gelu).
- **LayerNorm**: normalizzazione per-posizione a media 0/varianza 1, con `gamma`/`beta`
  appresi.
- **Connessione residua (skip connection)**: `x + f(x)`, l'autostrada che fa arrivare
  il gradiente intatto ai primi strati.
- **Pre-norm / post-norm**: LayerNorm dentro il ramo (prima di attn/ffn) vs dopo la
  somma; pre-norm è più stabile.
- **Blocco transformer**: l'unità `attn → ffn` con residual e norm, impilabile.

**In Fase 7** metteremo tutto insieme nel **GPT completo**: uno stack di blocchi, con
in più i **positional embedding** (che risolvono la cecità all'ordine notata in [Fase
5.6](#sec-5-6)) e la testa finale. E costruiremo la **generazione autoregressiva** con
le sue manopole — `temperature` e `top-k`. Sarà **RonkLM v1**: un vero GPT giocattolo
che scrive pseudo-Collodi, carattere per carattere.

---

*Fine del capitolo Fase 6.*

---

<a name="fase-7"></a>
# Fase 7 — RonkLM: il GPT completo

Ci siamo. Mettiamo insieme tutti i pezzi costruiti finora in un vero transformer
decoder-only, colmiamo l'ultimo buco concettuale (l'ordine delle parole) e costruiamo
la generazione con le sue manopole. Il risultato è **RonkLM v1**: un GPT giocattolo,
scritto interamente a mano sopra `ronkgrad`, che scrive pseudo-Collodi carattere per
carattere.

📁 File: [`ronklm/models/gpt.py`](ronklm/models/gpt.py),
[`ronklm/generate.py`](ronklm/generate.py)

---

<a name="sec-7-0"></a>
## 7.0 Assemblare tutti i pezzi

L'architettura del GPT è, letteralmente, la somma di ciò che abbiamo costruito:

```
indici (B, T)
  │
  ▼  token embedding (Fase 4)     +   positional embedding (nuovo)
  (B, T, C)                            (T, C)
  │
  ▼  Block × n_layer (Fase 6)     ← attention (5) + feed-forward + residual + norm
  (B, T, C)
  │
  ▼  LayerNorm finale (Fase 6)
  │
  ▼  Linear → vocab (Fase 4)
  (B, T, vocab)   = logits del prossimo carattere a OGNI posizione
```

Nel codice il forward è di una brevità che, dopo tutto questo percorso, quasi
commuove:

```python
tok = self.tok_emb(idx)          # (B, T, C)   significato dei caratteri
pos = self.pos_emb(np.arange(T)) # (T, C)      posizione nella sequenza
x = tok + pos                    # (B, T, C)   somma
for blk in self.blocks:
    x = blk(x)                   # attenzione + elaborazione, N volte
x = self.ln_f(x)                 # normalizzazione finale
return self.head(x)              # (B, T, vocab)
```

Ogni riga è un pezzo che conosciamo dall'interno. Non c'è una sola operazione di cui
non sappiamo derivare il gradiente. Questo è il senso dell'intero Percorso A.

---

<a name="sec-7-1"></a>
## 7.1 Il positional embedding: dare un ordine alla sequenza

C'è un solo ingrediente nuovo, e risolve il difetto notato in [Fase 5.6](#sec-5-6):
l'attention è **cieca all'ordine**.

> **📖 Concetto: l'attention è un'operazione su insiemi.** Se permuti i token di input
> (e le maschere), i prodotti `q · k` non cambiano: nulla, nel meccanismo di attention,
> sa che un token viene *prima* di un altro. Ma "ma la" ≠ "la ma": l'ordine è metà del
> linguaggio. All'attention manca il senso della posizione.

La soluzione è elegante: una **seconda tabella di embedding**, indicizzata non dal
carattere ma dalla **posizione** (0, 1, 2, …, block_size−1). La riga `t` di questa
tabella è la "firma" appresa della posizione `t`. La sommiamo al token embedding:

```python
x = tok + pos     # ogni carattere porta con sé "chi sono" E "dove sono"
```

Così ogni vettore in ingresso ai blocchi codifica due informazioni sovrapposte:
*quale* carattere è (dal token embedding) e *in quale posizione* si trova (dal
positional embedding). Le teste di attention possono ora imparare pattern posizionali
("guarda il carattere immediatamente precedente") oltre che di contenuto ("cerca una
vocale").

> **📖 Perché sommare e non concatenare.** La somma mantiene la dimensione (i blocchi
> restano identici e impilabili). E con vettori appresi in uno spazio ampio, la rete
> ha campo libero di dedicare "direzioni" diverse ai due tipi di informazione se le
> serve. Empiricamente funziona bene quanto la concatenazione, a costo zero. (I LLM
> moderni usano schemi più sofisticati — RoPE — ma l'embedding posizionale appreso è
> quello di GPT-2 ed è perfetto per capire il problema.)

> **📖 Perché la LayerNorm finale.** Dopo l'ultimo blocco, il segnale è la somma di
> tutti i contributi residui accumulati, su una scala non controllata. La testa che
> produce i logit lavora molto meglio su un segnale rinormalizzato. È lo standard di
> GPT-2, coerente con la logica pre-norm della [sezione 6.5](#sec-6-5).

---

<a name="sec-7-2"></a>
## 7.2 Lo stack di blocchi e la testa finale

I blocchi (Fase 6) si impilano semplicemente in una lista, perché — lo avevamo
verificato — preservano la forma `(B, T, C)`: l'output di uno è l'input valido del
successivo. Più blocchi = più profondità = più capacità di comporre trasformazioni
complesse. La `head` finale (un `Linear`) proietta ogni vettore di posizione sui
`vocab` logit del prossimo carattere.

Il forward produce logit per **ogni** posizione `(B, T, vocab)`, e la loss è la
cross-entropy su **tutte** le `B·T` posizioni contemporaneamente:

```python
return cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))
```

> **🔧 Qui si raccoglie tutto ciò che si era seminato.** Il "T esempi al prezzo di
> uno" preparato in [Fase 0.8](#sec-0-8) (Y = X spostato) e reso legale dalla maschera
> causale in [Fase 5.3](#sec-5-3): ogni batch da `(B, T)` produce `B·T` esempi di
> addestramento in un solo forward. Un batch `(16, 32)` = 512 predizioni per passo.
> Senza questa struttura, addestrare transformer non sarebbe economicamente possibile.

---

<a name="sec-7-3"></a>
## 7.3 La generazione: temperature e top-k

Un GPT addestrato *predice*; per farlo *scrivere* serve la generazione autoregressiva
([`generate.py`](ronklm/generate.py)): forward sugli ultimi `block_size` caratteri,
prendi i logit dell'**ultima** posizione, campiona il prossimo carattere, appendilo,
ripeti.

> **🔧 Perché si troncano gli ultimi block_size caratteri.** La tabella posizionale ha
> esattamente `block_size` righe e le matrici di attenzione sono `T×T`: oltre quella
> finestra il modello semplicemente non è definito. È il famoso **limite di contesto**
> dei LLM — ora sai da quali due tensori nasce.

Due manopole controllano *come* si campiona:

> **📖 Temperature.** Si dividono i logit per `τ` prima della softmax. Per
> l'esponenziale della softmax: `τ < 1` *allarga* le differenze tra i logit →
> distribuzione più appuntita → testo più conservativo e ripetitivo; `τ > 1` le
> comprime → più vario e più sgangherato; `τ → 0` = greedy (sceglie sempre il massimo:
> degenere e ciclico, come previsto in [Fase 1.3](#sec-1-3)). Non cambia *l'ordine*
> delle preferenze del modello: cambia quanto ci si azzarda a deviare dalla prima
> scelta.

> **📖 Top-k.** La coda della distribuzione (decine di caratteri a probabilità minuscola
> ma non nulla) ogni tanto viene comunque pescata, e un singolo carattere assurdo (una
> `%` in mezzo a una parola) può far *deragliare tutto il seguito*: il modello non ha
> mai visto contesti con quel carattere lì, e genera spazzatura da spazzatura (è
> l'**errore composto**: il testo esce dalla distribuzione su cui il modello è stato
> addestrato). Top-k taglia la coda: tiene solo i `k` logit migliori, azzera gli altri,
> rinormalizza. Insieme, temperature e top-k sono le stesse due manopole dei LLM di
> produzione.

---

<a name="sec-7-4"></a>
## 7.4 Il checkpoint autosufficiente

Il metodo `save` scrive in un unico file `.npz`: **i pesi + la config + il
vocabolario**.

> **📖 Perché tutto insieme.** Un modello char-level ricaricato con un vocabolario
> diverso da quello di training produce spazzatura *deterministica* e difficilissima da
> diagnosticare: i pesi sono "giusti" ma parlano un'altra mappa di indici (il carattere
> 30 ora è `q` invece di `a`). E una config diversa non fa nemmeno combaciare le shape.
> Il checkpoint deve essere **autosufficiente**: pesi, mappa dei caratteri e
> architettura, sempre insieme. (Abbiamo un test che salva e ricarica, e verifica che i
> logit siano identici bit per bit: la prova che il modello ricaricato *è* lo stesso.)

---

<a name="sec-7-5"></a>
## 7.5 RonkLM v1: i numeri e il testo

Abbiamo addestrato un GPT piccolo (**3 layer, 4 teste, n_embd 64, block_size 32, ~160k
parametri**) per 3000 passi. Ecco la spina dorsale sperimentale del progetto, la tabella
delle NLL di validation che cresce a ogni modello:

```
modello                      NLL val    perplexity   cosa ha aggiunto
─────────────────────────────────────────────────────────────────────────────
uniforme (tira a caso)       4.234       69          niente
bigram (conteggio, Fase 1)   2.346       10.4        distribuzione sul prossimo char
MLP (contesto 8, Fase 4)     1.897        6.7        embedding + contesto + hidden
GPT (RonkLM v1, Fase 7)      1.632        5.1        attention + posizione + profondità
```

Ogni riga è una *idea* che abbiamo aggiunto e il miglioramento misurabile che ha
comprato. Dal caos (69 caratteri equiprobabili) a un modello che, in media, esita tra
~5 caratteri: la sorpresa per carattere si è più che dimezzata rispetto al bigram.

E il testo? Ecco RonkLM v1 che completa il prompt "Pinocchio " (temperature 0.6, top-k 20):

```
Pinocchio a stare il mio tirò Pinocchio, con mi pesse diretto dalla spaggio dettorna
da di farò a parere di gallina di grande di sè: — Ma in poco da nottere a casa
mozzare a un bel pesce di mani un po' di piedi.
— No, rangia si disse:
— Non la vocina le portice di cas
```

Facciamo il punto con onestà da ricercatore. **Non è italiano coerente** — e non
poteva esserlo: 160k parametri, mezzo megabyte di testo, un modello che sta su una CPU.
Ma guarda cosa è comparso rispetto al bigram sillabico di [Fase 1](#sec-1-6): quasi
**tutte parole vere** (`stare`, `pesce`, `casa`, `piedi`, `grande`, `disse`), il nome
`Pinocchio` scritto giusto e ripetuto, la **struttura dei dialoghi** collodiani (`— Ma
…`, `— No, … si disse:`), la punteggiatura al posto giusto, gli a-capo dei paragrafi.
Manca il senso globale — la frase parte e si perde — ma la *forma* dell'italiano
narrativo c'è. Questo è esattamente il risultato corretto a questa scala: lo scopo del
Percorso A era **vedere la macchina imparare e capirne ogni pezzo**, non ottenere uno
scrittore (quello è l'obiettivo del Percorso B, con 50M di parametri e gigabyte di
testo).

> **🔧 Cosa fa la temperature, visto dal vivo.** A temperature 0.6 (sopra) il testo è
> più prudente e ripetitivo; alzandola a 0.9 diventa più avventuroso e sgangherato
> (più parole inventate, più varietà). Sono le due estremità del compromesso
> "probabile vs vario" di cui parlavamo fin da [Fase 1.3](#sec-1-3).

**Questo è RonkLM v1**: un GPT completo, scritto interamente a mano — dal gradiente
della cross-entropy al motore di autograd, dalla self-attention al positional
embedding — senza una sola riga di PyTorch. Ogni numero che produce, sappiamo da dove
viene.

---

<a name="sec-7-6"></a>
## 7.6 Glossario Fase 7 / cosa arriva in Fase 8

Nuovi termini:

- **Positional embedding**: tabella appresa indicizzata dalla posizione, sommata al
  token embedding per dare al modello il senso dell'ordine.
- **Decoder-only / GPT**: transformer fatto di soli blocchi con maschera causale.
- **Generazione autoregressiva**: produrre testo un token alla volta, rialimentando
  l'output.
- **Temperature**: fattore che rende il campionamento più conservativo (<1) o più vario
  (>1).
- **Top-k**: tenere solo i k caratteri più probabili prima di campionare.
- **Limite di contesto**: la lunghezza massima di sequenza (= block_size), fissata dalla
  tabella posizionale.
- **Checkpoint**: file che salva pesi + vocabolario + config, autosufficiente.

**In Fase 8** passeremo da "gira" a "gira bene e si usa": un training loop robusto (con
learning-rate **schedule**: warmup + cosine decay), una **CLI** per addestrare e
generare da riga di comando in modo riproducibile, e una serie di **esperimenti
documentati** su come ogni iperparametro (profondità, contesto, temperature) cambia il
risultato. È la chiusura del Percorso A.

---

*Fine del capitolo Fase 7.*

---

<a name="fase-8"></a>
# Fase 8 — Training serio, CLI ed esperimenti

RonkLM v1 gira. Questa fase lo porta da "gira" a "gira bene, si misura e si usa": un
training loop robusto, una CLI per addestrare e generare in modo riproducibile, e una
serie di esperimenti che mostrano *cosa compra* ogni pezzo del transformer. È la
chiusura del Percorso A.

📁 File: [`ronklm/train.py`](ronklm/train.py),
[`scripts/train_ronklm.py`](scripts/train_ronklm.py)

---

<a name="sec-8-0"></a>
## 8.0 Da "gira" a "gira bene e si usa"

Fin qui abbiamo addestrato con cicli scritti a mano dentro script di prova. Va bene per
capire, ma per *usare* il modello e per fare esperimenti seri servono tre cose che
mancavano: un modo riproducibile di lanciare un training (la CLI), un loop che si
misura e salva il modello migliore (non l'ultimo), e un controllo intelligente del
learning rate nel tempo (lo schedule). Le costruiamo qui.

---

<a name="sec-8-1"></a>
## 8.1 La CLI: esperimenti riproducibili dal comando

Lo script `train_ronklm.py` offre due sottocomandi, `train` e `generate`:

```bash
python scripts/train_ronklm.py train --steps 3000 --n-layer 3 --n-embd 64 --out ckpt.npz
python scripts/train_ronklm.py generate --ckpt ckpt.npz --prompt "Pinocchio " --temperature 0.8 --top-k 20
```

> **📖 Perché una CLI e non un notebook che si modifica.** Ogni esperimento resta
> riproducibile dal suo **comando**: il comando, insieme al seed fisso, individua
> univocamente il run. Se modificassi il codice a ogni prova, la confrontabilità tra
> esperimenti — che è tutto il punto della [sezione 8.4](#sec-8-4) — andrebbe persa.
> Il comando *è* la documentazione dell'esperimento.

---

<a name="sec-8-2"></a>
## 8.2 Il loop robusto: eval, best-checkpoint, clipping

Il `train()` in [`train.py`](ronklm/train.py) aggiunge tre pratiche essenziali al ciclo
base della [Fase 2.4](#sec-2-4).

> **📖 Eval mediata su più batch.** La loss di un singolo batch è rumorosa
> ([Fase 2.4](#sec-2-4)): su un batch fortunato sembrerebbe un progresso che non esiste.
> Misuriamo la NLL di validation mediando su 20–25 batch *fissi* (seed fisso): un numero
> stabile su cui prendere decisioni.

> **📖 Salvare il *best*, non l'ultimo.** Se il modello inizia a overfittare
> ([Fase 4.6](#sec-4-6)), l'ultimo checkpoint è *peggiore* di uno intermedio. Salviamo
> il modello con la miglior NLL di validation vista finora: è la forma più semplice di
> "early stopping".

> **📖 Gradient clipping.** Su tanti batch, prima o poi qualcuno produce un gradiente
> anomalo (un pezzo di testo strano, una coincidenza numerica). Un singolo passo gigante
> può buttare il modello in una zona da cui non si riprende (un "loss spike"). Il
> clipping taglia la *norma globale* del gradiente a una soglia: un'assicurazione a
> costo praticamente nullo. Lo verifichiamo con un test (limita la norma, ma lascia
> intatti i gradienti piccoli).

---

<a name="sec-8-3"></a>
## 8.3 Il learning-rate schedule: warmup e cosine decay

Il learning rate non resta fisso: parte da ~0, sale linearmente (warmup) per i primi
passi, poi scende dolcemente seguendo un coseno fino a un valore minimo.

> **📖 Perché il warmup — legato ad Adam ([Fase 4.5](#sec-4-5)).** Nei primissimi passi
> le medie mobili di Adam sono stime basate su pochissimi campioni; in particolare il
> secondo momento (che sta al *denominatore* del passo) può essere sottostimato → passi
> enormi in direzioni rumorose, su una rete appena inizializzata che è nel suo punto più
> fragile. Il warmup tiene i passi piccoli finché le stime non maturano.

> **📖 Perché il decay finale.** A fine training si è vicini a un minimo; con passi
> grandi si *orbita* attorno al minimo senza entrarci (l'ampiezza dell'oscillazione è
> proporzionale al lr). Ridurre il lr permette di *depositarsi* nel minimo. Il coseno è
> la forma dolce standard, senza salti bruschi. (Lo verifichiamo con due test: il warmup
> sale, il coseno scende fino a `min_lr`.)

---

<a name="sec-8-4"></a>
## 8.4 Il laboratorio: cosa compra ogni iperparametro

Gli iperparametri non si capiscono leggendone la definizione: si capiscono
*toccandoli* e guardando l'effetto. Abbiamo fatto una serie di esperimenti
**one-factor-at-a-time** (cambiando *una* cosa per volta), tutti con lo stesso seed e
lo stesso numero di passi, così le differenze sono attribuibili a quel solo fattore.

**Esperimento 1 — la profondità** (quanti blocchi in pila; block_size 32, 1000 passi):

```
n_layer   parametri   NLL val
   1        36 k       2.030
   2        65 k       1.942
   4       121 k       1.886
```

Trend nettissimo e monotono: **più blocchi = NLL più bassa**. Ogni blocco aggiunge un
giro di "comunica (attention) → pensa (FFN)", e componendone di più il modello
rappresenta trasformazioni più ricche. È la conferma sperimentale del perché i GPT sono
"profondi".

**Esperimento 2 — il contesto** (quanti caratteri guarda; n_layer 2, 1000 passi):

```
block_size   parametri   NLL val
    16         64 k       1.944
    32         65 k       1.942
    64         66 k       1.986   ← peggio!
```

Qui un risultato **controintuitivo e istruttivo**: a parità di training (1000 passi),
il contesto più lungo (64) va *peggio* di quello medio (32). Come mai? Un contesto più
lungo significa più posizioni da imparare a usare e un'attenzione più grande da tarare:
con un *budget di training corto* non fa in tempo a ripagare l'investimento. Non è che
"più contesto è peggio" in assoluto — è che più contesto ha bisogno di *più passi* per
rendere. È esattamente il tipo di verità che si scopre solo *misurando*, e che un
tutorial superficiale ("più grande è meglio") nasconderebbe.

> **Nota onesta sul metodo.** Questi sono run brevi (1000 passi) e piccoli, scelti per
> girare in fretta su CPU: mostrano il *trend*, non la NLL più bassa raggiungibile.
> RonkLM v1 ([Fase 7.5](#sec-7-5)), addestrato più a lungo, arriva a 1.632. E ci sono
> esperimenti che *non* abbiamo fatto (larghezza `n_embd`, numero di teste, schedule
> on/off, weight decay): li lasciamo come esercizio, segnalando esplicitamente che
> mancano — un laboratorio non deve mai far credere di aver coperto tutto.

> **📖 Perché one-factor-at-a-time.** Cambiando due cose insieme non si saprebbe a
> quale attribuire la differenza. È meno efficiente di una ricerca su griglia completa,
> ma qui l'obiettivo non è trovare l'ottimo: è *capire il contributo di ogni idea*. È il
> metodo sperimentale applicato al nostro stesso modello. (Nota onesta: questi run sono
> brevi e piccoli, scelti per essere eseguibili in fretta su CPU; servono a mostrare il
> **trend**, non a raggiungere la NLL più bassa possibile. Un training più lungo, come
> quello di RonkLM v1 in [Fase 7.5](#sec-7-5), arriva più in basso.)

---

<a name="sec-8-5"></a>
## 8.5 Chiusura del Percorso A

Con la Fase 8 il Percorso A è **completo**. Fermiamoci a guardare cosa abbiamo
costruito, partendo da NumPy e nient'altro:

- un **tokenizer** e una pipeline di dati (Fase 0);
- la **loss** cross-entropy e il primo modello, il bigram (Fase 1);
- la **backpropagation**, derivata a mano (Fase 2);
- un intero **motore di autograd**, `ronkgrad`, verificato al bit (Fase 3);
- gli **embedding**, gli strati nascosti, l'ottimizzatore **AdamW** (Fase 4);
- la **self-attention** causale (Fase 5);
- il **blocco transformer** con multi-head, feed-forward, LayerNorm e residual (Fase 6);
- il **GPT completo**, RonkLM v1, con positional embedding e generazione (Fase 7);
- il **training serio** con schedule, CLI ed esperimenti (Fase 8).

Ogni singola operazione che RonkLM esegue — ogni moltiplicazione, ogni gradiente, ogni
`softmax` — l'abbiamo scritta e capita noi. Quando in un LLM vero incontrerai
`nn.MultiheadAttention`, `F.cross_entropy`, `torch.optim.AdamW`, `loss.backward()`, non
vedrai più scatole nere: vedrai cose di cui possiedi la versione fatta a mano.

**Cosa arriva dopo (Percorso B).** RonkLM v1 scrive *forma* di italiano ma non *senso*,
perché è minuscolo. Per arrivare a un modello che scrive frasi e paragrafi sensati
servono ~50M di parametri, gigabyte di testo e una GPU — cose che NumPy su CPU non può
reggere (il perché quantitativo è in [Fase I.2](#sec-i-2), se rileggi il piano). Il
Percorso B (Fasi 9–12) porta lì: port a PyTorch *dimostrato equivalente* a questo
motore, tokenizer BPE scritto a mano, corpus grande, training su GPU. Ma è un altro
viaggio. Per ora, abbiamo fatto la cosa più importante: **capito**.

---

*Fine del capitolo Fase 8 — e del Percorso A.*

---

<a name="fase-9"></a>
# Fase 9 — Il port a PyTorch, con equivalenza dimostrata

Apriamo il Percorso B. Riscriviamo RonkLM in PyTorch e — questo è il punto — **proviamo
con dei numeri che i due motori dicono esattamente la stessa cosa**. Non è la fase in
cui abbandoniamo il lavoro fatto: è la fase in cui il lavoro fatto diventa lo
**strumento di verifica** di tutto ciò che verrà.

📁 File: [`ronklm_torch/model.py`](ronklm_torch/model.py),
[`tests/test_equivalence.py`](tests/test_equivalence.py),
[`scripts/benchmark.py`](scripts/benchmark.py)

---

<a name="sec-9-0"></a>
## 9.0 Perché adesso, e perché non prima

Per tutto il Percorso A abbiamo *vietato* PyTorch. Ora lo introduciamo. Non è un
cambio di idea: è il piano che si compie.

> **📖 Il ragionamento.** Se avessimo iniziato con PyTorch, `loss.backward()` sarebbe
> stato magia, `nn.MultiheadAttention` una scatola nera, `AdamW` una sigla. Avendo
> invece scritto a mano il motore di autograd, l'attention e l'ottimizzatore, adesso
> PyTorch non è più un framework misterioso: è **la versione veloce di cose che
> possediamo**. Ogni sua API corrisponde a codice che abbiamo derivato e testato noi.

E c'è un secondo motivo, pratico: **serve la GPU**. Il nostro motore NumPy gira su CPU,
e (come calcolato in [I.2](#sec-i-2) del piano) addestrare ~150M parametri lì
richiederebbe *anni*. PyTorch è il ponte verso la GPU, e quindi verso un modello che
scrive italiano vero.

---

<a name="sec-9-1"></a>
## 9.1 La tabella di corrispondenza: PyTorch non ha concetti nuovi

Il vero contenuto didattico della fase è questa tabella, che sta anche in cima a
[`ronklm_torch/__init__.py`](ronklm_torch/__init__.py):

| Nostro (Percorso A) | PyTorch | Costruito in |
|---|---|---|
| `autograd.Tensor` con `.grad` | `torch.Tensor(requires_grad=True)` | Fase 3 |
| `Tensor.backward()` (topo-sort) | `loss.backward()` | Fase 3 |
| `nn.Module` + `parameters()` | `torch.nn.Module` | Fase 4 |
| `nn.Linear` (`x @ W + b`) | `torch.nn.Linear` (`x @ W.T + b`) | Fase 4 |
| `nn.Embedding` (`gather_rows`) | `torch.nn.Embedding` | Fase 4 |
| `nn.LayerNorm` (mean/var a mano) | `torch.nn.LayerNorm` | Fase 6 |
| `Tensor.gelu()` (composita) | `F.gelu(approximate='tanh')` | Fase 3/6 |
| `autograd.cross_entropy` (fusa) | `F.cross_entropy` | Fase 3 |
| `optim.AdamW` (scritto a mano) | `torch.optim.AdamW` | Fase 4 |

> **Leggi la tabella da destra a sinistra**: per *ogni* pezzo di PyTorch che useremo,
> esiste una riga del nostro codice che sappiamo spiegare. Non c'è **nessun concetto
> nuovo** — solo ingegneria migliore: kernel scritti in C++/CUDA, operazioni fuse,
> gestione della memoria. La differenza tra `ronkgrad` e PyTorch è *implementativa*,
> non concettuale.

Il codice del port è quasi noioso da leggere, ed è un ottimo segno: `Head`,
`MultiHeadAttention`, `FeedForward`, `Block`, `GPT` hanno la stessa forma di prima,
con le stesse costanti (`-1e9` per la maschera, `eps=1e-5` per LayerNorm, gelu in
approssimazione tanh). Ogni dettaglio deve combaciare, altrimenti il test di
equivalenza — che vedremo tra poco — se ne accorge.

---

<a name="sec-9-2"></a>
## 9.2 La trappola della trasposizione

C'è **una** differenza reale tra i due mondi, e vale la pena isolarla perché è il tipo
di bug che rovina i port:

```
il NOSTRO Linear:      W ha forma (n_in, n_out),   calcola  x @ W + b
il Linear di PyTorch:  weight ha forma (n_out, n_in), calcola x @ weight.T + b
```

Le due convenzioni sono equivalenti, ma **trasposte**. Quindi copiare i pesi da un
modello all'altro richiede una trasposizione (`W.T`), e lo stesso vale quando
confrontiamo i *gradienti*.

> **⚠️ Perché è pericolosa.** Se sbagliassimo la trasposizione, il modello non
> crasherebbe: le shape combacerebbero comunque (in una rete dove molte matrici sono
> quadrate) o l'errore emergerebbe solo in certi punti. Otterremmo semplicemente un
> modello che *impara peggio*, e passeremmo giorni a incolpare gli iperparametri. È
> l'archetipo del bug che "non rompe niente di visibile" — la stessa famiglia del
> gradiente sbagliato di [Fase 2.7](#sec-2-7). L'unica difesa è verificare
> numericamente. Che è esattamente ciò che facciamo.

---

<a name="sec-9-3"></a>
## 9.3 La prova di equivalenza: i numeri

Il metodo: **stessi pesi** (copiati dal modello NumPy a quello torch), **stesso
input**, e si confrontano tre cose. Per un confronto severo mettiamo anche il modello
torch in `float64` (il nostro motore lavora in doppia precisione), così le differenze
residue sono solo arrotondamenti nell'ultimo bit.

Ecco i risultati reali:

```
FORWARD    max |logit_numpy − logit_torch|        = 1.8e-15
LOSS       |loss_numpy − loss_torch|              = 4.4e-16
BACKWARD   max |grad_numpy − grad_torch|          = 8.7e-17     (su tutti i 38 parametri)

TRAINING (5 passi di AdamW, stessi batch):
  step |        loss NumPy |        loss torch |  scarto
    0  | 3.69110184804839 | 3.69110184804839 | 0.00e+00
    1  | 3.64976273348430 | 3.64976273348430 | 0.00e+00
    2  | 3.39537279493138 | 3.39537279493138 | 0.00e+00
    3  | 3.65451616982575 | 3.65451616982575 | 4.44e-16
    4  | 3.55488599279950 | 3.55488599279950 | 0.00e+00
```

Fermiamoci a capire **cosa significa**. `1e-15` non è "molto simile": è *l'ultimo bit*
di un numero in doppia precisione — il limite di ciò che un computer può
rappresentare. E le loss durante il training coincidono fino alla **14ª cifra
decimale**, spesso in modo *esatto*.

> **Questo dimostra che tutto il Percorso A era giusto.** Il motore di autograd con il
> suo ordinamento topologico, l'`_unbroadcast`, la self-attention con la maschera e lo
> scaling, la LayerNorm composita, l'AdamW con la bias-correction e il weight decay
> disaccoppiato: ogni singolo pezzo, scritto a mano da zero, produce **gli stessi
> identici numeri** di una libreria sviluppata da centinaia di ingegneri. Non
> "qualcosa di simile": gli stessi numeri.

C'è anche un test-guardia (`test_all_parameters_are_covered`) che verifica che i due
modelli abbiano lo *stesso numero di parametri*: se un domani aggiungessimo un layer da
una parte e dimenticassimo l'altra, il confronto non "passerebbe per omissione".

---

<a name="sec-9-4"></a>
## 9.4 Il benchmark: quanto costa capire

Ora la domanda pratica: **quanto ci è costata, in velocità, la scelta didattica di
scrivere tutto a mano?** Misuriamo i token al secondo di training (forward + backward +
update) sulla configurazione di RonkLM v1:

```
NumPy-CPU (ronkgrad)        9.819 tok/s    1,0x
torch-CPU                  55.722 tok/s    5,7x
```

**5,7× solo cambiando motore, sulla stessa CPU.** Da dove viene il guadagno? Non da un
algoritmo migliore — abbiamo appena dimostrato che i calcoli sono identici — ma da:

- **niente grafo di oggetti Python**: il nostro `Tensor` crea un oggetto Python per
  ogni operazione intermedia, con la relativa closure `_backward`. PyTorch tiene il
  grafo in C++.
- **kernel ottimizzati e fusi**: operazioni scritte in C++ con vettorizzazione SIMD,
  e più operazioni fuse in un solo passaggio sulla memoria.
- **meno allocazioni**: riuso dei buffer invece di creare nuovi array a ogni passo.

> **🔧 La lezione.** Il costo della chiarezza è ~6× su CPU — un prezzo che valeva
> assolutamente la pena pagare per capire, e che ora non paghiamo più. E il salto vero
> non è questo: è la GPU.

### Il benchmark alla scala vera, e una lezione inattesa

Sulla config di RonkLM v1 (160k parametri) la GPU è risultata **più lenta** della CPU:

```
NumPy-CPU     10.125 tok/s        torch-CUDA fp32   44.400 tok/s
torch-CPU     49.003 tok/s        torch-CUDA bf16   38.518 tok/s
```

> **📖 Perché la GPU perde sui modelli piccoli.** Con 512 token per passo e matrici
> minuscole, ogni *kernel* (l'operazione lanciata sulla GPU) fa pochissimo lavoro, e
> domina l'**overhead di lancio**: la CPU spende più tempo a *dire alla GPU cosa fare*
> di quanto la GPU ne spenda a farlo. Le migliaia di core restano in gran parte fermi.
> Lezione generale: **una GPU va misurata al carico per cui è fatta**, altrimenti si
> trae la conclusione sbagliata.

Rifacendo la misura alla scala target (134M parametri, contesto 512):

```
torch-CPU            835 tok/s     1,0x
torch-CUDA fp32   21.289 tok/s    25,5x
torch-CUDA bf16   26.016 tok/s    31,2x     ← ora bf16 è il più veloce, come atteso
```

**31×**. E si nota che `bf16`, inutile sul modello piccolo, diventa il migliore quando
c'è abbastanza lavoro da dare ai tensor core.

### Il collo di bottiglia eravamo noi (e come l'abbiamo tolto)

Un dato però stonava: **8,7 GB di VRAM per un batch di appena 8**, e a batch 16 si
sfiorava il limite dei 16 GB. Il colpevole era il nostro port, fedele ma ingenuo:

- `MultiHeadAttention` fa un **ciclo Python su 12 teste separate**, ognuna con le sue 3
  piccole moltiplicazioni: decine di kernel minuscoli invece di pochi grandi;
- ogni testa **materializza la matrice di attenzione 512×512** e la tiene in memoria
  per il backward. Con 12 teste × 12 layer, è lì che finiva la VRAM.

La soluzione è quella dei GPT veri, e sta in `CausalSelfAttention`:

1. **QKV fuso**: una sola `Linear(n_embd, 3·n_embd)` invece di 36 piccole.
2. **Teste come dimensione di batch**: un `reshape` a `(B, n_head, T, head_size)` e
   tutte le teste si calcolano insieme, senza cicli Python.
3. **FlashAttention** (`F.scaled_dot_product_attention`): non materializza *mai* la
   matrice T×T — la calcola a blocchi nella memoria veloce (SRAM) della GPU.

Il risultato, sempre a 134M parametri e contesto 512, in bf16:

```
 batch |   didattica (ciclo teste)   |   ottimizzata (QKV fuso + Flash)
-------|-----------------------------|----------------------------------
    8  |  30.633 tok/s    8,7 GB     |   62.796 tok/s    4,5 GB
   16  |   9.918 tok/s   15,6 GB     |   74.652 tok/s    7,3 GB
   32  |   1.904 tok/s   29,4 GB(*)  |   78.314 tok/s   12,7 GB  ← ottimo
   48  |   1.264 tok/s   43,2 GB(*)  |   29.441 tok/s   18,2 GB(*)
(*) oltre i 16 GB: la memoria trabocca nella RAM di sistema e le prestazioni crollano
```

**2,6× più veloce e con meno della metà della VRAM.** Il punto ottimale è batch 32:
78.314 token/s in 12,7 GB.

> **🔧 E qui si vede a cosa serve davvero il test di equivalenza.** Abbiamo appena
> riscritto il pezzo più delicato del modello in una forma completamente diversa — pesi
> fusi in una matrice sola, teste ricomposte con dei reshape, un kernel di attention di
> cui non vediamo il codice. Un cambiamento così, senza rete, è il modo classico di
> introdurre un bug silenzioso. Invece abbiamo semplicemente **riverificato contro il
> riferimento NumPy**: forward e gradienti coincidono ancora. *Possiamo ottimizzare in
> modo aggressivo perché abbiamo qualcosa che ci dice se abbiamo rotto la matematica.*
> Questo è il dividendo del Percorso A, e si incassa proprio adesso.

### Cosa significa per la Fase 12

```
78.314 tok/s × 3600 s × 8 ore  ≈  2,26 miliardi di token
```

Con ~1,4 miliardi di token unici di italiano (Fase 11) sono circa **1,6 epoche** in una
notte di training — vicino al regime Chinchilla per un modello da 150M. Il piano
regge: **il 150M è addestrabile sulla 4080 Super nel budget previsto.**

---

<a name="sec-9-5"></a>
## 9.5 Anatomia della memoria: dove finiscono 43 GB

Durante i benchmark è successa una cosa che vale un capitolo a sé: il Task Manager
mostrava la **memoria di sistema all'88%** (56 su 63,6 GB), la memoria GPU dedicata
esaurita e quella "condivisa" in uso. Su una macchina con 16 GB di VRAM che sta
addestrando un modello da 134M. Capire *perché* spiega tre cose insieme: come funziona
la memoria di una GPU, perché le prestazioni sono crollate, e qual è il tetto operativo
per la Fase 12.

### 9.5.1 Memoria "dedicata" e "condivisa": il meccanismo di Windows

> **📖 Concetto.** Windows presenta due voci di memoria per la GPU:
> - **dedicata** = la VRAM vera, fisicamente sulla scheda (16 GB sulla 4080 Super);
> - **condivisa** = una quota della RAM di sistema (di solito la metà: 31,8 GB su
>   63,6) che il sistema è disposto a *prestare* alla GPU.
>
> Quando un'allocazione CUDA non entra nella VRAM, il driver Windows (WDDM) **non
> restituisce un errore**: sposta silenziosamente dei blocchi nella memoria condivisa,
> cioè nella RAM di sistema, raggiungibile solo attraverso il bus PCIe.

Ed ecco la spiegazione dell'88%: la RAM di sistema era occupata (a) dalle applicazioni
normali della macchina — nel nostro caso ~23 GB tra browser, editor e vari — e (b) dai
blocchi di memoria GPU traboccati. Somma: ~56 GB.

> **⚠️ Perché questo comportamento è insidioso.** Su Linux, un'allocazione che non entra
> in VRAM di solito produce un **errore netto** (`CUDA out of memory`): brutto ma
> onesto, capisci subito. Su Windows ottieni invece un **degrado silenzioso**: il
> programma continua a funzionare, i risultati sono corretti, ma la velocità crolla di
> uno o due ordini di grandezza. Se non guardassi i token/s, penseresti che "la GPU è
> lenta" e cercheresti la causa nel posto sbagliato.

Ed è esattamente ciò che avevamo misurato:

```
variante didattica, batch 32:   29,4 GB richiesti  ->   1.904 tok/s   (traboccando)
variante didattica, batch  8:    8,7 GB richiesti  ->  30.633 tok/s   (tutto in VRAM)
```

Lo stesso identico codice è **16× più lento** solo perché la memoria non ci sta.
Il picco di RAM e il crollo di velocità **non sono due problemi: sono lo stesso
problema**.

> **Nota**: anche la variante *ottimizzata* traboccava a batch 48 (18,2 GB > 16 GB), e
> infatti lì scendeva da 78.314 a 29.441 tok/s. Il fenomeno non dipende
> dall'implementazione: dipende dallo sforare la VRAM.

### 9.5.2 Dove va la memoria, voce per voce

Contiamo. Configurazione: 134M parametri, 12 layer, 12 teste, `n_embd` 768,
contesto T=512, vocabolario 32k, calcolo in bf16 (2 byte per numero).

**Parte fissa** — non dipende dal batch:

| Voce | Conto | Memoria |
|---|---|---|
| Pesi (copia master in fp32) | 134M × 4 byte | 537 MB |
| Gradienti | 134M × 4 byte | 537 MB |
| Stati di AdamW (`m` e `v`) | 2 × 134M × 4 byte | 1,07 GB |
| | **subtotale** | **~2,1 GB** |

Questi 2,1 GB ci sono sempre, anche con batch 1. È il "costo di esistere" del modello.

**Parte variabile** — le *attivazioni* salvate durante il forward per poter fare il
backward. Queste crescono col batch, ed è qui che si consuma tutto:

| Voce | Conto (batch 32) | Memoria |
|---|---|---|
| **Matrici di attenzione** (versione ingenua) | 32 × 12 teste × 512² × 2 byte × ~2 tensori × 12 layer | **~4,8 GB** |
| Attivazioni della FFN (espansione 4×) | 32 × 512 × 3072 × 2 byte × ~2 × 12 layer | ~2,4 GB |
| Logit finali + intermedi della cross-entropy | 32 × 512 × 32.000 × 2 byte × ~2 | ~2,1 GB |
| Flusso residuo e tensori vari per layer | 32 × 512 × 768 × 2 byte × (parecchi) × 12 | ~1–2 GB |

A batch 48 tutte queste voci crescono di 1,5× e il totale arriva ai **43 GB** osservati.

### 9.5.3 Il termine quadratico è il protagonista

Guarda la prima riga della tabella: la memoria delle matrici di attenzione è

```
batch × n_teste × T² × n_layer
```

C'è un **T al quadrato**. È la conseguenza diretta di ciò che l'attention *fa*: per
ogni posizione calcola un punteggio verso *ogni altra* posizione — una matrice T×T
(la stessa che avevamo stampato come heatmap in [Fase 5.5](#sec-5-5), lì 15×15, qui
512×512 per ognuna delle 144 combinazioni testa/layer).

Le conseguenze pratiche sono brutali:

| Contesto T | Memoria attention (relativa) |
|---|---|
| 256 | 1× |
| 512 | **4×** |
| 1024 | **16×** |
| 2048 | **64×** |

> **📖 Ecco perché il "contesto lungo" è il lusso costoso dei LLM.** Raddoppiare la
> finestra di contesto non raddoppia il costo: lo **quadruplica**. È il motivo per cui i
> modelli commerciali fanno pagare di più i contesti lunghi, per cui esistono decine di
> ricerche per rendere l'attention sub-quadratica, e per cui la nostra scelta di T=512
> per la Fase 12 non è timidezza ma aritmetica.

### 9.5.4 Come FlashAttention fa sparire il termine T²

`F.scaled_dot_product_attention` risolve il problema con due idee:

1. **Tiling**: non calcola la matrice T×T tutta insieme. La elabora a **blocchi**
   dentro la SRAM della GPU (la memoria piccolissima e velocissima vicino ai core),
   accumulando il risultato. La matrice completa non esiste mai in memoria.
2. **Ricalcolo nel backward**: invece di *salvare* la matrice di attenzione per il
   passaggio all'indietro, la **ricalcola** al volo quando serve. È un baratto:
   si spende un po' più di calcolo per non spendere memoria.

Il risultato, misurato da noi:

```
batch 32, versione ingenua:      29,4 GB   ->   1.904 tok/s
batch 32, con FlashAttention:    12,7 GB   ->  78.314 tok/s
```

Il termine quadratico sparisce dal conto della memoria, tutto resta dentro i 16 GB, e
la velocità è quella vera della scheda.

### 9.5.5 Cosa significa per la Fase 12

Da questa analisi escono tre numeri operativi che useremo per il training del 150M:

- **Batch 32 (12,7 GB) è il tetto comodo** sulla 4080 Super: lascia ~3 GB di margine
  per la frammentazione e per il resto del sistema.
- **Batch 48 (18,2 GB) va evitato**: trabocca, e le prestazioni si dimezzano e oltre.
- **T=512 è la scelta giusta** per questo budget; se un domani volessimo T=1024,
  a parità di VRAM dovremmo circa dimezzare il batch.

E se servisse un batch *effettivo* più grande (a questa scala si usano ~0,5M token per
update)? La risposta è la **gradient accumulation**, già prevista nel piano
(Fase 12.2): si sommano i gradienti di più micro-batch prima di aggiornare i pesi.
Matematicamente identico a un batch grande — perché il backward della somma distribuisce
il gradiente, la stessa proprietà vista in [Fase 3.3](#sec-3-3) — ma con memoria
costante. È così che si addestrano modelli enormi su schede piccole.

---

<a name="sec-9-6"></a>
## 9.6 Mixed precision e GPU: cosa cambia davvero

Due concetti che entrano ora nel progetto e che governeranno il training della Fase 12.

> **📖 Perché la GPU è così più veloce.** Una CPU ha pochi core molto "intelligenti"
> (grandi cache, predizione dei salti, esecuzione fuori ordine): è ottimizzata per fare
> *una cosa complicata alla volta, in fretta*. Una GPU ha migliaia di core semplici:
> è ottimizzata per fare *la stessa operazione semplice su migliaia di dati insieme*.
> Le reti neurali sono esattamente questo: moltiplicazioni di matrici, cioè milioni di
> moltiplicazioni-e-somme tutte indipendenti. È il carico di lavoro perfetto per una
> GPU — e il motivo per cui il deep learning moderno è nato quando qualcuno ha pensato
> di usare le schede video per farci matematica.

> **📖 Mixed precision (bf16).** I numeri in virgola mobile si possono rappresentare
> con 32 bit (`fp32`) o 16 (`bf16`/`fp16`). Le GPU moderne hanno unità dedicate (i
> *tensor core*) che macinano i formati a 16 bit a velocità multiple rispetto ai 32
> bit, e usano metà memoria e metà banda. La tecnica "mixed precision" fa i calcoli
> pesanti in 16 bit ma tiene una copia *master* dei pesi in 32 bit, perché gli
> aggiornamenti dell'ottimizzatore sono piccoli e in 16 bit si perderebbero
> nell'arrotondamento. `bf16` (brain float) in particolare sacrifica precisione per
> mantenere lo stesso *range* di `fp32`, il che lo rende molto più robusto agli
> overflow rispetto a `fp16`. Lo useremo nella Fase 12.

---

<a name="sec-9-7"></a>
## 9.7 Glossario Fase 9 / cosa arriva in Fase 10

Nuovi termini:

- **Port**: riscrivere lo stesso programma su un'altra tecnologia mantenendone il
  comportamento.
- **Test di equivalenza**: confronto numerico tra due implementazioni che devono
  produrre gli stessi risultati.
- **Doppia precisione (`float64`)**: numeri a 64 bit; usati nel test per rendere le
  tolleranze severe.
- **Mixed precision / `bf16`**: calcolare in 16 bit tenendo i pesi master in 32.
- **Tensor core**: unità della GPU dedicate alle moltiplicazioni di matrici a bassa
  precisione.
- **`register_buffer`**: in PyTorch, un tensore che appartiene al modulo ma **non** è un
  parametro addestrabile (noi lo usiamo per la maschera causale).
- **Memoria GPU dedicata / condivisa**: la VRAM sulla scheda / la quota di RAM di
  sistema che Windows presta alla GPU quando la VRAM finisce.
- **Spilling**: il traboccare delle allocazioni dalla VRAM alla RAM di sistema; non dà
  errore ma fa crollare le prestazioni (accessi via PCIe).
- **Attivazioni**: i tensori intermedi del forward che vanno *salvati* per poter
  calcolare il backward; sono la voce di memoria che cresce col batch.
- **Tiling**: elaborare una matrice a blocchi nella memoria veloce (SRAM) invece che
  tutta insieme; è il trucco di FlashAttention.
- **Gradient accumulation**: sommare i gradienti di più micro-batch prima di
  aggiornare, per ottenere un batch effettivo grande a memoria costante.

**In Fase 10** costruiremo il **tokenizer BPE, scritto a mano** — il pezzo che avevamo
volutamente rimandato nel Percorso A ([sezione 0.3](#sec-0-3)). Adesso serve davvero: a
livello di carattere il modello spreca capacità a compitare e il contesto rende poco.
Il BPE impara dai dati quali sequenze frequenti meritano un simbolo dedicato, e
trasformerà le "quasi-parole" in parole vere.

---

*Fine del capitolo Fase 9.*

---

<a name="fase-10"></a>
# Fase 10 — Il tokenizer BPE, scritto a mano

Nel Percorso A avevamo deliberatamente evitato il BPE ([sezione 0.3](#sec-0-3)): sarebbe
stato un ostacolo tra noi e i concetti che contavano. Adesso serve davvero, e lo
costruiamo da zero come tutto il resto — è lo stesso algoritmo dei tokenizer dei GPT
reali.

📁 File: [`ronklm/bpe.py`](ronklm/bpe.py)

---

<a name="sec-10-0"></a>
## 10.0 Perché il char-level non basta più

Tre costi che a 50–150M parametri diventano proibitivi:

1. **Contesto sprecato.** 512 posizioni char-level ≈ 80 parole italiane: troppo poche
   per la coerenza di un paragrafo. Con il BPE (~3,8 caratteri per token in italiano)
   le *stesse* 512 posizioni valgono ~300 parole. E siccome l'attention costa **T²**
   ([sezione 9.5](#sec-9-5)), comprare contesto allungando T è carissimo, mentre
   *densificare i token* è quasi gratis.
2. **Capacità sprecata.** Un modello char-level spende una fetta dei suoi strati solo
   per *compitare* ("m-a-n-g-i-a-r-e è una parola") prima ancora di poter modellare la
   sintassi. Il BPE gli consegna le parole frequenti già intere.
3. **Segnale più povero.** Predire la prossima *lettera* è spesso ovvio; predire il
   prossimo *token* è un compito semanticamente più ricco, quindi ogni passo di
   training insegna di più.

---

<a name="sec-10-1"></a>
## 10.1 L'algoritmo BPE in una pagina

*Byte Pair Encoding* è compressione guidata dai dati, e l'idea sta in una frase:

> Parti dai byte. Trova la **coppia adiacente più frequente** nel corpus. Fondila in un
> **nuovo simbolo**. Ripeti finché il vocabolario non raggiunge la taglia voluta.

Un esempio giocattolo. Corpus: `"basso basso passo"`.

```
inizio:      b a s s o _ b a s s o _ p a s s o
coppia piu' frequente: "s"+"s" (3 volte)  ->  nuovo simbolo X=ss
dopo:        b a X o _ b a X o _ p a X o
coppia piu' frequente: "a"+"X" (3 volte)  ->  nuovo simbolo Y=aX
dopo:        b Y o _ b Y o _ p Y o
...e cosi' via
```

Le sequenze frequenti si guadagnano un simbolo dedicato; quelle rare restano scomposte.
Il risultato è un vocabolario **adattivo**: `"che"`, `"zione"`, `"Michele"` diventano
token singoli, mentre una parola rarissima resta spezzata — ma *sempre
rappresentabile*.

Per codificare un testo nuovo, si riapplicano le fusioni **nell'ordine in cui sono state
imparate** (le prime imparate sono le più frequenti, quindi vanno applicate per prime).

---

<a name="sec-10-2"></a>
## 10.2 Perché partire dai byte

Il nostro BPE parte dai **256 byte**, non dai caratteri. È una scelta con una
conseguenza importante:

> **📖 Nessun testo è mai "fuori vocabolario".** Qualunque cosa — italiano, cinese,
> emoji, dati binari — è una sequenza di byte. Nel caso peggiore il tokenizer la
> scompone in byte singoli, che sono sempre nei primi 256 token. È la proprietà che il
> word-level non poteva dare ([sezione 0.3](#sec-0-3)): niente parole sconosciute, mai.

Lo verifichiamo con un test esplicito: il tokenizer è addestrato solo su *Pinocchio*
(che non contiene né emoji né cirillico né giapponese), eppure:

```python
tok.decode(tok.encode("🍕🚀"))        == "🍕🚀"        ✓
tok.decode(tok.encode("Привет мир"))  == "Привет мир"  ✓
tok.decode(tok.encode("日本語のテキスト")) == "日本語のテキスト" ✓
```

> **🔧 Un effetto collaterale da capire.** Siccome si lavora sui byte, alcuni token
> intermedi corrispondono a *pezzi* di un carattere multi-byte (la `à` in UTF-8 sono
> due byte). Se li stampi da soli vedi il carattere di sostituzione `�`. Non è un bug:
> è un token che ha senso solo insieme al successivo. Il testo decodificato completo è
> sempre corretto.

---

<a name="sec-10-3"></a>
## 10.3 La pre-tokenizzazione: non fondere attraverso le parole

Prima di applicare il BPE, spezziamo il testo in "parole" con un'espressione regolare
(la stessa idea di GPT-2). Senza questo passaggio, l'algoritmo fonderebbe volentieri
attraverso gli spazi e creerebbe token come `"della_casa"`: spreco di vocabolario e
pessima generalizzazione.

Un dettaglio elegante della convenzione: **lo spazio resta attaccato alla parola che
segue** (`" casa"` invece di `" "` + `"casa"`). Così il modello distingue `"casa"` a
inizio riga da `" casa"` dentro una frase, senza sprecare un token per lo spazio
isolato.

> **🔧 Un test che sbagliava (e cosa ha insegnato).** Avevo scritto un test che
> pretendeva che *nessun* token contenesse uno spazio oltre la prima posizione. È
> fallito su un token `"  "` (due spazi). Ma il test aveva torto, non il codice: le
> **sequenze di spazi** sono un'unità a sé e devono poter diventare un token unico —
> servono a comprimere indentazione e righe vuote, e lo fa anche GPT-2. L'invariante
> giusta è: *o il token è tutto spazi, oppure ha al massimo uno spazio iniziale*. È un
> buon promemoria che un test rosso non significa sempre "codice rotto".

---

<a name="sec-10-4"></a>
## 10.4 Il problema di velocità e come l'abbiamo risolto

La prima versione era corretta ma inutilizzabile su Wikipedia. Il motivo:

> **📖 Il costo della versione ingenua.** Dopo *ogni* fusione, ricontava **tutte** le
> coppie di **tutte** le parole. Con 16.000 fusioni e ~260.000 parole uniche sono
> decine di miliardi di operazioni: ore di calcolo.

La cura è un'indicizzazione incrementale, e vale la pena capirla perché è un pattern
generale:

1. **Non scorrere il corpus, scorri le parole uniche.** Wikipedia ha miliardi di token
   ma solo qualche centinaio di migliaia di parole *distinte*. Si contano le coppie una
   volta sola, pesate per la frequenza di ciascuna parola.
2. **Tieni un indice `coppia → quali parole la contengono`.** Quando fondi una coppia,
   tocchi **solo** quelle parole, non tutte.
3. **Aggiorna i conteggi in differenza**, togliendo le vecchie coppie e aggiungendo le
   nuove, invece di ricontare da zero.
4. **Usa un max-heap con cancellazione pigra** per pescare la coppia più frequente
   senza riscandire il dizionario: le voci obsolete si scartano al momento del prelievo
   confrontandole col conteggio corrente.

Il risultato misurato: l'addestramento di un vocabolario da **16.384 token su 30 MB di
Wikipedia è passato da "ore" a 12 secondi**. Stesso identico risultato, solo senza
sprecare lavoro.

---

<a name="sec-10-5"></a>
## 10.5 Cosa ha imparato, guardato da vicino

Come per le heatmap di attention ([sezione 5.5](#sec-5-5)), ispezionare ciò che
l'algoritmo ha appreso è la verifica più convincente. Ecco alcune fusioni, in ordine di
apprendimento, dal BPE addestrato su Wikipedia italiana:

```
merge     0:  ' d'          (567.788 volte)
merge   500:  'uro'
merge  1000:  'ide'
merge  1500:  ' acqu'
merge  2000:  ' membro'
merge  3000:  ' attore'
merge  4000:  ' Michele'
merge  4500:  ' giocò'
merge  7000:  ' svilupp'
merge  8500:  ' classificata'
merge 12500:  ' possedeva'
merge 16127:  ' necessarie'
```

Si legge la struttura dell'italiano emergere da sola: prima i digrammi frequentissimi
(`' d'`), poi suffissi e morfemi (`'uro'`, `'ide'`), poi radici (`' acqu'`,
`' svilupp'`), infine parole intere sempre più specifiche (`' possedeva'`,
`' necessarie'`) e nomi propri comuni nell'enciclopedia (`' Michele'`, `' Buenos'`,
`' Innsbruck'`). **Nessuno ha scritto una regola grammaticale: è tutto statistica sui
dati.**

Sul nostro corpus, la compressione risultante è di circa **3,8 caratteri per token** —
cioè le stesse 512 posizioni di contesto ora valgono quasi quattro volte più testo che
nel Percorso A.

---

<a name="fase-11"></a>
# Fase 11 — Il corpus grande: da Wikipedia ai token

📁 File: [`data/corpus_b/`](data/corpus_b/) — `download_wikipedia.py`,
`extract_wikipedia.py`, `tokenize_corpus.py`

<a name="sec-11-0"></a>
## 11.0 La pipeline in quattro passi

```
1. DOWNLOAD    dump ZIM di Wikipedia IT da Kiwix           8,29 GB
2. ESTRAZIONE  ZIM -> testo pulito (HTML via, filtri)      4,14 GB, 1.099.087 articoli
3. BPE         addestramento su un campione                vocab 16.384
4. TOKENIZZA   testo -> uint16 binario + split train/val   ~1,1 mld token
```

Due decisioni pratiche che hanno fatto la differenza:

> **🔧 Il mirror.** Il server ufficiale di Kiwix dava **3,8 MB/s**; il mirror
> `mirrors.dotsrc.org` ne dava **47**. Su 8,3 GB è la differenza tra 2 ore e 3 minuti.
> Vale sempre la pena misurare invece di accettare la prima fonte.

> **🔧 Il disco.** Il corpus sta su un **SSD NVMe**, non sull'HDD dove vive il
> progetto. Il file dei token verrà letto ad *accesso casuale* a ogni batch: da un
> disco meccanico il caricamento dati diventerebbe il collo di bottiglia e la GPU
> resterebbe ferma ad aspettare. Regola: in un training ben fatto il collo di bottiglia
> dev'essere la GPU, mai il disco.

---

<a name="sec-11-1"></a>
## 11.1 Guardare i dati: tre bug trovati a occhio

La disciplina imparata in [Fase 0.4](#sec-0-4) — *stampare il testo estratto e
leggerlo* — ha ripagato immediatamente. La prima versione dell'estrattore produceva
questo:

```
!!

!!

!!

A questo titolo corrispondono più voci, di seguito elencate.
Questa è una pagina di disambiguazione; se sei giunto qui cliccando un collegamento...
...
Questa voce è stata pubblicata da Wikipedia. Il testo è rilasciato in base alla
licenza Creative Commons Attribution-Share Alike 4.0...
```

Tre difetti, tutti gravi e tutti invisibili senza guardare:

1. **Le pagine di disambiguazione passavano tutte.** Il mio filtro cercava
   `"disambigua"` nel *titolo*, ma le disambiguazioni hanno titoli normalissimi
   (`"$5,000 Reward"`): il marcatore sta nel **testo**. Risultato: un corpus pieno di
   elenchi di rimandi invece che di prosa.
2. **Il titolo ripetuto 3–4 volte** all'inizio di ogni documento (l'HTML dello ZIM lo
   contiene nel `<title>`, nell'intestazione e nell'`<h1>`, e io ne aggiungevo un
   quarto). Il modello avrebbe imparato quel tic.
3. **Il boilerplate di licenza in fondo a ogni articolo.** Un milione di articoli × la
   stessa formula legale = il modello l'avrebbe vista più di qualunque altra frase
   italiana, e imparata a memoria.

C'era anche un bug più sottile nel primo tentativo di correzione: il collassatore di
righe ripetute confrontava righe *adiacenti*, ma i titoli duplicati sono separati da
righe vuote — quindi non ne eliminava nemmeno uno. Andava confrontata l'ultima riga
**non vuota**.

---

<a name="sec-11-2"></a>
## 11.2 I filtri di qualità e perché sono severi

Su 3.226.161 voci del dump ne teniamo **1.099.087** (circa un terzo). Cosa buttiamo, e
perché:

| Filtro | Motivo |
|---|---|
| Redirect e voci non-HTML | non sono testo |
| Namespace `Categoria:`, `Template:`, `Portale:`… | metadati, non prosa |
| Articoli < 400 caratteri | stub: poco segnale, molto rumore |
| Pagine di disambiguazione | elenchi di rimandi |
| Pagine-lista (>80% righe corte) | "Elenco di…", cronologie: non prosa |
| Testo con < 60% di lettere | tabelle di numeri, codici |
| Duplicati esatti (hash del contenuto) | vedi sotto |

> **📖 Perché la deduplicazione è il filtro più importante.** I duplicati (a) fanno
> **memorizzare** invece di generalizzare — lo stesso testo visto 50 volte è 50 volte
> più memorizzato; (b) **contaminano la validation**: se lo stesso documento finisce in
> train e in val, la loss di validation *mente*, e siccome è l'unica bussola di cui ci
> fidiamo ([sezione 4.6](#sec-4-6)), un val contaminato rompe l'intero esperimento.

> **⚠️ Un limite dichiarato.** L'estrazione gira su 15 processi paralleli, e la
> deduplicazione è **per processo**: due articoli identici capitati in intervalli
> diversi sopravvivono entrambi. Sui dati reali il tasso di duplicati esatti in
> Wikipedia è bassissimo (0 su 6.000 nei test), quindi il compromesso vale i ~45 minuti
> risparmiati. Lo scriviamo esplicitamente invece di far finta di niente: un limite
> noto è gestibile, uno nascosto no.

---

<a name="sec-11-3"></a>
## 11.3 Binarizzazione e memmap

L'ultimo passo trasforma 4,14 GB di testo in un file binario di token.

> **📖 Perché pre-tokenizzare invece di farlo al volo.** Tokenizzare costa CPU. Farlo a
> ogni epoca significherebbe rifare ogni volta lo stesso identico lavoro, tenendo la
> GPU affamata. Lo facciamo una volta e salviamo il risultato.

> **📖 Perché `uint16`.** Il nostro vocabolario ha 16.384 token, ben sotto i 65.536
> rappresentabili con 2 byte. Quindi 1,1 miliardi di token occupano ~2,2 GB — un file
> che si legge comodamente.

> **📖 Perché `np.memmap`.** Invece di caricare i 2,2 GB in RAM, li lasciamo su disco e
> il sistema operativo porta in memoria **solo le finestre effettivamente lette**. Il
> training parte in un istante e usa memoria costante, qualunque sia la dimensione del
> corpus. È lo schema di nanoGPT, e ora ne capiamo ogni ragione.

Anche la scrittura è in **streaming**: i blocchi tokenizzati vengono scritti man mano
invece di essere accumulati in una lista. Accumulare avrebbe richiesto ~5 GB di RAM di
picco (2,2 per la lista + 2,2 per la concatenazione finale); così ne bastano pochi MB.

Infine lo split train/val, con la stessa logica contigua della [Fase 0.7](#sec-0-7): la
validation è un blocco **in coda**, testo che il modello non vede mai durante
l'addestramento.

---

*Fine dei capitoli Fasi 10 e 11.*
