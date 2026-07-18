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
