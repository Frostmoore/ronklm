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

*Fine del capitolo Fase 0. Il prossimo capitolo (Fase 1) verrà aggiunto qui sotto,
mantenendo l'indice in cima aggiornato.*
