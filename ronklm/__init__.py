"""RonkLM — un piccolo Language Model costruito da zero per capire come funziona.

Percorso A (Fasi 0-8): tutto in NumPy puro, nessun framework di deep learning.

Sotto-moduli (nascono una fase alla volta):
    tokenizer  -> CharTokenizer: testo <-> interi           (Fase 0)
    dataset    -> split train/val, get_batch                (Fase 0)
    autograd   -> ronkgrad: Tensor con backward             (Fase 3)
    nn         -> Module, Linear, Embedding, LayerNorm...    (Fase 4+)
    optim      -> SGD, AdamW                                 (Fasi 2/4)
    models/    -> bigram, mlp, attention, block, gpt         (Fasi 1-7)
"""

__version__ = "0.1.0"  # segue le fasi del Percorso A, non il branch git
