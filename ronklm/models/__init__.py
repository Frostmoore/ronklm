"""Modelli di RonkLM, uno per fase, dal piu' semplice al GPT completo.

    bigram_count   -> BigramCount   : conteggio puro, niente training   (Fase 1)
    bigram_neural  -> BigramNeural  : 1 layer, gradiente a mano         (Fase 2)
    mlp            -> (Fase 4)
    attention      -> (Fase 5)
    block          -> (Fase 6)
    gpt            -> (Fase 7)
"""
