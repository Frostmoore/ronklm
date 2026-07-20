"""RonkLM-torch — il port PyTorch di RonkLM (Percorso B, Fase 9).

Questo pacchetto e' SEPARATO da `ronklm/` di proposito: i due motori convivono.
`ronklm/` (NumPy puro) resta l'implementazione di RIFERIMENTO -- quella che abbiamo
derivato a mano e verificato coi gradient check -- e diventa il banco di prova di
questo port (vedi tests/test_equivalence.py).

Tabella di corrispondenza (il vero contenuto didattico della Fase 9):

    ronkgrad / ronklm                  PyTorch                        costruito in
    ---------------------------------  -----------------------------  ------------
    autograd.Tensor(requires grad)     torch.Tensor(requires_grad=True)  Fase 3
    Tensor.backward()                  loss.backward()                   Fase 3
    nn.Module + parameters()           torch.nn.Module                   Fase 4
    nn.Linear (x @ W + b)              torch.nn.Linear (x @ W.T + b)     Fase 4
    nn.Embedding (gather_rows)         torch.nn.Embedding                Fase 4
    nn.LayerNorm (mean/var a mano)     torch.nn.LayerNorm                Fase 6
    Tensor.gelu() (composita)          F.gelu(approximate='tanh')        Fase 3/6
    autograd.cross_entropy (fusa)      F.cross_entropy                   Fase 3
    optim.AdamW (scritto a mano)       torch.optim.AdamW                 Fase 4
    models/gpt.GPT                     ronklm_torch.model.GPT            Fase 7

Nessun concetto nuovo: solo ingegneria migliore (kernel C++/CUDA, fusione, memoria).
"""
