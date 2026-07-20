"""Test del tokenizer BPE scritto a mano (Fase 10).

Cosa dimostrano:
    - round-trip esatto su testo italiano, punteggiatura, accenti;
    - la proprieta' fondamentale del byte-level: NON fallisce MAI, nemmeno su simboli
      mai visti in addestramento (emoji, alfabeti stranieri) -- al peggio scompone in byte;
    - la compressione migliora al crescere del vocabolario;
    - il training e' deterministico;
    - save/load preservano esattamente la codifica;
    - i merge appresi hanno senso (sono pezzi di italiano, non rumore).
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ronklm.bpe import BPETokenizer  # noqa: E402

CORPUS = os.path.join(os.path.dirname(__file__), "..", "data", "input.txt")


def _corpus() -> str:
    with open(CORPUS, encoding="utf-8", newline="") as f:
        return f.read()


def _small_tokenizer(vocab_size: int = 1000) -> BPETokenizer:
    return BPETokenizer().train(_corpus(), vocab_size=vocab_size)


def test_roundtrip_italian():
    tok = _small_tokenizer()
    for s in [
        "Pinocchio corse verso la bottega di Geppetto.",
        "Perché però città virtù è già —",
        "«Buongiorno!» disse il Grillo... e se ne andò.",
        "\n\nDue righe vuote sopra.\n",
    ]:
        assert tok.decode(tok.encode(s)) == s, f"round-trip fallito su {s!r}"


def test_never_fails_on_unseen_symbols():
    """La proprieta' chiave del byte-level: qualunque cosa e' rappresentabile.
    Il corpus di training (Pinocchio) non contiene emoji ne' cirillico ne' cinese."""
    tok = _small_tokenizer()
    for s in ["🍕🚀", "Привет мир", "日本語のテキスト", "\x00\x01binario"]:
        assert tok.decode(tok.encode(s)) == s, f"round-trip fallito su {s!r}"


def test_empty_string():
    tok = _small_tokenizer()
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_vocab_size_respected():
    tok = BPETokenizer().train(_corpus(), vocab_size=800)
    assert tok.vocab_size <= 800
    assert tok.vocab_size >= 256  # i byte ci sono sempre


def test_compression_improves_with_vocab():
    text = _corpus()[:200_000]
    small = BPETokenizer().train(text, vocab_size=400)
    big = BPETokenizer().train(text, vocab_size=3000)
    sample = text[:20_000]
    ratio_small = len(sample) / len(small.encode(sample))
    ratio_big = len(sample) / len(big.encode(sample))
    assert ratio_big > ratio_small, f"{ratio_big:.2f} non migliore di {ratio_small:.2f}"
    assert ratio_big > 2.5, f"compressione troppo bassa: {ratio_big:.2f} char/token"


def test_training_is_deterministic():
    text = _corpus()[:100_000]
    a = BPETokenizer().train(text, vocab_size=600)
    b = BPETokenizer().train(text, vocab_size=600)
    assert a.merges == b.merges
    assert a.encode("Geppetto e Pinocchio") == b.encode("Geppetto e Pinocchio")


def test_save_load_roundtrip():
    tok = _small_tokenizer()
    s = "Il gatto e la volpe camminavano insieme, però..."
    before = tok.encode(s)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "bpe.pkl")
        tok.save(p)
        tok2 = BPETokenizer.load(p)
    assert tok2.merges == tok.merges
    assert tok2.encode(s) == before
    assert tok2.decode(before) == s


def test_learned_merges_look_italian():
    """Ispezione: i primi merge devono essere pezzi ricorrenti di italiano.
    E' l'equivalente per il tokenizer delle heatmap di attention: una finestra
    diretta su cosa l'algoritmo ha imparato."""
    tok = _small_tokenizer(vocab_size=600)
    first = [tok.vocab[i].decode("utf-8", errors="replace") for i in range(256, 320)]
    # ci aspettiamo digrammi/pezzi italiani molto comuni
    attesi = {"er", "to", "la", "ar", "no", "en", "ta", "ti", "co", "an"}
    trovati = attesi.intersection(set(first))
    assert len(trovati) >= 4, f"merge poco italiani: {first[:20]}"


def test_word_boundary_is_preserved():
    """La pre-tokenizzazione non fonde MAI attraverso un confine di parola: ' casa'
    puo' essere un token, 'la casa' (due parole) no.

    Attenzione all'invariante giusta: le SEQUENZE DI SPAZI sono un'unita' a se' e
    possono legittimamente diventare un token unico ('  ', '\\n\\n') -- serve a
    comprimere indentazione e righe vuote, e lo fa anche il tokenizer di GPT-2.
    Quindi la regola e': o il token e' tutto spazi, oppure ha al massimo uno spazio
    iniziale e nessun altro."""
    tok = _small_tokenizer(vocab_size=2000)
    for tid, b in tok.vocab.items():
        if tid < 256:
            continue
        s = b.decode("utf-8", errors="replace")
        if s.strip() == "":
            continue  # token di soli spazi/a-capo: legittimo
        assert " " not in s[1:], f"il token {s!r} attraversa un confine di parola"


if __name__ == "__main__":
    from _runner import run

    run(globals())
