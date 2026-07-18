"""Test del CharTokenizer (Fase 0.5).

Cosa dimostrano:
    - round-trip: codificare e poi decodificare ridà il testo identico, su TUTTO
      il corpus (la proprieta' fondamentale di un tokenizer senza perdite);
    - determinismo: lo stesso testo produce sempre lo stesso vocabolario ordinato;
    - fail-loud: encode/decode sollevano su input fuori vocabolario;
    - serializzazione: save/load preservano esattamente la mappa.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ronklm.tokenizer import CharTokenizer  # noqa: E402

CORPUS = os.path.join(os.path.dirname(__file__), "..", "data", "input.txt")


def _load_corpus() -> str:
    with open(CORPUS, encoding="utf-8", newline="") as f:
        return f.read()


def test_roundtrip_on_full_corpus():
    text = _load_corpus()
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_short_string():
    tok = CharTokenizer.from_text("abc ,.\n")
    s = "cab\n.a "
    assert tok.decode(tok.encode(s)) == s


def test_vocab_is_deterministic_and_sorted():
    text = "il gatto è sul tetto"
    a = CharTokenizer.from_text(text)
    b = CharTokenizer.from_text(text)
    assert a.chars == b.chars
    assert a.chars == sorted(a.chars)  # ordine stabile => indici stabili


def test_vocab_size_matches_unique_chars():
    text = "aabbbccccd"
    tok = CharTokenizer.from_text(text)
    assert tok.vocab_size == 4  # a b c d


def test_encode_raises_on_unknown_char():
    tok = CharTokenizer.from_text("abc")
    try:
        tok.encode("z")  # 'z' non nel vocab
    except ValueError:
        return
    raise AssertionError("encode doveva sollevare su carattere sconosciuto")


def test_decode_raises_on_out_of_range_index():
    tok = CharTokenizer.from_text("abc")  # indici validi 0..2
    try:
        tok.decode([0, 1, 99])
    except ValueError:
        return
    raise AssertionError("decode doveva sollevare su indice fuori range")


def test_indices_are_contiguous_from_zero():
    tok = CharTokenizer.from_text("xyz abc")
    assert sorted(tok.stoi.values()) == list(range(tok.vocab_size))


def test_save_load_roundtrip(tmp_path_str: str | None = None):
    import tempfile

    text = "Pinocchio, è un burattino! «ohi»... —"
    tok = CharTokenizer.from_text(text)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "vocab.json")
        tok.save(p)
        tok2 = CharTokenizer.load(p)
    assert tok2.chars == tok.chars
    assert tok2.stoi == tok.stoi
    assert tok2.encode(text) == tok.encode(text)


if __name__ == "__main__":
    from _runner import run

    run(globals())
