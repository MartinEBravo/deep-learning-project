from transformers import AutoTokenizer # type: ignore

from dlkth.config import DEFAULT_MODEL


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])


def get_tokenizer(model_name: str | None):
    tokenizer = AutoTokenizer.from_pretrained(model_name or DEFAULT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def tokenize_text(tokenizer: AutoTokenizer, text: str) -> list[int]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return tokens


def detokenize_text(tokenizer: AutoTokenizer, tokens: list[int]) -> str:
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text
