from transformers import AutoTokenizer # type: ignore

from dlkth.config import DEFAULT_MODEL


def get_tokenizer(model_name: str | None) -> AutoTokenizer:
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
