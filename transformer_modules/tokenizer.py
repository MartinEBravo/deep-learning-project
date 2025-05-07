from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
tokenizer.pad_token = tokenizer.eos_token
encode = lambda s: tokenizer.encode(s, add_special_tokens=True)
decode = lambda l: tokenizer.decode(l, skip_special_tokens=True)
vocab_size = tokenizer.vocab_size