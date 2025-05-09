from transformers import AutoTokenizer
import config

tokenizer = AutoTokenizer.from_pretrained(config.tok_name)
if 'gpt2' in config.tok_name:
  tokenizer.pad_token = tokenizer.eos_token
else:
  tokenizer.pad_token = tokenizer.sep_token
encode = lambda s: tokenizer.encode(s, add_special_tokens=True)
decode = lambda l: tokenizer.decode(l, skip_special_tokens=True)
vocab_size = tokenizer.vocab_size