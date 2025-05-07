import torch
import config
import data
import model as gpt_model
import train as trainer
import tokenizer as tk

# Data
train_data, val_data = data.load_data(config.input_file, tk.encode)

# Model
m = gpt_model.GPTLanguageModel(
    vocab_size=tk.vocab_size,
    n_embd=config.n_embd,
    n_head=config.n_head,
    n_layer=config.n_layer,
    block_size=config.block_size,
    dropout=config.dropout
).to(config.device)
optimizer = torch.optim.AdamW(m.parameters(), lr=config.learning_rate, weight_decay=1e-2)

# Train
trainer.train_model(m, optimizer, train_data, val_data,
    lambda split, td, vd: data.get_batch(split, td, vd, config.block_size, config.batch_size, config.device),
    lambda: trainer.estimate_loss(m,
        lambda split, td, vd: data.get_batch(split, td, vd, config.block_size, config.batch_size, config.device),
        train_data, val_data, config.eval_iters),
    config.max_iters, config.eval_interval)

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
print(tk.decode(m.generate(context, max_new_tokens=500, sep_token_id=tk.tokenizer.sep_token_id)[0].tolist()))