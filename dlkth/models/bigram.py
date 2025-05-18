import torch
import torch.nn as nn
from torch.nn import functional as F

from dlkth.data_loader import get_batch


torch.manual_seed(1337)


class Bigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def train_model(
        self,
        train_data,
        val_data,
        block_size,
        batch_size,
        learning_rate,
        device,
        eval_iters,
        max_iters,
        eval_interval,
    ):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        for iter in range(max_iters):
            if iter % eval_interval == 0:
                # Estimate Loss
                out = {}
                self.eval()
                for split in ["train", "val"]:
                    losses = torch.zeros(eval_iters)
                    for k in range(eval_iters):
                        X, Y = get_batch(
                            split, train_data, val_data, block_size, batch_size, device
                        )
                        _, loss = self(X, Y)
                        losses[k] = loss.item()
                    out[split] = losses.mean()
                self.train()
                losses = out
                print(
                    f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )

            # sample a batch of data
            xb, yb = get_batch(
                "train", train_data, val_data, block_size, batch_size, device
            )

            # evaluate the loss
            _, loss = self(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
