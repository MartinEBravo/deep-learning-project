import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


torch.manual_seed(1337)


class RNN(nn.Module):
    def __init__(
        self, vocab_size, n_embd, block_size, hidden_dim, n_layers=2, dropout=0.2
    ):
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.lstm = nn.LSTM(
            n_embd, hidden_dim, n_layers, batch_first=True, dropout=dropout
        )
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        x = self.embedding(idx)
        h0 = torch.zeros(self.n_layers, B, self.hidden_dim).to(device)
        c0 = torch.zeros(self.n_layers, B, self.hidden_dim).to(device)
        output, _ = self.lstm(x, (h0, c0))
        output = self.ln_f(output)
        logits = self.head(output)

        loss = None
        if targets is not None:
            logits = logits.view(B * T, self.vocab_size)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
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
        train_losses, val_losses = [], []
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for iter in range(max_iters):
            if iter % eval_interval == 0:
                self.eval()
                out = {}
                for split, data in [("train", train_data), ("val", val_data)]:
                    losses = []
                    for _ in range(eval_iters):
                        i = torch.randint(len(data) - block_size, (batch_size,)).item()
                        x = data[i : i + block_size].unsqueeze(0).to(device)
                        y = data[i + 1 : i + 1 + block_size].unsqueeze(0).to(device)
                        _, loss = self(x, y)
                        losses.append(loss.item())
                    out[split] = np.mean(losses)
                self.train()
                print(
                    f"Step {iter}: train loss {out['train']:.4f}, val loss {out['val']:.4f}"
                )
                train_losses.append(out["train"])
                val_losses.append(out["val"])

            i = torch.randint(len(train_data) - block_size, (1,)).item()
            xb = train_data[i : i + block_size].unsqueeze(0).to(device)
            yb = train_data[i + 1 : i + 1 + block_size].unsqueeze(0).to(device)
            _, loss = self(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return train_losses, val_losses
