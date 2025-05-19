import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dlkth.config import HiddenInit


class RNNBaseline(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        block_size: int,
        hidden_dim: int,
        n_layers: int = 1,
        hidden_init: HiddenInit = HiddenInit.ZEROS,
        dropout: float = 0.0,
    ) -> None:
        super(RNNBaseline, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=n_embd,
        )
        self.rnn = nn.RNN(
            input_size=n_embd,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(
            in_features=hidden_dim,
            out_features=vocab_size,
        )

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_size = hidden_dim
        self.hidden_init = hidden_init

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if self.hidden_init == HiddenInit.ZEROS:
            return torch.zeros(
                self.n_layers, batch_size, self.hidden_size, device=device
            )

        elif self.hidden_init == HiddenInit.RANDOM:
            return torch.randn(
                self.n_layers, batch_size, self.hidden_size, device=device
            )

        elif self.hidden_init == HiddenInit.XAVIER:
            weight = torch.empty(
                self.n_layers, batch_size, self.hidden_size, device=device
            )
            return nn.init.xavier_uniform_(weight)

        raise NotImplementedError(
            f"Hidden initialization method {self.hidden_init} not implemented."
        )

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor = None,
    ) -> tuple[torch.Tensor, float | None]:
        B, T = idx.shape
        device = idx.device

        x = self.embedding(idx)
        h = self.init_hidden(B, device)

        output, _ = self.rnn(x, h)

        del idx
        del x
        del h
        gc.collect()

        output = self.ln_f(output)
        logits = self.head(output)

        loss = None
        if targets is not None:
            logits = logits.view(B * T, self.vocab_size)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
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
        train_data: list[int],
        val_data: list[int],
        block_size: int,
        batch_size: int,
        learning_rate: float,
        device: torch.device,
        eval_iters: int,
        max_iters: int,
        eval_interval: int,
    ) -> None:
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
