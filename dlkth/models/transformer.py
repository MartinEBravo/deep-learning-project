import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        _, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        return logits

    def get_loss_by_criterion(
        self,
        y_hat: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        B, T, C = y_hat.shape
        print(f"y_hat: {y_hat.shape}, targets: {targets.shape}")
        y_hat = y_hat.view(B * T, C)
        targets = targets.view(B * T)

        return criterion(y_hat, targets)


    def generate(self, idx, max_new_tokens, block_size, sep_token_id=None):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            #####--------------------------------------########
            # cortar si aparece el token [SEP]- QUITAR ESTA PARTE SI USAMOS TOKENIZER GPT2 AL FINAL
            if sep_token_id is not None and sep_token_id in idx[0]:
                sep_index = (idx[0] == sep_token_id).nonzero(as_tuple=True)[0].item()
                idx = idx[:, : sep_index + 1]
                break  # salimos del bucle si encontramos SEP
        return idx

    def train_model(    
        self,
        train_loader: DataLoader,
        batch_size: int,
        optimizer: Optimizer,
        criterion: nn.Module,
        n_epochs: int,
        device: torch.device,
        show_every_n_batches=100,
        # self,
        # optimizer,
        # train_data,
        # n_epochs,
        # criterion,
        # val_data,
        # get_batch,
        # estimate_loss,
        # max_iters,
        # eval_interval,
        # patience=3,
    ):
        best_val_loss = float("inf")
        best_model = copy.deepcopy(self.state_dict())
        patience_counter = 0

        batch_losses = []

        #for iter in range(max_iters):
        for epoch_i in range(n_epochs):
            for batch_i, (inputs, labels) in enumerate(train_loader, 1):
                # if iter % eval_interval == 0 or iter == max_iters - 1:
                #     losses = estimate_loss(
                #         self, get_batch, train_data, val_data, eval_interval
                #     )
                #     val_loss = losses["val"]
                #     print(
                #         f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                #     )

                #     # early stopping check
                #     if val_loss < best_val_loss:
                #         best_val_loss = val_loss
                #         best_model = copy.deepcopy(self.state_dict())
                #         patience_counter = 0
                #     else:
                #         patience_counter += 1
                #         if patience_counter >= patience:
                #             print(f"Early stopping at step {iter}")
                #             break

                inputs, labels = inputs.to(device), labels.to(device)

                y_hat = self.forward(inputs).to(device)

                loss = self.get_loss_by_criterion(
                    y_hat=y_hat,
                    targets=labels,
                    criterion=criterion,
                )
                batch_losses.append(loss)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if batch_i % show_every_n_batches == 0:
                    print(
                        "Epoch: {:>4}/{:<4}  Loss: {}\n".format(
                            epoch_i, n_epochs, np.average(batch_losses)
                        )
                    )
                    batch_losses = []


        self.load_state_dict(best_model)


# @torch.no_grad()
# def estimate_loss(model, get_batch, train_data, val_data, eval_iters, criterion):
#     out = {}
#     for split in ["train", "val"]:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             X, Y = get_batch(split, train_data, val_data)
#             y_hat = model.forward(X)
#             loss = model.get_loss_by_criterion(y_hat, Y, criterion)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     return out
