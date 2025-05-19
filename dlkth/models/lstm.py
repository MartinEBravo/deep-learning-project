import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dlkth.config import HiddenInit


class LSTMBaseline(nn.Module):
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
        super(LSTMBaseline, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=n_embd,
        )
        self.lstm = nn.LSTM(
            input_size=n_embd,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.hidden_init == HiddenInit.ZEROS:
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
            c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)

        elif self.hidden_init == HiddenInit.RANDOM:
            h0 = torch.randn(self.n_layers, batch_size, self.hidden_size, device=device)
            c0 = torch.randn(self.n_layers, batch_size, self.hidden_size, device=device)

        elif self.hidden_init == HiddenInit.XAVIER:
            h0 = nn.init.xavier_uniform_(torch.empty(self.n_layers, batch_size, self.hidden_size, device=device))
            c0 = nn.init.xavier_uniform_(torch.empty(self.n_layers, batch_size, self.hidden_size, device=device))

        else:
            raise NotImplementedError(f"Hidden init {self.hidden_init} not implemented.")

        return (h0, c0)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor = None,
    ) -> tuple[torch.Tensor, float | None]:
        B, T = idx.shape
        device = idx.device

        x = self.embedding(idx)
        (h0, c0) = self.init_hidden(B, device)

        output, _ = self.lstm(x, (h0, c0))

        del idx
        del x
        del h0
        del c0

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

    # def forward_backprop(
    #     self,
    #     x: torch.Tensor,
    #     y: torch.Tensor,
    #     hidden: tuple[torch.Tensor, torch.Tensor],
    #     criterion: nn.Module,
    #     optimizer: Optimizer | None = None,
    # ) -> tuple:
    #     logits, hidden = self.forward(x=x, hidden=hidden)
    #     logits_last = logits[:, -1, :]
    #     loss = criterion(logits_last, y)

    #     if optimizer is not None:
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     return loss.item(), (hidden[0].detach(), hidden[1].detach())

    # def train_model(
    #     self,
    #     train_loader: DataLoader,
    #     val_loader: DataLoader,
    #     optimizer: Optimizer,
    #     criterion: nn.Module,
    #     n_epochs: int,
    #     device: torch.device,
    #     show_every_n_batches: int = 100,
    # ) -> None:
    #     self.to(device)

    #     for epoch in range(n_epochs):
    #         self.train()
    #         epoch_loss = 0.0

    #         for i, (x, y) in enumerate(train_loader):
    #             x, y = x.to(device), y.to(device)

    #             batch_size = x.size(0)

    #             h = self.init_hidden(batch_size, device)
    #             loss, _ = self.forward_backprop(x, y, h, criterion, optimizer)

    #             del x
    #             del y

    #             self.training_report.train_batch_loss.append(float(loss))
    #             epoch_loss += loss

    #             if show_every_n_batches and (i + 1) % show_every_n_batches == 0:
    #                 print(f"Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss:.4f}")
                
    #             gc.collect()

    #         avg_loss = epoch_loss / len(train_loader)
    #         self.training_report.train_loss.append(avg_loss)

    #         self.eval()
    #         val_loss = 0.0
    #         with torch.no_grad():
    #             for x, y in val_loader:
    #                 x, y = x.to(device), y.to(device)

    #                 batch_size = x.size(0)

    #                 h = self.init_hidden(batch_size, device)
    #                 logits, _ = self.forward(x, h)

    #                 del x

    #                 logits_last = logits[:, -1, :]

    #                 loss = criterion(logits_last, y)

    #                 del y
    #                 del logits
    #                 del logits_last

    #                 val_loss += loss.item()

    #         gc.collect()

    #         avg_val_loss = val_loss / len(val_loader)

    #         self.training_report.val_loss.append(avg_val_loss)
    #         print(f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # def test_model(
    #     self,
    #     test_loader: DataLoader,
    #     criterion: nn.Module,
    #     device: torch.device,
    # ) -> float:
    #     self.eval()
    #     test_loss = 0.0

    #     with torch.no_grad():
    #         for x, y in test_loader:
    #             x, y = x.to(device), y.to(device)

    #             batch_size = x.size(0)

    #             h = self.init_hidden(batch_size, device)
    #             logits, _ = self.forward(x, h)

    #             del x

    #             logits_last = logits[:, -1, :]

    #             loss = criterion(logits_last, y)

    #             del y
    #             del logits
    #             del logits_last

    #             test_loss += loss.item()

    #     gc.collect()

    #     avg_test_loss = test_loss / len(test_loader)
    #     return avg_test_loss

    # def generate(
    #     self,
    #     start_sequence: str,
    #     tokenizer: AutoTokenizer,
    #     device: torch.device,
    #     predict_len: int = 100,
    #     temperature: float = 1.0,
    # ) -> str:
    #     self.eval()

    #     input_ids = tokenize_text(tokenizer, start_sequence)
    #     input_tensor = torch.tensor([input_ids], device=device)
    #     h = self.init_hidden(1, device)

    #     with torch.no_grad():
    #         _, h = self.forward(input_tensor, h)

    #     generated_ids = input_ids.copy()
    #     input_id = input_tensor[:, -1:]

    #     for _ in range(predict_len):
    #         logits, h = self.forward(input_id, h)
    #         logits = logits[:, -1, :] / temperature

    #         probs = F.softmax(logits, dim=-1).to("cpu").detach().numpy().squeeze()
    #         next_token_id = int(np.random.choice(len(probs), p=probs))

    #         del logits
    #         del probs

    #         generated_ids.append(next_token_id)

    #         input_id = torch.tensor([[next_token_id]], device=device)

    #         gc.collect()

    #     return detokenize_text(tokenizer, generated_ids)


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
