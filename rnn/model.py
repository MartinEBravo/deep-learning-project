import os

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class RNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        output_size: int,
        embedding_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float = 0.5,
    ) -> None:
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x=x.long()

        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        out = self.fc(lstm_out)

        out = out.view(batch_size, -1, self.output_size)
        
        out = out[:, -1]

        return out, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        weights = next(self.parameters()).data
        hidden = (weights.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device), 
                    weights.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        
        return hidden

    def forward_back_prop(
        self,
        optimizer: Optimizer,
        criterion: nn.Module,
        inp: torch.Tensor,
        target: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> tuple[float, tuple[torch.Tensor, torch.Tensor]]:
        self.to(device)
        
        h = tuple([each.data for each in hidden])
        
        self.zero_grad()
        inputs, targets = inp.to(device), target.to(device)
        
        output, h = self.forward(inputs, h)
        
        loss = criterion(output, targets)
        
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 5)
        optimizer.step()

        return loss.item(), h

    def train_model(
        self,
        train_loader: DataLoader,
        batch_size: int,
        optimizer: Optimizer,
        criterion: nn.Module,
        n_epochs: int,
        device: torch.device,
        show_every_n_batches=100,
    ) -> None:
        batch_losses = []

        print("Training for %d epoch(s)..." % n_epochs)
        for epoch_i in range(1, n_epochs + 1):
            hidden = self.init_hidden(batch_size, device)
            
            for batch_i, (inputs, labels) in enumerate(train_loader, 1):
                n_batches = len(train_loader.dataset)//batch_size

                if(batch_i > n_batches):
                    break
                
                loss, hidden = self.forward_back_prop(
                    optimizer=optimizer,
                    criterion=criterion,
                    inp=inputs,
                    target=labels,
                    hidden=hidden,
                    device=device,
                )          
                batch_losses.append(loss)

                if batch_i % show_every_n_batches == 0:
                    print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                        epoch_i, n_epochs, np.average(batch_losses)))
                    batch_losses = []

        print("Training complete.")


def load_model(filename: str) -> RNN:
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    return torch.load(save_filename, weights_only=False)


def save_model(filename: str, model: RNN):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    torch.save(model, save_filename)
