import pickle

import os
import torch
from torch.utils.data import TensorDataset, DataLoader


def get_data_path(filename):
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, "../data", filename)


def load_data(filename):
    path = get_data_path(filename)
    with open(path, "r") as f:
        return f.read()


def get_batch(split, train_data, val_data, block_size, batch_size, device):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def get_rnn_train_loader(
    tokens: list[int],
    sequence_length: int,
    batch_size: int,
) -> DataLoader:
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    n_batches = len(tokens) // batch_size
    tokens = tokens[: n_batches * batch_size]
    x, y = [], []

    for ii in range(0, len(tokens) - sequence_length):
        i_end = ii + sequence_length
        batch_x = tokens[ii : ii + sequence_length]
        x.append(batch_x)
        batch_y = tokens[i_end]
        y.append(batch_y)
    data = TensorDataset(torch.tensor(x), torch.tensor(y))
    return DataLoader(data, shuffle=True, batch_size=batch_size)


def load_preprocess() -> tuple:
    return pickle.load(open("preprocess.p", mode="rb"))
