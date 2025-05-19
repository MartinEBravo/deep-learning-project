
import os
import torch


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
