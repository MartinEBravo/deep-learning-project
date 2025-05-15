import pickle

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer

from dlkth.tokenizer import tokenize_text


def load_data(path: str) -> str:
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()
    return data


def batch_data(
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
    x, y = [], []
    tokens = tokens[: n_batches * batch_size]

    for ii in range(0, len(tokens) - sequence_length):
        i_end = ii + sequence_length
        batch_x = tokens[ii : ii + sequence_length]
        x.append(batch_x)
        batch_y = tokens[i_end]
        y.append(batch_y)

    data = TensorDataset(
        torch.from_numpy(np.asarray(x)), torch.from_numpy(np.asarray(y))
    )
    data_loader = DataLoader(data, shuffle=True, batch_size=batch_size)

    return data_loader


def batch_block_data(
    blocks_of_tokens: list[list[int]],
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
    x, y = [], []
    tokens = tokens[: n_batches * batch_size]

    for ii in range(0, len(tokens) - sequence_length):
        i_end = ii + sequence_length
        batch_x = tokens[ii : ii + sequence_length]
        x.append(batch_x)
        batch_y = tokens[i_end]
        y.append(batch_y)

    data = TensorDataset(
        torch.from_numpy(np.asarray(x)), torch.from_numpy(np.asarray(y))
    )
    data_loader = DataLoader(data, shuffle=True, batch_size=batch_size)

    return data_loader


def preprocess_and_save_data(dataset_path: str, tokenizer: AutoTokenizer) -> None:
    text = load_data(dataset_path)

    tokens = tokenize_text(tokenizer=tokenizer, text=text)

    vocab_to_int = tokenizer.get_vocab()
    int_to_vocab = {idx: token for token, idx in vocab_to_int.items()}

    pickle.dump((tokens, vocab_to_int, int_to_vocab), open("preprocess.p", "wb"))


def load_preprocess() -> tuple:
    return pickle.load(open("preprocess.p", mode="rb"))


def split_into_blocks(encoded_data, block_size, pad_token_id):
    num_blocks = (len(encoded_data) + block_size - 1) // block_size
    blocks = [
        encoded_data[i * block_size : (i + 1) * block_size] for i in range(num_blocks)
    ]
    blocks = [
        torch.cat([b, torch.full((block_size - len(b),), pad_token_id)])
        if len(b) < block_size
        else b
        for b in blocks
    ]
    return blocks


def load_transformer_data(input_file, encode, block_size, pad_token_id):
    text = load_data(input_file)
    encoded_data = torch.tensor(encode(text), dtype=torch.long)
    blocks = split_into_blocks(encoded_data, block_size, pad_token_id)
    data = torch.cat(blocks)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data


def get_batch(split, train_data, val_data, block_size, batch_size, device):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
