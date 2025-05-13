import os
import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer

from .tokenizer import tokenize_text


def load_data(path: str) -> str:
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data


def batch_data(
    words: str,
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
    n_batches = len(words)//batch_size
    x, y = [], []
    words = words[:n_batches*batch_size]
    
    for ii in range(0, len(words)-sequence_length):
        i_end = ii+sequence_length        
        batch_x = words[ii:ii+sequence_length]
        x.append(batch_x)
        batch_y = words[i_end]
        y.append(batch_y)
    
    data = TensorDataset(torch.from_numpy(np.asarray(x)), torch.from_numpy(np.asarray(y)))
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
