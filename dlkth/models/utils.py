import os
import torch


def load_model(filename: str):
    save_filename = os.path.splitext(os.path.basename(filename))[0]
    return torch.load(save_filename, weights_only=False)


def save_model(filename: str, model):
    save_filename = os.path.splitext(os.path.basename(filename))[0]
    torch.save(model, save_filename)
