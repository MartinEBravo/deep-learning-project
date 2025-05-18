import os
import torch

import json


def load_model(filename: str):
    return torch.load(filename, weights_only=False)


def save_model(filename: str, model):
    torch.save(model, filename)


def save_results(filename: str, results: dict):
    json_object = json.dumps(results, indent=4)
    with open(filename, "w") as outfile:
        outfile.write(json_object)
