import torch
import os
import json


def get_checkpoint_path(filename, save_dir=None):
    if save_dir is not None:
        return os.path.join(save_dir, filename)
    # si no se especifica save_dir, usa por defecto el antiguo comportamiento
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, "../checkpoints", filename)


def load_model(filename: str, save_dir=None):
    path = get_checkpoint_path(filename, save_dir)
    return torch.load(path, weights_only=False)


def save_model(filename: str, model, save_dir=None):
    path = get_checkpoint_path(filename, save_dir)
    torch.save(model, path)


def save_results(filename: str, results: dict, save_dir=None):
    json_object = json.dumps(results, indent=4)
    path = get_checkpoint_path(filename, save_dir)
    with open(path, "w") as outfile:
        outfile.write(json_object)
