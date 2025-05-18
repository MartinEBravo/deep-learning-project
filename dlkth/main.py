import time
import torch

from dlkth.data_loader import (
    load_data
)
from dlkth.models import Bigram
from utils import save_model, save_results
from dlkth.tokenizer import CharTokenizer


def train_workflow(model_name: str, dataset: str):
    # Load text
    text = load_data(f"data/{dataset}.txt")

    # Load Tokenizer
    tokenizer = CharTokenizer(text)
    encode = tokenizer.encode
    decode = tokenizer.decode
    vocab_size = tokenizer.vocab_size

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Model instantiation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_name == 'bigram':
        model = Bigram(
            vocab_size,
            n_embd=32
            ).to(device)
    elif model_name == 'rnn':
        raise NotImplementedError("RNN training not implemented yet")
    elif model_name == 'transformers':
        raise NotImplementedError("Transformer training not implemented yet")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Train model
    train_losses, val_losses = model.train_model(
        train_data,
        val_data,
        block_size=8,
        batch_size=32,
        learning_rate=1e-2,
        device=device,
        eval_iters=200,
        max_iters=3000,
        eval_interval=300
    )


    # Generate sample
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = decode(model.generate(context, max_new_tokens=500)[0].tolist())

    # Save losses and model
    timestamp = int(time.time())
    losses = {
        "model": model_name,
        "dataset": dataset,
        "time": timestamp,
        "train": train_losses,
        "val": val_losses,
        "sample": sample,
    }
    path = f"checkpoints/{model_name}_{dataset}_{timestamp}"
    save_results(f"{path}.json", results=losses)
    save_model(f"{path}.pt", model=model)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training pipeline")
    parser.add_argument('model', choices=['bigram', 'rnn', 'transformers'], help='The model architecture')
    parser.add_argument('dataset', choices=['el_quijote'], help='Dataset')
    args = parser.parse_args()

    train_workflow(args.model, args.dataset)