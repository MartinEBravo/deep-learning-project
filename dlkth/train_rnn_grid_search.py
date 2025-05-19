import time
import torch

from dlkth.data_loader import load_data
from dlkth.models import RNNBaseline
from dlkth.utils import save_model, save_results
from dlkth.tokenizer import CharTokenizer
from dlkth.config import HiddenInit


def train_workflow(
    n_layer: int,
    dropout: int,
    hidden_ini: str,
    dataset: str,
    save_dir="./checkpoints"):
    # Load text
    text = load_data(f"{dataset}.txt")

    # Load Tokenizer
    tokenizer = CharTokenizer(text)
    encode = tokenizer.encode
    decode = tokenizer.decode
    vocab_size = tokenizer.vocab_size

    model_name = f"rnn_nlayer={n_layer}&dropout={dropout}&hidden_ini={hidden_ini}"

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Model instantiation
    begin = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RNNBaseline(
        vocab_size,
        n_embd=128,
        block_size=128,
        hidden_dim=256,
        n_layers=n_layer,
        hidden_init=HiddenInit(hidden_ini),
        dropout=dropout,
    ).to(device)

    train_losses, val_losses = model.train_model(
        train_data,
        val_data,
        block_size=128,
        batch_size=1,
        learning_rate=3e-3,
        device=device,
        eval_iters=200,
        max_iters=3000,
        eval_interval=25,
    )

    # Generate sample
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = decode(model.generate(context, max_new_tokens=500)[0].tolist())
    print(sample)

    # Save losses and model
    timestamp = int(time.time())
    losses = {
        "model": model_name,
        "dataset": dataset,
        "time": timestamp,
        "duration": time.time() - begin,
        "train": train_losses,
        "val": val_losses,
        "sample": sample,
    }
    path = f"{model_name}_{dataset}_{timestamp}"
    save_results(f"{path}.json", results=losses, save_dir=save_dir)
    save_model(f"{path}.pt", model=model, save_dir=save_dir)
