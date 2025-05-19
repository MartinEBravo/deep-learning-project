import torch
import numpy as np
from dlkth.tokenizer import CharTokenizer
from dlkth.data_loader import load_data
from dlkth.models import Bigram, Transformer
from dlkth.utils import load_model

def calculate_loss_and_perplexity(model, tokenizer, texts, device="cpu"):
    model.eval()
    losses = []
    for text in texts:
        encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, loss = model(encoded[:, :-1], encoded[:, 1:])
        losses.append(loss.item())
    avg_loss = np.mean(losses)
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity

def generate_texts(model, tokenizer, num_texts=100, length=200, device="cpu"):
    model.eval()
    generated_texts = []
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    for _ in range(num_texts):
        out_idx = model.generate(context, max_new_tokens=length)[0].tolist()
        text = tokenizer.decode(out_idx)
        generated_texts.append(text)
    return generated_texts

def main(checkpoint_path, dataset_path, device="cpu"):
    # 1. Load dataset and tokenizer
    text = load_data(dataset_path)
    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size

    # 2. Load model
    model = load_model(checkpoint_path)
    model = model.to(device)

    # 3. Generate texts
    generated_texts = generate_texts(model, tokenizer, num_texts=100, length=200, device=device)

    # 4. Calculate loss and perplexity for each generated text
    avg_loss, perplexity = calculate_loss_and_perplexity(model, tokenizer, generated_texts, device=device)

    print(f"Promedio de Loss en 100 textos: {avg_loss:.4f}")
    print(f"Perplexity promedio: {perplexity:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=["bigram", "transformer"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="el_quijote.txt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(args.model_name, args.checkpoint, args.dataset, device=args.device)