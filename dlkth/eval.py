import os
import json
import torch
import numpy as np
import tqdm

from dlkth.utils import load_model
from dlkth.tokenizer import CharTokenizer
from dlkth.data_loader import load_data


CHECKPOINTS_DIR = "/vol/checkpoints"
REPORTS_DIR = "/vol/reports"


def calculate_perplexity(model, tokenizer, text, device):
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor(
            tokenizer.encode(text), dtype=torch.long, device=device
        ).unsqueeze(0)
        if input_ids.size(1) < 2:
            return np.nan
        targets = input_ids.clone()
        _, loss = model(input_ids[:, :-1], targets[:, 1:])
        return np.exp(loss.item())


def generate_and_eval(model, tokenizer, device, n_samples=100, max_new_tokens=100):
    outputs = []
    model.eval()
    for _ in tqdm.tqdm(range(n_samples)):
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
        text = tokenizer.decode(generated)
        perplexity = calculate_perplexity(model, tokenizer, text, device)
        outputs.append({"text": text, "perplexity": perplexity})
    return outputs


def eval_all(n_samples=100, max_new_tokens=100):
    results = []
    for filename in os.listdir(CHECKPOINTS_DIR):
        if filename.endswith(".json"):
            path = os.path.join(CHECKPOINTS_DIR, filename)
            with open(path, "r") as f:
                obj = json.load(f)
                model_name = obj["model"]
                dataset_name = obj["dataset"]
                pt_name = filename.replace(".json", ".pt")
                pt_path = os.path.join(CHECKPOINTS_DIR, pt_name)
                if not os.path.isfile(pt_path):
                    print(f"[WARN] Missing {pt_path}, skipping...")
                    continue

                device = "cuda" if torch.cuda.is_available() else "cpu"
                try:
                    model = load_model(pt_name, save_dir=CHECKPOINTS_DIR, device=device)
                except Exception as e:
                    print(f"[ERROR] Can't load model {pt_name}: {e}")
                    continue

                dataset_file = f"{dataset_name}.txt"
                text = load_data(dataset_file)
                tokenizer = CharTokenizer(text)
                model = model.to(device)

                print(f"Evaluating {model_name} on {dataset_name}...")
                outputs = generate_and_eval(
                    model,
                    tokenizer,
                    device,
                    n_samples=n_samples,
                    max_new_tokens=max_new_tokens,
                )
                perplexities = [
                    s["perplexity"] for s in outputs if not np.isnan(s["perplexity"])
                ]
                mean_ppl = float(np.mean(perplexities))
                std_ppl = float(np.std(perplexities))

                results.append(
                    {
                        "model": model_name,
                        "dataset": dataset_name,
                        "mean_perplexity": mean_ppl,
                        "std_perplexity": std_ppl,
                        "samples": outputs,
                    }
                )

    os.makedirs(REPORTS_DIR, exist_ok=True)
    output_path = os.path.join(REPORTS_DIR, "perplexity_samples.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print("Saved report to:", output_path)
