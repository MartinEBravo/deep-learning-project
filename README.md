# Back to the Present: Ancient Speech in Our Days with RNNs & Transformers

This project explores the use of RNNs and Transformer-based models to generate text in ancient or classical languages, particularly focusing on texts such as *El Quijote* and *Tirant lo Blanc* (in ancient Valencian). The ultimate goal is to bring historical speech styles back to life using modern neural architectures.

## 📚 Description

We compare three architectures:

- `Bigram` baseline model
- `RNN` implemented from scratch
- `Transformer` implemented from scratch

Each model is trained on ancient literary texts to replicate their structure and style. We report training loss curves, text samples, and evaluate each model using perplexity on generated outputs.

## 📁 Project Structure

- `dlkth/`: Source code including models, tokenizer, training pipeline.
- `data/`: Text corpora (e.g., `el_quijote.txt`, `valenciano.txt`)
- `checkpoints/`: Saved weights and training metadata.
- `reports/`: Evaluation metrics (JSON) and figures (PDF).
- `modal_train.py`: Modal-compatible training launcher.
- `modal_eval.py`: Evaluation script to generate text + compute perplexities.

## 🧪 Training

To train models (on Modal):

```bash
make train
```

## 📊 Evaluation

Generate 100 text samples and compute mean + std perplexity:

```bash
make eval
```

The script:
- Loads each `.pt` checkpoint in `/vol/checkpoints`
- Matches it with its `.json` metadata
- Reconstructs the tokenizer from the original dataset
- Generates samples and computes perplexity
- Saves summary report to `/vol/reports/perplexity_summary.json`

Example result:

```json
[
  {
    "model": "transformer",
    "dataset": "valenciano",
    "mean_perplexity": 2.57,
    "std_perplexity": 0.47,
    "samples": [
      {"text": "En lo temps que lo rey anava...", "perplexity": 2.43}
    ]
  }
]
```

## 📈 Loss Curves

Loss curves are plotted for each model/dataset combination. The figure is saved to PDF using LaTeX formatting (NeurIPS-style):

```bash
notebook plot.ipynb
```

Outputs:

- `reports/loss_plot.pdf`

## 💾 Setup

```bash
pip install -e .
make download
```

### Requirements
- `torch`
- `numpy`
- `transformers`
- `matplotlib`
- `modal`
- `tqdm`
- `pandas`

## 🗃 Volumes

We use two Modal Volumes:

- `checkpoints`: stores model weights and metadata
- `reports`: stores evaluation outputs (JSON, PDFs)

To sync locally:

```bash
make download
```

## 👥 Authors

Martín Bravo, Álvaro Mazcuñán, Adriana Rodríguez  
KTH Royal Institute of Technology  
Stockholm, Sweden
