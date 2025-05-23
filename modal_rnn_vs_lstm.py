import modal

checkpoints_vol = modal.Volume.from_name("checkpoints", create_if_missing=True)

app = modal.App(name="train")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy<2", "pandas", "tqdm", "transformers")    
    .add_local_dir(".", "/root/project") 
    .add_local_dir("data", "/root/project/data")
)

@app.function(
    gpu="H100",
    image=image,
    timeout=60*60,
    volumes={"/vol/checkpoints": checkpoints_vol}
)
def train_main(model_name: str, dataset: str):
    import sys
    sys.path.append("/root/project")
    from dlkth.train_rnn_vs_lstm import train_workflow

    train_workflow(model_name, dataset, save_dir="/vol/checkpoints")
    checkpoints_vol.commit()


@app.local_entrypoint()
def main():
    datasets = [
        # "el_quijote", 
        "valenciano"
        # "shakespeare"
    ]
    models = [
        "rnn_baseline",
        "lstm1",
        "lstm2",
        "lstm3",
    ]

    for dataset in datasets:
        for model in models:
            train_main.remote(model, dataset)
