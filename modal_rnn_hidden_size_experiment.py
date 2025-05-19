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
def train_main(hidden_size: int, dataset: str):
    import sys
    sys.path.append("/root/project")
    from dlkth.train_rnn_hidden_size_experiment import train_workflow

    train_workflow(hidden_dim=hidden_size, dataset=dataset, save_dir="/vol/checkpoints")
    checkpoints_vol.commit()


@app.local_entrypoint()
def main():
    datasets = [
        # "el_quijote", 
        "valenciano"
        # "shakespeare"
    ]
    hidden_sizes = [64, 128, 256, 512]

    for dataset in datasets:
        for hidden_size in hidden_sizes:
            train_main.remote(hidden_size=hidden_size, dataset=dataset)
