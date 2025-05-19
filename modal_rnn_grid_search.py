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
def train_main(n_layer: int, dropout: int, hidden_init: str, dataset: str):
    import sys
    sys.path.append("/root/project")
    from dlkth.train_rnn_grid_search import train_workflow

    train_workflow(
        n_layer=n_layer,
        dropout=dropout,
        hidden_ini=hidden_init,
        dataset=dataset,
        save_dir="/vol/checkpoints",
    )
    checkpoints_vol.commit()


@app.local_entrypoint()
def main():
    datasets = [
        # "el_quijote", 
        "valenciano"
        # "shakespeare"
    ]
    n_layers = [1, 2, 3]
    dropouts = [0.0, 0.1, 0.2]
    hidden_inits = ["zeros", "random", "xavier"]

    for dataset in datasets:
        for n_layer in n_layers:
            for dropout in dropouts:
                for hidden_init in hidden_inits:
                    train_main.remote(
                        n_layer=n_layer,
                        dropout=dropout,
                        hidden_init=hidden_init,
                        dataset=dataset,
                    )
