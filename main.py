import modal

app = modal.App()
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy<2", "transformers")
    .add_local_dir(".", "/root/project") 
    .add_local_dir("data", "/root/project/data")
    .add_local_dir("checkpoints", "/root/project/checkpoints")     
)

@app.function(gpu="A100", image=image)
def train_main(model_name: str, dataset: str):
    import sys
    sys.path.append("/root/project")
    from dlkth.train import train_workflow

    train_workflow(model_name, dataset)

@app.local_entrypoint()
def main():
    model_name = "transformer"
    dataset = "el_quijote"
    train_main.remote(model_name, dataset)