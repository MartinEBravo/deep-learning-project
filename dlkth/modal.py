import modal

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "transformers")
    .add_local_dir(".", "/root/project")
)
volume = modal.Volume.persisted("dlkth-models")
app = modal.App("dlkth-trainer", image=image)


@app.function(gpu="A100", timeout=60 * 60, mounts={"/checkpoints": volume})
def train_main(model_name: str, dataset: str):
    import sys

    sys.path.append("/root/project")
    from dlkth.train import train_workflow

    train_workflow(model_name, dataset)
    print("Entrenamiento completado.")


@app.local_entrypoint()
def main():
    model_name = "transformers"
    dataset = "el_quijote"
    train_main.remote(model_name, dataset)
