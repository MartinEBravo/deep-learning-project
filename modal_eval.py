import modal

checkpoints_vol = modal.Volume.from_name("checkpoints", create_if_missing=True)
reports_vol = modal.Volume.from_name("reports", create_if_missing=True)

app = modal.App(name="eval")
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
    volumes={
        "/vol/checkpoints": checkpoints_vol,
        "/vol/reports": reports_vol,
    },
)
def run_eval():
    import sys
    sys.path.append("/root/project")
    from dlkth.eval import eval_all
    eval_all()

@app.local_entrypoint()
def main():
    run_eval.remote()