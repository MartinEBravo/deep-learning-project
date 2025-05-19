format:
	ruff format dlkth
	ruff check --fix dlkth

train:
	modal run main.py

download:
	modal volume get --force checkpoints / checkpoints