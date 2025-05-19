format:
	ruff format dlkth
	ruff check --fix dlkth

train:
	modal run modal_train.py

eval:
	modal run modal_eval.py

download:
	modal volume get --force checkpoints / checkpoints
	modal volume get --force reports / reports