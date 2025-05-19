format:
	ruff format dlkth
	ruff check --fix dlkth

train:
	modal run modal_train.py

rnn_vs_lstm:
	modal run modal_rnn_vs_lstm.py

eval:
	modal run modal_eval.py

download:
	modal volume get --force checkpoints / checkpoints
	modal volume get --force reports / reports