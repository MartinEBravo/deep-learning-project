# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
#max_iters = 5000
max_iters = 2000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
#n_embd = 384
n_embd = 256
#n_head = 6
#n_layer = 6
n_head = 4
n_layer = 4
#dropout = 0.2
dropout = 0.4
# ------------
input_file = '../data/book.txt'