from collections import Counter
import numpy as np
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer
from huggingface_hub import login


login()


tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token



SPECIAL_WORDS = {'PADDING': '<PAD>'}


def load_model(filename):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    return torch.load(save_filename, weights_only=False)


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    token = dict()
    token['.'] = '<PERIOD>'
    token[','] = '<COMMA>'
    token['"'] = 'QUOTATION_MARK'
    token[';'] = 'SEMICOLON'
    token['!'] = 'EXCLAIMATION_MARK'
    token['?'] = 'QUESTION_MARK'
    token['('] = 'LEFT_PAREN'
    token[')'] = 'RIGHT_PAREN'
    token['-'] = 'QUESTION_MARK'
    token['\n'] = 'NEW_LINE'
    return token


def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    n_batches = len(words)//batch_size
    x, y = [], []
    words = words[:n_batches*batch_size]
    
    for ii in range(0, len(words)-sequence_length):
        i_end = ii+sequence_length        
        batch_x = words[ii:ii+sequence_length]
        x.append(batch_x)
        batch_y = words[i_end]
        y.append(batch_y)
    
    data = TensorDataset(torch.from_numpy(np.asarray(x)), torch.from_numpy(np.asarray(y)))
    data_loader = DataLoader(data, shuffle=True, batch_size=batch_size)

    return data_loader


def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden, device):
    """
    Forward and backward propagation on the neural network
    :param decoder: The PyTorch Module that holds the neural network
    :param decoder_optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    rnn.to(device)
    
    # creating variables for hidden state to prevent back-propagation
    # of historical states 
    h = tuple([each.data for each in hidden])
    
    rnn.zero_grad()
    # move inputs, targets to GPU 
    inputs, targets = inp.to(device), target.to(device)
    
    output, h = rnn(inputs, h)
    
    loss = criterion(output, targets)
    
    # perform backpropagation and optimization
    loss.backward()
    nn.utils.clip_grad_norm_(rnn.parameters(), 5)
    optimizer.step()

    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), h


def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, device, show_every_n_batches=100):
    batch_losses = []
    
    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size, device)
        
        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            
            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break
            
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden, device=device)          
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(batch_losses)))
                batch_losses = []

    # returns a trained rnn
    return rnn


def generate(rnn, prime_id, pad_token_id, device, predict_len=100):
    """
    Generate text using the trained RNN and BERT tokenizer.
    """
    rnn.eval()

    sequence_length = 10  # same as training
    current_seq = np.full((1, sequence_length), pad_token_id)
    current_seq[0, -1] = prime_id
    predicted_ids = [prime_id]

    for _ in range(predict_len):
        input_tensor = torch.LongTensor(current_seq).to(device)
        hidden = rnn.init_hidden(input_tensor.size(0), device)

        output, _ = rnn(input_tensor, hidden)
        p = F.softmax(output, dim=1).data.cpu()

        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()
        p = p.numpy().squeeze()

        next_token_id = np.random.choice(top_i, p=p/p.sum())
        predicted_ids.append(next_token_id)

        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1] = next_token_id

    return predicted_ids


def load_data(path: str) -> str:
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data


def preprocess_and_save_data(dataset_path):
    """
    Preprocess Text Data using BERT tokenizer
    """
    text = load_data(dataset_path)

    # Tokenize using BERT tokenizer
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Use tokenizer's vocab
    vocab_to_int = tokenizer.get_vocab()
    int_to_vocab = {idx: token for token, idx in vocab_to_int.items()}

    token_dict = {}  # no longer needed, but keeping for compatibility
    pickle.dump((tokens, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    return pickle.load(open('preprocess.p', mode='rb'))


def save_model(filename, decoder):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    torch.save(decoder, save_filename)


def create_lookup_tables(text):
        """
        Create lookup tables for vocabulary
        :param text: The text of tv scripts split into words
        :return: A tuple of dicts (vocab_to_int, int_to_vocab)
        """
        # TODO: Implement Function
        word_count = Counter(text)
        sorted_vocab = sorted(word_count, key = word_count.get, reverse=True)
        int_to_vocab = {ii:word for ii, word in enumerate(sorted_vocab)}
        vocab_to_int = {word:ii for ii, word in int_to_vocab.items()}
        
        # return tuple
        return (vocab_to_int, int_to_vocab)


class RNN(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        # TODO: Implement function
        
        # define embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # define lstm layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        
        
        # set class variables
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # define model layers
        self.fc = nn.Linear(hidden_dim, output_size)
    
    

    def forward(self, x, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        # TODO: Implement function   
        batch_size = x.size(0)
        x=x.long()
        
        # embedding and lstm_out 
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        # stack up lstm layers
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # dropout, fc layer and final sigmoid layer
        out = self.fc(lstm_out)
        
        # reshaping out layer to batch_size * seq_length * output_size
        out = out.view(batch_size, -1, self.output_size)
        
        # return last batch
        out = out[:, -1]

        # return one batch of output word scores and the hidden state
        return out, hidden
    
    

    def init_hidden(self, batch_size, device):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # create 2 new zero tensors of size n_layers * batch_size * hidden_dim
        weights = next(self.parameters()).data
        hidden = (weights.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device), 
                    weights.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        
        # initialize hidden state with zero weights, and move to GPU if available
        
        return hidden


if __name__ == "__main__":
    data_dir = "./data/el_quijote.txt"
    text = load_data(data_dir)

    view_line_range = (0, 10)


    print('Dataset Stats')
    print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

    lines = text.split('\n')
    print('Number of lines: {}'.format(len(lines)))
    word_count_line = [len(line.split()) for line in lines]
    print('Average number of words in each line: {}'.format(np.average(word_count_line)))

    print()
    print('The lines {} to {}:'.format(*view_line_range))
    print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))

    #preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
    preprocess_and_save_data(data_dir)


    int_text, vocab_to_int, int_to_vocab, token_dict = load_preprocess()


    # test dataloader

    test_text = range(50)
    t_loader = batch_data(test_text, sequence_length=5, batch_size=10)

    data_iter = iter(t_loader)
    sample_x, sample_y = next(data_iter)

    print(sample_x.shape)
    print(sample_x)
    print()
    print(sample_y.shape)
    print(sample_y)

    # Data params
    # Sequence Length
    sequence_length = 10  # of words in a sequence
    # Batch Size
    batch_size = 128

    # data loader - do not change
    train_loader = batch_data(int_text, sequence_length, batch_size)

    # Training parameters
    # Number of Epochs
    num_epochs = 1
    # Learning Rate
    learning_rate = 0.01

    # Model parameters
    # Vocab size
    vocab_size = len(vocab_to_int)
    # Output size
    output_size = vocab_size
    # Embedding Dimension
    embedding_dim = 200
    # Hidden Dimension
    hidden_dim = 250
    # Number of RNN Layers
    n_layers = 2

    # Show stats for every n number of batches
    show_every_n_batches = 500

    device = torch.device("mps")
    # device = torch.device("cuda")

    rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
    rnn.to(device)

    # defining loss and optimization functions for training
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # training the model
    trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, device, show_every_n_batches)

    # saving the trained model
    save_model('../trained_models/rnn_quijote/trained_rnn', trained_rnn)
    print('Model Trained and Saved')

    _, vocab_to_int, int_to_vocab, token_dict = load_preprocess()
    trained_rnn = load_model('../trained_models/rnn_quijote/trained_rnn')

    # tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    # tokenizer.pad_token = tokenizer.eos_token

    gen_length = 50  # desired generation length
    prime_texts = ["dulcinea"]  # prompt text(s)

    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    for prime_text in prime_texts:
        encoded = tokenizer.encode(prime_text, add_special_tokens=False)
        prime_id = encoded[-1] if encoded else pad_token_id  # fallback if empty
        generated_script = generate(trained_rnn, prime_id, pad_token_id, device, gen_length)
        print(tokenizer.decode(generated_script, skip_special_tokens=True))
