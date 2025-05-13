import time
import torch
from torch import nn

from dlkth.data_loader import (
    batch_data,
    load_data,
    preprocess_and_save_data,
    load_preprocess,
)
from dlkth.config import (
    config_rnn,
    config_transformer
)
from dlkth.models.rnn import RNN, generate
from dlkth.models.utils import (
    save_model, 
    load_model
)
from dlkth.models.transformer import Transformer
from dlkth.tokenizer import get_tokenizer, tokenize_text, detokenize_text

def run_workflow(
        model_name: str = "rnn" # "rnn" or "transformer"
    ):

    # Tokenization
    tokenizer = get_tokenizer(config.tokenizer_model_name)
    preprocess_and_save_data(dataset_path=config.data_dir, tokenizer=tokenizer)
    tokens, vocab_to_int, _ = load_preprocess()

    # Create Dataset
    train_loader = batch_data(
        words=tokens,
        sequence_length=config.text_sequence_length,
        batch_size=config.text_batch_size,
    )
    vocab_size = len(vocab_to_int)
    output_size = vocab_size

    
    device = torch.device(config.device_name)

    if model_name.lower() == "rnn":
        config = config_rnn
        model = RNN(
            vocab_size=vocab_size,
            output_size=output_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dropout=config.dropout,
        )

    elif model_name.lower() == "transformer":
        config = config_transformer
        model = Transformer( 
            vocab_size=vocab_size,
            n_embd=config.embedding_dim,
            n_head=config.n_head,
            n_layer=config.n_layers,
            block_size=config.block_size
            dropout=config.dropout,
        )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train_model(
        train_loader=train_loader,
        batch_size=config.text_batch_size,
        optimizer=optimizer,
        criterion=criterion,
        n_epochs=config.num_epochs,
        device=device,
        show_every_n_batches=config.show_every_n_batches,
    )

    save_model(f"../trained_models/trained_{model}_{time.time()}.pt", model)
    print("Model Trained and Saved")

    # _, vocab_to_int, _ = load_preprocess()
    # trained_rnn = load_model(f"../trained_models/trained_{model}.pt")

    # if config.should_generate_text:
    #     pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    #     for prime_text in config.start_text_generated:
    #         encoded = tokenize_text(tokenizer=tokenizer, text=prime_text)
    #         prime_id = encoded[-1] if encoded else pad_token_id

    #         generated_script = generate(
    #             rnn=trained_rnn,
    #             prime_id=prime_id,
    #             pad_token_id=pad_token_id,
    #             device=device,
    #             predict_len=config.generated_token_length,
    #         )

    #         print(detokenize_text(tokenizer=tokenizer, tokens=generated_script))
