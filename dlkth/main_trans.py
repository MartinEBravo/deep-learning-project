import torch
from torch import nn

from dlkth.data_loader import (
    batch_data,
    load_data,
    preprocess_and_save_data,
    load_preprocess,
)
from dlkth.config import config_rnn as config
from dlkth.models.transformer import GPTLanguageModel
from dlkth.tokenizer import get_tokenizer, tokenize_text, detokenize_text


if __name__ == "__main__":
    text = load_data(config.data_dir)
    tokenizer = get_tokenizer(config.tokenizer_model_name)

    preprocess_and_save_data(dataset_path=config.data_dir, tokenizer=tokenizer)

    tokens, vocab_to_int, int_to_vocab = load_preprocess()

    train_loader = batch_data(
        words=tokens,
        sequence_length=config.text_sequence_length,
        batch_size=config.text_batch_size,
    )

    vocab_size = len(vocab_to_int)
    output_size = vocab_size

    device = torch.device(config.device_name)

    gpt = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=config.embedding_dim,
        n_head=config.n_head,
        n_layer=config.n_layers,
        block_size=config.text_sequence_length,
        output_size=output_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        dropout=config.dropout,
    )
    gpt.to(device)

    optimizer = torch.optim.Adam(gpt.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    gpt.train_model(
        train_loader=train_loader,
        batch_size=config.text_batch_size,
        optimizer=optimizer,
        criterion=criterion,
        n_epochs=config.num_epochs,
        device=device,
        show_every_n_batches=config.show_every_n_batches,
    )

    # save_model("../trained_models/rnn_quijote/trained_rnn", rnn)
    # print("Model Trained and Saved")

    # _, vocab_to_int, int_to_vocab = load_preprocess()
    # trained_rnn = load_model("../trained_models/rnn_quijote/trained_rnn")

    if config.should_generate_text:
        pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

        for prime_text in config.start_text_generated:
            encoded = tokenize_text(tokenizer=tokenizer, text=prime_text)
            prime_id = encoded[-1] if encoded else pad_token_id

            generated_script = gpt.generate(
                rnn=gpt,
                prime_id=prime_id,
                pad_token_id=pad_token_id,
                device=device,
                predict_len=config.generated_token_length,
            )

            print(detokenize_text(tokenizer=tokenizer, tokens=generated_script))
