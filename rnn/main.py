import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .config import config
from .data_loader import batch_data, load_data, preprocess_and_save_data, load_preprocess
from .model import RNN, save_model, load_model
from .tokenizer import get_tokenizer, tokenize_text, detokenize_text


def generate(
    rnn: RNN,
    prime_id: int,
    pad_token_id: int,
    device: torch.device,
    predict_len: int = 100,
) -> list[int]:
    sequence_length = 10  # same as training
    current_seq = np.full((1, sequence_length), pad_token_id)
    current_seq[0, -1] = prime_id
    predicted_ids = [prime_id]

    for _ in range(predict_len):
        input_tensor = torch.LongTensor(current_seq).to(device)
        hidden = rnn.init_hidden(input_tensor.size(0), device)

        output, _ = rnn.forward(input_tensor, hidden)
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

    rnn = RNN(
        vocab_size=vocab_size,
        output_size=output_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        dropout=0.5,
    )
    rnn.to(device)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    
    rnn.train_model(
        train_loader=train_loader,
        batch_size=config.text_batch_size,
        optimizer=optimizer,
        criterion=criterion,
        n_epochs=config.num_epochs,
        device=device,
        show_every_n_batches=config.show_every_n_batches,
    )

    save_model('../trained_models/rnn_quijote/trained_rnn', rnn)
    print('Model Trained and Saved')

    _, vocab_to_int, int_to_vocab = load_preprocess()
    trained_rnn = load_model('../trained_models/rnn_quijote/trained_rnn')

    if config.should_generate_text:
        pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

        for prime_text in config.start_text_generated:
            encoded = tokenize_text(tokenizer=tokenizer, text=prime_text)
            prime_id = encoded[-1] if encoded else pad_token_id

            generated_script = generate(
                rnn=trained_rnn,
                prime_id=prime_id,
                pad_token_id=pad_token_id,
                device=device,
                predict_len=config.generated_token_length,
            )

            print(detokenize_text(tokenizer=tokenizer, tokens=generated_script))
