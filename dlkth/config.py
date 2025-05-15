from dataclasses import dataclass

import torch

DEFAULT_MODEL = "google-bert/bert-base-cased"


@dataclass
class Config:
    """
    Parent configuration class.
    """

    # Tokenizer parameters
    tokenizer_model_name: str

    # Data parameters
    data_dir: str
    text_sequence_length: int
    text_batch_size: int

    # Training parameters
    dropout: float
    num_epochs: int
    learning_rate: float
    embedding_dim: int
    hidden_dim: int
    n_layers: int  # Miscellaneous parameters
    show_every_n_batches: int
    device_name: str

    # Text generation parameters
    should_generate_text: bool
    generated_token_length: int
    start_text_generated: list[str]


@dataclass
class ConfigRNN(Config):
    """
    Configuration class for the RNN model.
    """

    # M    hidden_dim: int
    pass


@dataclass
class ConfigTransformers(Config):
    """
    Configuration class for the Transformers model.
    """

    n_head: int
    eval_interval: int
    block_size: int


config_rnn = ConfigRNN(
    tokenizer_model_name="google-bert/bert-base-cased",
    data_dir="./data/el_quijote.txt",
    text_sequence_length=10,
    text_batch_size=128,
    num_epochs=15,
    learning_rate=1e-3,
    dropout=0.1,
    embedding_dim=256,
    hidden_dim=256,
    n_layers=6,
    show_every_n_batches=500,
    device_name="cuda" if torch.cuda.is_available() else "cpu",
    should_generate_text=True,
    generated_token_length=1000,
    start_text_generated=["dulcinea"],
)


config_transformer = ConfigTransformers(
    tokenizer_model_name="google-bert/bert-base-cased",
    data_dir="./data/el_quijote.txt",
    text_sequence_length=10,
    text_batch_size=64,
    num_epochs=455,
    learning_rate=1e-3,
    dropout=0.1,
    embedding_dim=256,
    hidden_dim=256,
    n_layers=6,
    show_every_n_batches=500,
    device_name="cuda" if torch.cuda.is_available() else "cpu",
    should_generate_text=True,
    generated_token_length=1000,
    start_text_generated=["dulcinea"],
    n_head=6,
    eval_interval=1,
    block_size=128,
)
