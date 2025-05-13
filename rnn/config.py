from dataclasses import dataclass


@dataclass
class Config:
    """
    Configuration class for the RNN model.
    """
    # Tokenizer parameters
    tokenizer_model_name: str

    # Data parameters
    data_dir: str
    text_sequence_length: int
    text_batch_size: int

    # Training parameters
    num_epochs: int
    learning_rate: float

    # Model parameters
    embedding_dim: int
    hidden_dim: int
    n_layers: int

    # Miscellaneous parameters
    show_every_n_batches: int
    device_name: str

    # Text generation parameters
    should_generate_text: bool
    generated_token_length: int
    start_text_generated: list[str]


config = Config(
    tokenizer_model_name="google-bert/bert-base-cased",
    data_dir="./data/el_quijote.txt",
    text_sequence_length=10,
    text_batch_size=128,
    num_epochs=1,
    learning_rate=0.01,
    embedding_dim=200,
    hidden_dim=250,
    n_layers=2,
    show_every_n_batches=500,
    device_name="mps",
    should_generate_text=True,
    generated_token_length=1000,
    start_text_generated=["dulcinea"],
)
