from dataclasses import dataclass

@dataclass
class ModelArguments:
    encoder_config_path: str
    decoder_config_path: str
    tokenizer_path: str