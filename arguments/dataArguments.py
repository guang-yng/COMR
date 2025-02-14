from dataclasses import dataclass

@dataclass
class DataArguments:
    dataset_path: str
    image_width: int
