import torch
import os
from PIL import Image
from torchvision import transforms
from transformers import (
    PreTrainedTokenizerFast,
    AutoConfig,
    AutoModel,
    GenerationConfig
)
from models import SMTModel 
from safetensors import safe_open
import datasets


if __name__ == "__main__":
    checkpoint = "outputs/SMT-pdmx-small/checkpoint-94464"
    tokenizer_path = "tokenizers/musescoreabc314_hf_tokenizer.json"
    img = Image.open("libertango_rendered.jpg").convert("RGB")
    # dataset = datasets.load_from_disk("datasets/pdmx-v0-clean")
    # img = dataset['test'][0]['image']

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    model = SMTModel.from_pretrained(checkpoint)
    model = model.to('cuda')
    model.eval()
    tokenizer.model_max_length = model.config.decoder.max_position_embeddings
    model_image_width = model.config.encoder.image_size

    transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda img: img.resize(
                    (model_image_width, int(img.height * model_image_width/ img.width))
                )
            ),
            transforms.RandomInvert(p=1.0),
            transforms.Grayscale(), transforms.ToTensor(),
        ]
    )
    img = transform(img).unsqueeze(0).to('cuda')

    generation_config = GenerationConfig(
        max_length=2048,
        num_beams=3
    )
    outputs = model.generate(generation_config=generation_config, pixel_values=img)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))