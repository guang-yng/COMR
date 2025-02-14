import fire
import os
import torch
import random
from transformers import (
    Seq2SeqTrainingArguments,
    HfArgumentParser, 
    PreTrainedTokenizerFast,
    Seq2SeqTrainer, 
    AutoConfig,
    VisionEncoderDecoderConfig,
    set_seed
)
from arguments import DataArguments, ModelArguments
from datasets import load_from_disk
from models import SMTModel
from torchvision import transforms
from functools import partial
from metrics import error_rates, remove_special_tokens
from tqdm import tqdm

def collate_fn(tokenizer: PreTrainedTokenizerFast, batch):
    num_channel, _, width = batch[0]['pixel_values'].shape
    max_height = max([example['pixel_values'].shape[1] for example in batch])
    pixel_values = torch.ones((len(batch), num_channel, max_height, width))
    for i, example in enumerate(batch):
        h = example['pixel_values'].shape[1]
        pixel_values[i, :, :h, :] = example['pixel_values']

    new_batch = {'pixel_values': pixel_values}
    tokenized = tokenizer(
        [example['transcription'] for example in batch], 
        padding=True, padding_side='right', 
        truncation=True, max_length=tokenizer.model_max_length,
        return_tensors='pt'
    )
    new_batch['labels'] = tokenized["input_ids"].clone().masked_fill(tokenized["input_ids"] == tokenizer.pad_token_id, -100)[:, 1:]
    return new_batch


def main(config_path: str):
    parser = HfArgumentParser((Seq2SeqTrainingArguments, DataArguments, ModelArguments))
    training_args, data_args, model_args = parser.parse_json_file(json_file=config_path)
    set_seed(training_args.seed)

    torch.set_float32_matmul_precision('high')

    #### Load dataset
    dataset = load_from_disk(data_args.dataset_path)
    # dataset['train'] = dataset['train'].select(range(64))
    dataset['val'] = dataset['val'].select(random.sample(range(len(dataset['val'])), 800))
    dataset['test'] = dataset['test'].select(random.sample(range(len(dataset['test'])), 16))

    transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda img: img.resize(
                    (data_args.image_width, int(img.height * data_args.image_width / img.width))
                )
            ),
            transforms.RandomInvert(p=1.0),
            transforms.Grayscale(), transforms.ToTensor(),
        ]
    )

    #### Load model and tokenizer
    set_seed(training_args.seed)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=model_args.tokenizer_path)
    encoder_config = AutoConfig.from_pretrained(model_args.encoder_config_path)
    decoder_config = AutoConfig.from_pretrained(model_args.decoder_config_path)
    decoder_config.vocab_size = tokenizer.vocab_size
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    config.pad_token_id = tokenizer.pad_token_id
    tokenizer.model_max_length = config.decoder.max_position_embeddings
    tokenizer.bos_token_id = config.decoder.bos_token_id
    tokenizer.eos_token_id = config.decoder.eos_token_id
    model = SMTModel(config=config)

    def tokenize_func(examples):
        return {
            'transcription_tokenized': tokenizer(examples['transcription'], add_special_tokens=False)['input_ids'],
            'one_line_transcription_tokenized': tokenizer(examples['one_line_transcription'], add_special_tokens=False)['input_ids']
        }

    dataset['val'] = dataset['val'].map(tokenize_func, num_proc=training_args.dataloader_num_workers, batched=True)

    def transform_func(example_batch):
        if 'image' in example_batch:
            example_batch['pixel_values'] = [transform(img) for img in example_batch['image']]
        return example_batch

    for split in dataset:
        dataset[split].set_transform(transform_func)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=partial(collate_fn, tokenizer),
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        compute_metrics=partial(error_rates, tokenizer, 
                                training_args.dataloader_num_workers, 
                                dataset['val']['transcription_tokenized'],
                                dataset['val']['one_line_transcription_tokenized']),
    )

    #### Train
    if training_args.do_train:
        trainer.train()#resume_from_checkpoint="outputs/SMT-pdmx-small/checkpoint-23616")

    #### Evaluate
    if training_args.do_eval:
        checkpoints = os.listdir(training_args.output_dir)
        for checkpoint in sorted(checkpoints, key=lambda x: int(x.split("-")[1])):
            if "checkpoint" in checkpoint:
                print(f"evaluating {checkpoint}")
                path = os.path.join(training_args.output_dir, checkpoint)
                trainer._load_from_checkpoint(path)
                result = trainer.evaluate()
                print(result)

    if training_args.do_predict:
        checkpoints = [file for file in os.listdir(training_args.output_dir) if file.startswith('checkpoint')]
        checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        checkpoint = "checkpoint-94464"
        trainer._load_from_checkpoint(os.path.join(training_args.output_dir, checkpoint))
        result = trainer.predict(dataset['test'])

        special_tokens = [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, -100]
        preds = remove_special_tokens(result.predictions, special_tokens)
        label_ids = remove_special_tokens(result.label_ids, special_tokens)

        preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
        refs_text = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        os.makedirs("examples", exist_ok=True)
        for i, (pred, ref) in enumerate(tqdm(zip(preds_text, refs_text), desc="Saving examples...")):
            img = dataset['test'][i]['image']
            print(dataset['test'][i]['filename'])
            img.save(f"examples/{i}.png")
            with open(f"examples/pred_{i}.txt", "w") as f:
                f.write(pred)
            with open(f"examples/ref_{i}.txt", "w") as f:
                f.write(ref)
    

if __name__ == "__main__":
    fire.Fire(main)