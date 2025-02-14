from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
from tokenizers.pre_tokenizers import Split, Sequence
from tokenizers import Regex, decoders, AddedToken
import datasets

dataset_name = "datasets/pdmx-v0-clean"
save_name = "pdmx_v0_tokenizer"

def get_abc_content():
    data = datasets.load_from_disk(dataset_name)
    if 'train' in data:
        data = data['train']
    data_entries = data['transcription']
    for abc_content in data_entries:
        yield abc_content


tokenizer = Tokenizer(BPE(unk_token="<|unkown|>"))
trainer = BpeTrainer(
    vocab_size=2048, 
    show_progress=True, 
    special_tokens=["<|unkown|>", "<|begin_of_abc|>", "<|end_of_abc|>", "<|text|>", "<|pad|>"]
)
pre_tokenizer = Sequence([
    Split("\n", behavior="isolated"),
    Split("<|text|>", behavior="removed"),
    Split(Regex(r"\||\[|\]|\"|%|\!"), behavior="isolated"),
    Split(Regex(r"\s"), behavior="merged_with_next")
])
tokenizer.pre_tokenizer = pre_tokenizer
tokenizer.enable_padding(direction="left", pad_id=4, pad_token="<|pad|>")
tokenizer.enable_truncation(max_length=2048)

tokenizer.train_from_iterator(get_abc_content(), trainer)
tokenizer.add_tokens(['<|text|>'])
tokenizer.post_processor = TemplateProcessing(
    single="<|begin_of_abc|> $A <|end_of_abc|>",
    special_tokens=[
        ("<|begin_of_abc|>", tokenizer.token_to_id("<|begin_of_abc|>")),
        ("<|end_of_abc|>", tokenizer.token_to_id("<|end_of_abc|>")),
    ],
)
tokenizer.decoder = decoders.BPEDecoder()
tokenizer.save(f"tokenizers/{save_name}.json")

