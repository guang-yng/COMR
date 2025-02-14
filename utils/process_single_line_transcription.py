# Remember to add PYTHONPATH=$(pwd)
from datasets import load_from_disk
from metrics import remove_comments

def func(example):
    return {"one_line_transcription": remove_comments(example["transcription"])}

if __name__ == "__main__":
    dataset = load_from_disk("datasets/pdmx-v0-clean")
    # ret = func(dataset['train'][69221])
    ds = dataset.map(func, num_proc=16, load_from_cache_file=False)
    for split in ds:
        if None in ds[split]['one_line_transcription']:
            print(f"None in {split}")
            breakpoint()
            exit
    ds.save_to_disk("datasets/pdmx-v0-clean-extended")