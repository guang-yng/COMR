import torch
from typing import Dict, List, Union, Any
from transformers import EvalPrediction, PreTrainedTokenizer
from nltk import edit_distance
from tqdm import tqdm
from .utils import remove_comments_batch
import multiprocessing
import numpy as np
import datasets

def _edit_distance(args):
    return edit_distance(*args)

def error_rate(preds, refs, desc="error rate", num_workers=8):
    with multiprocessing.Pool(num_workers) as pool:
        dists = list(tqdm(
            pool.imap_unordered(_edit_distance, zip(preds, refs)),
            desc=f"computing {desc}...",
            total=len(preds)
        ))
    return sum(dists) / sum(len(r) for r in refs) * 100

def remove_special_tokens(array, special_tokens):
    masks = np.isin(array, special_tokens, invert=True)
    return [a[mask] for a, mask in zip(array, masks)]

def error_rates(tokenizer: PreTrainedTokenizer, num_workers: int, label_ids, one_line_target_ids, p: EvalPrediction) -> Dict[str, float]:
    special_tokens = [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, -100]
    
    preds = remove_special_tokens(p.predictions, special_tokens)
    torch.save(p, 'eval_preds.pt')

    preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
    refs_text = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    clean_preds_text = remove_comments_batch(preds_text, num_workers)
    voice_lyrics_well_formed_rate = sum([t is not None for t in clean_preds_text]) / len(clean_preds_text) * 100
    clean_preds_text = [t1 if t1 is not None else t2 for t1, t2 in zip(clean_preds_text, preds_text)]
    one_line_target = tokenizer.batch_decode(one_line_target_ids, skip_special_tokens=True)

    SER = error_rate(preds, label_ids, "SER", num_workers)
    CER = error_rate(preds_text, refs_text, "CER", num_workers)
    LER = error_rate([p.split('\n') for p in preds_text], [r.split('\n') for r in refs_text], "LER", num_workers)
    
    metrics = {"LER": LER, "CER": CER, "SER": SER, "voice_lyrics_well_formed_rate": voice_lyrics_well_formed_rate}

    CER_one_line = error_rate(clean_preds_text, one_line_target, "CER_one_line", num_workers)
    LER_one_line = error_rate([p.split('\n') for p in clean_preds_text], [r.split('\n') for r in one_line_target], "LER_one_line", num_workers)
    SER_one_line = error_rate(
        tokenizer(clean_preds_text, add_special_tokens=False)['input_ids'], 
        one_line_target_ids,
        "SER_one_line", num_workers
    )
    
    metrics.update({"LER_one_line": LER_one_line, "CER_one_line": CER_one_line, "SER_one_line": SER_one_line})

    return metrics