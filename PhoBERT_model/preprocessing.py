import numpy as np
import pandas as pd 
import torch

def convert_samples_to_ids(texts, tokenizer, max_seq_length, labels=None):
    input_ids, attention_masks = [], []

    for text in texts:
        inputs = tokenizer.encode_plus(text, padding='max_length', max_length=max_seq_length, truncation=True)
        input_ids.append(inputs['input_ids'])
        masks = inputs['attention_mask']
        attention_masks.append(masks)

    if labels is not None:
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_masks, dtype=torch.long), torch.tensor(
            labels, dtype=torch.long)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_masks, dtype=torch.long)

