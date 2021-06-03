import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset


def process_data(text, label, tokenizer, max_len):
    
    input_ids_orig = tokenizer.tokenize(text)
    if len(input_ids_orig) > max_len - 2:
        # input_ids_orig = input_ids_orig[:max_len - 2]
        input_ids_orig = input_ids_orig[:150] + input_ids_orig[-(max_len - 152):]

    input_ids_orig = ['<s>'] + input_ids_orig + ['</s>']
    input_ids_orig = tokenizer.convert_tokens_to_ids(input_ids_orig)
    
    token_type_ids = [0]*len(input_ids_orig)
    mask = [1] * len(token_type_ids)

    padding_length = max_len - len(input_ids_orig)
    if padding_length > 0:
        input_ids_orig = input_ids_orig + ([1] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

    return {
        'ids': input_ids_orig,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'label': label,
    }
