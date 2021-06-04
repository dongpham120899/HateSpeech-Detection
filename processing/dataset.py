import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences


def process_data(text, tokenizer, max_len, token_type, label=None):
    
    if token_type == "enviBert":
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
        
    elif token_type == "phoBert":
        input_ids_orig, mask, token_type_ids = [], [], []

        inputs = tokenizer.encode_plus(text, padding='max_length', max_length=max_len, truncation=True)
        input_ids_orig = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            'ids': input_ids_orig,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'label': label,
        }

class ToxicityDataset(Dataset):
    def __init__(self, text, tokenizer, max_len, token_type, label):
        self.text = text
        self.label = label
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.token_type = token_type
    
    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        data = process_data(
            self.text[item], 
            self.tokenizer,
            self.max_len,
            self.token_type,
            self.label[item],
        )

        if self.token_type == "LSTM":
            if self.label is None:
                return {
                    'ids': torch.tensor(self.text[item], dtype=torch.long),
            }
            return {
                'ids': torch.tensor(self.text[item], dtype=torch.long),
                'targets': torch.tensor(self.label[item], dtype=torch.float),
            }
        else:
            if self.label is None:
                return {
                    'ids': torch.tensor(data["ids"], dtype=torch.long),
                    'mask': torch.tensor(data["mask"], dtype=torch.long),
                    'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            }
            return {
                'ids': torch.tensor(data["ids"], dtype=torch.long),
                'mask': torch.tensor(data["mask"], dtype=torch.long),
                'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
                'targets': torch.tensor(data["label"], dtype=torch.float),
            }
            
