import pandas as pd
import numpy as np
import torch
from underthesea import word_tokenize
from transformers import RobertaTokenizer, BertConfig, AutoTokenizer, RobertaConfig, AutoConfig
from torch.utils.data import DataLoader, Dataset

# load model phoBERT
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
phoBERT = 'vinai/phobert-base'
tokenizer_phoBERT = AutoTokenizer.from_pretrained(phoBERT, use_fast=False)
model_phoBERT = torch.load('PhoBERT_model/weights/spoken_form_phoBert_model_v2.pt', map_location='cpu')
model_phoBERT.eval()

mapping = {0:"Toxicity",1:"Obscence",2:"Threat",3:"Identity attack-Insult",4:"Sexual-explicit",5:"Sedition-Politics",6:"Spam"}

class ToxicityDataset(Dataset):
    def __init__(self, text, tokenizer, max_len):
        self.text = text
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        data = convert_samples(
            text = self.text[item], 
            tokenizer = self.tokenizer,
            max_len = self.max_len,
        )

        return {
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
        }

def convert_samples(text, tokenizer, max_len, labels=None):

    input_ids = tokenizer.encode_plus(text, padding='max_length', max_length=max_len, truncation=False)
    if len(input_ids['input_ids']) > max_len:
        input_ids_orig = input_ids['input_ids'][:150] + input_ids['input_ids'][-(max_len - 150):]
        attention_masks = input_ids['attention_mask'][:150] + input_ids['attention_mask'][-(max_len - 150):]
        token_type_ids = input_ids['token_type_ids'][:150] + input_ids['token_type_ids'][-(max_len - 150):]
    else:
        input_ids_orig = input_ids['input_ids']
        attention_masks = input_ids['attention_mask']
        token_type_ids = input_ids['token_type_ids']

    if labels is not None:
        return {
                'ids': input_ids_orig,
                'mask': mask,
                'token_type_ids': token_type_ids,
                'label': label,
                }
    return  {
            'ids': input_ids_orig,
            'mask': attention_masks,
            'token_type_ids': token_type_ids,
            }

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


def predict_phoBERT(sentences, model=model_phoBERT, tokenizer=tokenizer_phoBERT, max_seq_len=256, device=device):
    # sentences = word_tokenize(sentences, format="text")
    input_ids, attention_masks = convert_samples([sentences], tokenizer, max_seq_len)
    model.eval()

    y_pred = model(input_ids.to(device), attention_masks.to(device), token_type_ids=None )
    y_pred = y_pred.squeeze().detach().cpu().numpy()

    return y_pred


def predict_file_phoBERT(file, model = model_phoBERT, tokenizer = tokenizer_phoBERT, batch_size=16, max_len=256):

    if 'csv' in file:
      test = pd.read_csv(file)
      if 'spoken' in file:
        sentences = test.normed_comments.values
      else:
        sentences = test.comments.values
    else:
      with open(file) as f:
        sentences = f.read_lines()
        sentences = [i.strip() for i in sentences]

    test_set    = ToxicityDataset(sentences, tokenizer, max_len)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)
    
    test_shape = len(sentences)
    test_preds = np.zeros((test_shape, 7))
    
    tk0 = (enumerate(test_loader))
    with torch.no_grad():
        
        for idx, batch in tk0:
            input_ids, input_masks, input_segments = batch['ids'], batch['mask'], batch['token_type_ids']
            input_ids, input_masks, input_segments = input_ids.to(device), input_masks.to(device), input_segments.to(device)
            
            logits = model(input_ids = input_ids,
                             attention_mask = input_masks,
                             token_type_ids = input_segments,
                            )
            
            test_preds[idx*batch_size : (idx+1)*batch_size] = logits.detach().cpu().squeeze().numpy()
        
        preds = torch.sigmoid(torch.tensor(test_preds)).numpy()
        pred_target = list(np.argmax(preds, axis = 1))
        pred_target = [mapping[i] for i in pred_target]
        
    return pred_target, preds