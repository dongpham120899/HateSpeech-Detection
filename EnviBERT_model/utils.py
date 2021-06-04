import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from EnviBERT_model.dataset import process_data
from EnviBERT_model.tokenizer import XLMRobertaTokenizer
from torch.utils.data import DataLoader, Dataset

# load model enviBERT
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
# enviBERT = 'EnviBERT_model/enviBERT'
# tokenizer_enviBERT = XLMRobertaTokenizer(enviBERT)
# model_enviBERT = torch.load('EnviBERT_model/weights/spoken_form_EnviBERT_model_v2.pt', map_location='cpu')
# model_enviBERT.eval()

mapping = {0:"Toxicity",1:"Obscence",2:"Threat",3:"Identity attack-Insult",4:"Sexual-explicit",5:"Sedition-Politics",6:"Spam"}

class ToxicityDataset(Dataset):
    def __init__(self, text, tokenizer, max_len):
        self.text = text
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        data = process_data(
            text = self.text[item], 
            label = None,
            tokenizer = self.tokenizer,
            max_len = self.max_len,
        )

        return {
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
        }

def predict_enviBERT(sentence, tokenizer, model, max_len = 256, device=device):
  data = process_data(sentence, 0, tokenizer, max_len)

  ids = torch.tensor(data["ids"], dtype=torch.long, device=device)
  mask = torch.tensor(data["mask"], dtype=torch.long, device=device)
  token_type_ids = torch.tensor(data["token_type_ids"], dtype=torch.long, device=device)
  ids = torch.unsqueeze(ids, 0)
  mask = torch.unsqueeze(mask, 0)
  token_type_ids = torch.unsqueeze(token_type_ids, 0)

  out = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
  preds = torch.sigmoid(out)

  return preds.squeeze().cpu().detach().numpy()

def predict_file_enviBERT(file, tokenizer, model, batch_size=16, max_len=256):

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

    # sentences = sentences[:30]
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