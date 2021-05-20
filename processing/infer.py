import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from processing.datasets import process_data

def predict(sentence, model, tokenizer, model_type, max_sequence_length, device):
   if model_type == "LSTM":
      x_infer = tokenizer.texts_to_sequences([sentence])
      x_infer = pad_sequences(x_infer, max_sequence_length)
      x_infer = torch.LongTensor(x_infer, device=device)
      y_pred = model(x_infer)

      return torch.sigmoid(y_pred).detach().cpu().squeeze().numpy()

   else:
      data = process_data(sentence, 0, tokenizer, max_sequence_length)

      ids = torch.tensor(data["ids"], dtype=torch.long, device=device)
      mask = torch.tensor(data["mask"], dtype=torch.long, device=device)
      token_type_ids = torch.tensor(data["token_type_ids"], dtype=torch.long, device=device)
      ids = torch.unsqueeze(ids, 0)
      mask = torch.unsqueeze(mask, 0)
      token_type_ids = torch.unsqueeze(token_type_ids, 0)

      out = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
      preds = torch.sigmoid(out)

      return preds.squeeze().cpu().detach().numpy()
      