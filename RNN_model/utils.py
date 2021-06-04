import torch
import numpy as np
import pandas as pd
from RNN_model.preprocessing import preprocess
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, Dataset
from RNN_model.models import *
from RNN_model.tokenizer import ToxicityDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

file = open('RNN_model/full_comments_v2.txt','r')
docs = file.readlines()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(docs)

# model = NeuralNet(np.zeros((100,100)))
# model.load_state_dict("RNN_model/spoken_form_RNN_model.pt")
# model.eval()
mapping = {0:"Toxicity",1:"Obscence",2:"Threat",3:"Identity attack-Insult",4:"Sexual-explicit",5:"Sedition-Politics",6:"Spam"}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_RNN(sentence, model):
   sentence = preprocess(sentence)
   x_infer = tokenizer.texts_to_sequences([sentence])
   x_infer = pad_sequences(x_infer, maxlen=400)
   infer = ToxicityDataset(x_infer)
   infer_loader = DataLoader(infer, batch_size=1, shuffle=False)
   test_preds = np.zeros((len(infer), 7))
   for i, x_batch in enumerate(infer_loader):
      text = x_batch['text'].to("cpu")
      y_pred = sigmoid(model(text).detach().cpu().numpy())
      test_preds[i * 1:(i+1) * 1, :] = y_pred

   return test_preds[0]

def predict_file_RNN(file, model, batch_size=64):

   if 'csv' in file:
      test = pd.read_csv(file)
      if 'spoken' in file:
        sentences = test.normed_comments.map(lambda x: preprocess(x))
      else:
        sentences = test.comments.map(lambda x: preprocess(x))
   else:
      with open(file) as f:
        sentences = f.read_lines()
        sentences = [i.strip() for i in sentences]

   x_test = tokenizer.texts_to_sequences(sentences)
   x_test = pad_sequences(x_test, maxlen=300)

   test_set    = ToxicityDataset(x_test)
   test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)
    
   test_shape = len(x_test)
   test_preds = np.zeros((test_shape, 7))
    
   tk0 = enumerate(test_loader)
   with torch.no_grad():
        
      for idx, batch in tk0:
         text = batch['text'].to(device)
            
         logits = model(text)
         test_preds[idx*batch_size : (idx+1)*batch_size] = logits.detach().cpu().squeeze().numpy()
        
      preds = torch.sigmoid(torch.tensor(test_preds)).numpy()
      pred_target = list(np.argmax(preds, axis = 1))
      pred_target = [mapping[i] for i in pred_target]
        
   return pred_target, preds
