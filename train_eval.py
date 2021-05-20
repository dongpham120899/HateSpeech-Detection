import numpy as np
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from tqdm.notebook import tqdm
from processing.utils import AverageMeter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
custom_loss = nn.BCEWithLogitsLoss()

test_cols = ['Toxicity', 'Obscence', 'Threat', 'Identity attack - Insult', 'Sexual explicit', 'Sedition – Politics', 'Spam']

def train_model(model, train_loader, optimizer, scheduler, model_type):
    
    model.train()
    avg_loss = 0.
    losses = AverageMeter()
    tk0 = (enumerate(train_loader))
    
    for idx, batch in tk0:
        if model_type == "LSTM":
            input_ids, labels = batch['ids'], batch['targets']
            input_ids, labels = input_ids.to(device), labels.to(device)
    
            logits = model(input_ids)
        else:
            input_ids, input_masks, input_segments, labels = batch['ids'], batch['mask'], batch['token_type_ids'], batch['targets']
            input_ids, input_masks, input_segments, labels = input_ids.to(device), input_masks.to(device), input_segments.to(device), labels.to(device)
 
            logits = model(input_ids = input_ids,
                            attention_mask = input_masks,
                            token_type_ids = input_segments,
                        )

        loss = custom_loss(logits, labels)  
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        avg_loss += loss.item() / len(train_loader)

        losses.update(loss.item(), input_ids.size(0))
            # tk0.set_postfix(loss=losses.avg)

    torch.cuda.empty_cache()
    gc.collect()
    return avg_loss, model

    
def val_model(model, val_loader, val_shape, batch_size, model_type):

    avg_val_loss = 0.
    model.eval() # eval mode
    
    valid_preds = np.zeros((val_shape, 7))
    original = np.zeros((val_shape, 7))
    
    tk0 = (enumerate(val_loader))
    with torch.no_grad():
        
        for idx, batch in tk0:
            if model_type == "LSTM":
                input_ids, labels = batch['ids'], batch['targets']
                input_ids, labels = input_ids.to(device), labels.to(device)
            
                output_val = model(input_ids)
            else:
                input_ids, input_masks, input_segments, labels = batch['ids'], batch['mask'], batch['token_type_ids'], batch['targets']
                input_ids, input_masks, input_segments, labels = input_ids.to(device), input_masks.to(device), input_segments.to(device), labels.to(device)
                
                output_val = model(input_ids = input_ids,
                                attention_mask = input_masks,
                                token_type_ids = input_segments,
                                )
            
            logits = output_val
            avg_val_loss += custom_loss(logits, labels).item() / len(val_loader)  

            valid_preds[idx*batch_size : (idx+1)*batch_size] = logits.detach().cpu().squeeze().numpy()
            original[idx*batch_size : (idx+1)*batch_size]    = labels.detach().cpu().squeeze().numpy()
        
        preds = torch.sigmoid(torch.tensor(valid_preds)).numpy()
#         preds = np.argmax(preds, 1)

#         score = f1_score(original, preds)
        for i in range(len(test_cols)):
            # preds = np.where(preds < 0.52, 0, 1)
            print('validation for col: ', test_cols[i])
            f1_score_ = f1_score(original[:, i], np.round(preds[:,i]), average='macro')
            print('\r f1_score: %s' % (str(round(f1_score_, 5))), end = 100*' '+'\n')
            roc_auc = roc_auc_score(original[:, i], preds[:,i])
            print('\r roc_auc: %s' % (str(round(roc_auc, 5))), end = 100*' '+'\n')
            
            if i == 0:
                toxic_score = f1_score_
        
    return toxic_score, preds