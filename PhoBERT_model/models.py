import numpy as numpy
import pandas as pd 
from transformers import BertPreTrainedModel
import torch
import torch.nn as nn 
import torch.nn.functional as F

########## Model Bert+CNN+label attention ######
# get 4 hidden state in transformer + CNN, concat label attention in after CNN
class BertCNN(BertPreTrainedModel):
    def __init__(self, conf, MODEL_NAME, drop_out, lb_availible=False, max_len=256, max_len_tfidf=0):
        super(BertCNN, self).__init__(conf)
        self.config = conf
        self.lb_availible = lb_availible
        self.num_labels = conf.num_labels
        self.backbone = RobertaModel.from_pretrained(MODEL_NAME, config=self.config)

        self.convs = nn.ModuleList([nn.Conv1d(max_len+max_len_tfidf, 256, kernel_size) for kernel_size in [3,5,7]])
        self.dropout = nn.Dropout(drop_out)
        self.out = nn.Linear(self.config.hidden_size, self.num_labels)
        

    def forward(self, input_ids, attention_mask, token_type_ids):
        sequence_output, _, hidden_outputs = self.backbone(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            return_dict = False
        )
        sequence_output = torch.stack([hidden_outputs[-1], hidden_outputs[-2], hidden_outputs[-3]])
        sequence_output = torch.mean(sequence_output, dim=0)

        cnn = [F.relu(conv(sequence_output)) for conv in self.convs]
        max_pooling = []
        for i in cnn:
          max, _ = torch.max(i, 2)
          max_pooling.append(max)
        output = torch.cat(max_pooling, 1)
        
        output = self.dropout(output)
        logits = self.out(output)

        return torch.sigmoid(logits)
