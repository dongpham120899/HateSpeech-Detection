import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel


# model_name = '../HateSpeech/envibert/'


class BertBase(RobertaPreTrainedModel):
    def __init__(self, conf):
        super(BertBase, self).__init__(conf)
        self.config = conf
        self.backbone = XLMRobertaModel.from_pretrained(
            model_name, config=self.config)
        self.lstm_units = 768
        self.num_recurrent_layers = 1
        self.bidirectional = False

        self.lstm = nn.LSTM(input_size=self.config.hidden_size,
                            hidden_size=self.lstm_units,
                            num_layers=self.num_recurrent_layers,
                            bidirectional=self.bidirectional,
                            batch_first=True)

        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(self.config.hidden_size*2, 7)

    def forward(self, input_ids, attention_mask, token_type_ids):

        # with autocast():
        sequence_output, _, hidden_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )

            # sequence_output = torch.stack(hidden_outputs[-2:]).mean(0)
            # sequence_output = torch.cat(tuple([hidden_outputs[i] for i in [-1, -2]]), dim=-1)

        if self.bidirectional:
            n = 2
        else: n = 1

        h0 = Variable(torch.zeros(self.num_recurrent_layers * n,       # (L * 2 OR L, B, H)
                                      input_ids.shape[0],
                                      self.lstm_units))
        c0 = Variable(torch.zeros(self.num_recurrent_layers * n,        # (L * 2 OR L, B, H)
                                      input_ids.shape[0],
                                      self.lstm_units))

        sequence_output, _ = self.lstm(sequence_output, (h0, c0))

        avg_pool = torch.mean(sequence_output, 1)
        max_pool, _ = torch.max(sequence_output, 1)
        h_conc = torch.cat((max_pool, avg_pool), 1)
        output = self.dropout(h_conc)
        logits = self.out(output)

        return logits
