import torch
import torch.nn as nn

from os.path import join as pjoin
from fairseq.data import Dictionary
from processing.utils import *

CLS_token_id = 0
SEP_token_id = 2
PAD_token_id = 1

class SentencepieceBPE(object):
    def __init__(self, model_file):
        sentencepiece_model = model_file
        try:
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(sentencepiece_model)
        except ImportError:
            raise ImportError('Please install sentencepiece with: pip install sentencepiece')

    def encode(self, x: str) -> str:
        return ' '.join(self.sp.EncodeAsPieces(x))

    def decode(self, x: str) -> str:
        return x.replace(' ', '').replace('\u2581', ' ').strip()

    def is_beginning_of_word(self, x: str) -> bool:
        if x in ['<unk>', '<s>', '</s>', '<pad>']:
            # special elements are always considered beginnings
            # HACK: this logic is already present in fairseq/tasks/masked_lm.py
            # but these special tokens are also contained in the sentencepiece
            # vocabulary which causes duplicate special tokens. This hack makes
            # sure that they are all taken into account.
            return True
        return x.startswith('\u2581')

class XLMRobertaTokenizer:
    def __init__(self, pretrained_file):
        # load bpe model and vocab file
        bpe_model_file = pjoin(pretrained_file, 'sentencepiece.bpe.model')
        vocab_file = pjoin(pretrained_file, 'dict.txt')
        self.sp = SentencepieceBPE(bpe_model_file)
        self.bpe_dict = Dictionary().load(vocab_file)
        self.cls_token = "<s>"
        self.sep_token = "</s>"
        self.pad_token_id = 1
    
    def tokenize(self, sentence):
        return self.sp.encode(sentence).split(' ')

    def convert_tokens_to_ids(self, tokens):
        bpe_sentence = ' '.join(tokens)
        bpe_ids = self.bpe_dict.encode_line(bpe_sentence, add_if_not_exist=False,
                                            append_eos=False).tolist()
                                    


    # def encode(self, sentence, add_bos=False, add_eos=False):
    #     bpe_sentence = '<s> ' + self.sp.encode(sentence) + ' </s>'
    #     bpe_ids = self.bpe_dict.encode_line(bpe_sentence, append_eos=False).tolist()
    #     if not add_bos:
    #         bpe_ids = bpe_ids[1:]
    #     if not add_eos:
    #         bpe_ids = bpe_ids[:-1]
        return bpe_ids
        
    def decode(self, tokens):
        sentences = [self.sp.decode(self.bpe_dict.string(s)) for s in tokens]
        return sentences
    
    def encodeAsPieces(self, sentence):
        bpe_sentence = '<s> ' + self.sp.encode(sentence) + ' </s>'
        return bpe_sentence

    @property
    def vocab_size(self):
        return len(self.bpe_dict)



class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # # print(attention_mask)
        # x = torch.mean(features, dim=1, keepdims=True).squeeze()
        # with torch.no_grad():
        #     sum_mask_embeddings = mask_embeddings.sum(dim=1) 
        #     for dim in range(mask.size(0)):
        #         average_mask_embeddings[dim] = sum_mask_embeddings[dim] / mask[dim].sum()
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


# class XLMRForDialogueAct(RobertaPreTrainedModel):
#     authorized_missing_keys = [r"position_ids"]

#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels

#         self.roberta = RobertaModel(config, add_pooling_layer=False)
#         self.classifier = RobertaClassificationHead(config)

#         self.init_weights()

#     def average_features(self, features, attention_mask):
#         # print(attention_mask.size())
#         # attention_mask = attention_mask.unsqueeze(2)
#         # print(attention_mask[:, :10])
#         # print(features[:, :10, :10])
#         with torch.no_grad():
#             # print(attention_mask.size())
#             x_sum = torch.sum(attention_mask, dim=1, keepdims=True) #[batch, 1]
#             # print(x_sum)
#             # print(x_sum.size())
#             # attention_mask = attention_mask.repeat(1, features.size()[-1], 1).transpose(1, 2)
#             attention_mask = attention_mask.repeat(features.size()[-1], 1, 1).transpose(0, 1).transpose(1, 2) #[batch, seq_len, hidden_dim]
#             # print(attention_mask.size())
            
#             # x.repeat(1, 4, 1).transpose(1,2)
#             # print(attention_mask.size())
#             x = torch.mul(features, attention_mask) #[batch, seq_len, hidden_dim]
#             # print(x.size())
#             # print(x[:, :10, :10])
            
#             x = torch.sum(x, dim=1, keepdims=True).squeeze()/x_sum #[batch, hidden_dim]
#             # print(x.size())
#             # print(x[:, :10])
#             # print(x.size())
#         # attention_sum = torch.sum(attention_mask, dim=1, keepdims=True)
#         # print(attention_sum.size())

#         #x = torch.mean(features, dim=1, keepdims=True).squeeze()
    
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
#             Labels for computing the sequence classification/regression loss.
#             Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
#             If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
#             If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.roberta(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         sequence_output = outputs[0]
#         # print(type(sequence_output))
#         # print(sequence_output.size())
#         # print(attention_mask)
#         # x = self.average_features(sequence_output, attention_mask)
#         logits = self.classifier(sequence_output)
#         # print(logits.size())
#         # print(labels.size())

#         loss = None
#         if labels is not None:
#             if self.num_labels == 1:
#                 #  We are doing regression
#                 loss_fct = MSELoss()
#                 loss = loss_fct(logits.view(-1), labels.view(-1))
#             else:
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))

#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )