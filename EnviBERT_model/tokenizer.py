import os
import gc
import re
import string
import random
import time
import numpy as np
import pandas as pd
import transformers
import tokenizers
import torch.nn as nn

from nltk import sent_tokenize, download
from os.path import join as pjoin
from fairseq.data import Dictionary
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel

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



