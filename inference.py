from flask import Flask, render_template, request,jsonify
import time
import json
import torch
import pandas as pd
import torch
import numpy as np
import os
from PhoBERT_model.models import *
from PhoBERT_model.utils import predict_phoBERT, predict_file_phoBERT
from EnviBERT_model.tokenizer import XLMRobertaTokenizer
from EnviBERT_model.utils import predict_enviBERT, predict_file_enviBERT
from RNN_model.models import *
from RNN_model.utils import predict_RNN, predict_file_RNN
from Norm_TTS import norm_sentence


list_results = []
request_model = 'enviBERT'
file_path = ''
   

if __name__ == '__main__':

   pred_target, probs = predict_file_enviBERT('../HateSpeech/data/spoken_test.csv')
   print(pred_target)  