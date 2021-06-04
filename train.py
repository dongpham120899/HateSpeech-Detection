
import os
import gc
import re
import string
import random
import time
import numpy as np
import pandas as pd
import torch

from tqdm.notebook import tqdm
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaConfig, XLMRobertaModel
from transformers import AutoTokenizer, AutoConfig
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from processing.tokenizer import XLMRobertaTokenizer
from processing.utils import compute_output_arrays, read_txt, seed_all, read_csv
from processing.dataset import ToxicityDataset
from processing.preprocessing import preprocess, build_matrix
from torch.utils.data import DataLoader, Dataset
from models import BertBase, BertCNN, NeuralNet, Code_Learner

from transformers import AdamW, get_linear_schedule_with_warmup
from train_eval import train_model, val_model
import argparse

parser = argparse.ArgumentParser(description="Training model to HateSpeech Detection")
parser.add_argument("-max_sequence_length", type=int, default=400, metavar='N', help="Max sequence length of word, default: 400")
parser.add_argument("-train_batch_size", type=int, default=256, metavar='N', help="Batch size to training, default: 256")
parser.add_argument("-valid_batch_size", type=int, default=128, metavar='N', help="Batch size to validation, default: 128")
parser.add_argument("-accumulation_steps", type=float, default=1, metavar='N', help="accumulation steps to train, default: 1")
parser.add_argument("-num_labels", type=int, default=7, metavar='N', help="Number labels HateSpeech: Toxicity, Obscence, Threat, Identity attack - Insult, Sexual explicit, Sedition – Politics, Spam")
parser.add_argument("-learning_rate", type=float, default=0.003, metavar='N', help="Adam learning rate, default: 0.003")
parser.add_argument("-epochs", type=int, default=60, metavar='N', help="number of epochs, default: 60")
parser.add_argument("-model_type", default='LSTM', metavar='MODEL_TYPE', help="MODEL_TYPE, pick either phoBert or enviBert or LSTM, default: LSTM")
parser.add_argument("-glove_embedding_path", type=str, default='saved_LSTM_models/vectors_200d_v1.txt', metavar='PATH_FILE', help="Specific directory to weighted GloVe")
parser.add_argument("-embedding_dim", type=int, default=200, metavar='N', help="Embedding dimension size, default: 200")
parser.add_argument("-lstm_units", type=int, default=256, metavar='N', help="LSTM units, default: 256")
parser.add_argument("-embedding_type", default='baseline', metavar='EMBEDDING_TYPE', help="Embedding type; pick either coded or baseline, default: coded")
parser.add_argument('-model_file', default='models/64_8/epoch_84500.pt', metavar='PATH_FILE',help='specific directory to model you want to load')
parser.add_argument('-M', default=64, type=int, metavar='N', help='Number of source dictionaries, default: 64')
parser.add_argument('-K', default=16, type=int, metavar='N', help='Source dictionary size, default: 16')
parser.add_argument('-vocab_file', default='saved_LSTM_models/vocab_200d_v1.txt', metavar='PATH_FILE',help='file which contains GloVE vocab')
parser.add_argument('-data_folder', default='Datasets/', metavar='DIR', help='folder to retrieve embeddings, data, text files, etc.')
parser.add_argument('-text_type', default='raw_text', metavar="TEXT_TYPE", help='choose text type, spoken_form_text or raw_text, default: raw text')
parser.add_argument('-data_train_type', default='train_data', metavar="DATA_TYPE", help='choose data, full_data (train and test data) or only train_data, default: train data')
parser.add_argument('-weight_file', default='weight/spoken_form_LSTM_model_v2.pt', metavar='PATH_FILE', help='choose path to save weight model')
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
labels = ['Toxicity','Obscence','Threat','Identity attack - Insult','Sexual explicit','Sedition – Politics','Spam']
comments = 'comments'

def train_classifier(model ,epochs, lr, train_loader, valid_loader, model_type):
    model.train()

    i = 0
    best_val_score   = 0.0
    best_param_score = None

    if model_type != "LSTM":
        optimizer = AdamW(model.parameters(), lr=lr, eps=4e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05, num_training_steps= epochs*len(train_loader)//args.accumulation_steps)

    for epoch in (range(epochs)):

        if model_type == "LSTM":
            param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
            optimizer = torch.optim.Adam(param_lrs, lr=lr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.9 ** epoch)
        
        torch.cuda.empty_cache()
    
        start_time   = time.time()
        avg_loss, model  = train_model(model, train_loader, optimizer, scheduler, model_type)
        val_score, preds = val_model(model, valid_loader, val_shape=len(test), batch_size = args.valid_batch_size, model_type=model_type)
        elapsed_time = time.time() - start_time

        print('Epoch {}/{} \t loss={:.4f} \t val_score={:.4f} \t time={:.2f}s'.format(
          epoch + 1, args.epochs, avg_loss, val_score, elapsed_time))

        if val_score > best_val_score:
            print('Saving new model !!')
            i = 0
            best_val_score = val_score 
            best_param_score = model.state_dict()
        else:
            i += 1

    model.load_state_dict(best_param_score)

    return model, best_val_score

def loading_model():
    pass
if __name__ == '__main__':
    seed_all(123)
    path = 'Datasets/'
    train, test = read_csv(path)
    if args.data_train_type == "full_data":
        train = pd.concat((train, test))
    
    
    train = train.reset_index(drop = True)
    test = test.reset_index(drop = True)

    train_outputs = compute_output_arrays(train, columns = labels)
    test_outputs = compute_output_arrays(test, columns = labels)

    train_df = train.reset_index(drop=True)
    valid_df = test.reset_index(drop=True)

    # choose text type
    if args.text_type == "spoken_form_text":
        x_train = preprocess(train[comments])
        x_test = preprocess(test[comments])
    else:
        x_train = train[comments]
        x_test = test[comments]

    # choose model type
    if args.model_type == 'enviBert':
        model_name = 'enviBERT/'
        tokenizer = XLMRobertaTokenizer(model_name)

        # Initializing a RoBERTa configuration
        configuration = XLMRobertaConfig()
        model = XLMRobertaModel(configuration)
        configuration = model.config

        configuration.output_hidden_states = True
        configuration.vocab_size = tokenizer.vocab_size
        configuration.num_labels = args.num_labels

    elif args.model_type == 'phoBert':
        model_name = 'vinai/phobert-base'
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fact=False)

        # Initializing a RoBERTa configuration  
        configuration = AutoConfig.from_pretrained(model_name)
        configuration.output_hidden_states = True
        configuration.num_labels = args.num_labels

    elif args.model_type == "LSTM":
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(list(x_train) + list(x_test))

        x_train = tokenizer.texts_to_sequences(x_train)
        x_test = tokenizer.texts_to_sequences(x_test)
        x_train = pad_sequences(x_train, maxlen=args.max_sequence_length)
        x_test = pad_sequences(x_test, maxlen=args.max_sequence_length)

    train_set    = ToxicityDataset(x_train, tokenizer, args.max_sequence_length, args.model_type, train_outputs)
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True)
        
    valid_set    = ToxicityDataset(x_test, tokenizer, args.max_sequence_length, args.model_type, test_outputs)
    valid_loader = DataLoader(valid_set, batch_size=args.valid_batch_size, shuffle=False, drop_last=False)
    
    if args.model_type == "LSTM":
        max_features = None
        max_features = max_features or len(tokenizer.word_index) + 1

        if args.embedding_type == "baseline":
            # Loading weight embedding Glove
            glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, args.glove_embedding_path, args.embedding_dim, max_features)
            print('n unknown words (glove): ', len(unknown_words_glove))

            model = NeuralNet(torch.FloatTensor(glove_matrix).cpu(), args.lstm_units)
            model.zero_grad()
            model.to(device)
        elif args.embedding_type == "coded":
            # Load GloVE embeddings
            orig_embeddings = torch.load(args.data_folder + 'all_orig_emb.pt')
            # Load shared words and all GloVE words
            with open(args.data_folder + "shared_words.txt", "r") as file:
                shared_words = file.read().split('\n')
            with open(args.data_folder + "glove_words.txt", "r") as file:
                glove_words = file.read().split('\n')
            # Recreate GloVE_dict
            glove_dict = {}
            for i, word in enumerate(glove_words):
                glove_dict[word] = orig_embeddings[i]

            # Read GloVe vocab
            vocab_dict = {}
            with open(args.vocab_file) as file:
            # For every line, the first part is the word, and the rest is the vector.
                for i, line in enumerate(file):
                    entry = line.split()
                    word = entry[0]
                    embedding = np.array(entry[1:], dtype='int32')
                    # Add word -> embedding pair to dict
                    vocab_dict[word] = embedding


            # Initialize embedding
            code_embedding = torch.FloatTensor(np.random.uniform(-0.25, 0.25, (max_features, args.embedding_dim)))
            # load best model for code embedding generation
            code_learner = Code_Learner(args.embedding_dim, args.M, args.K)
            code_learner = torch.load(args.model_file, map_location='cpu')
        
            #Put model into CUDA memory if using GPU
            # code_embedding = code_embedding.to(device)
            # code_learner = code_learner.to(device)

            #For all words in vocab
            for i in range(max_features):
                # Try to see if it has a corresponding glove_vector
                try:
                    glove_vec = glove_dict[vocab_dict[i]]
                    glove_vec = glove_vec.to(device)
                    # If so, then generate our own embedding for the word using our model
                    code_embedding[i] = code_learner(glove_vec, training=False)
                # The word doesn't have a GloVE vector, keep it randomly initialized
                except KeyError:
                    pass
            model = NeuralNet(torch.FloatTensor(code_embedding).cpu(), args.lstm_units)
            model = model.to(device)

    elif args.model_type == "enviBert":
        model = BertBase.from_pretrained(model_name, config=configuration, model_name=model_name)
        model.zero_grad()
        model.to(device)
        torch.cuda.empty_cache()
    elif args.model_type == "phoBert":
        model = BertCNN.from_pretrained(model_name, config=configuration, model_name=model_name)
        model.zero_grad()
        model.to(device)
        torch.cuda.empty_cache()


    model, best_val_score = train_classifier(model, args.epochs, args.learning_rate, train_loader, valid_loader, args.model_type)

    print(best_val_score)

    ## save weight model
    torch.save(model, args.weight_file)
    

