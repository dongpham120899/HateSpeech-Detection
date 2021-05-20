import torchtext
import torch
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Create GloVE and IMDB embeddings')
# File name for GloVE vectors
parser.add_argument('--glove_file', default='saved_LSTM_models/vectors_200d_v1.txt', metavar='file',
                    help='file which contains GloVE embeddings')
# Directory where we want to write everything we save in this script to
parser.add_argument('--data_folder', default='saved_LSTM_models/', metavar='DIR',
                    help='folder to save embeddings, data, text files, etc.')
parser.add_argument('--train_file', default='Datasets/train_v7.csv', metavar='DIR',
                    help='file to train')
parser.add_argument('--test_file', default='Datasets/test_v7.csv', metavar='DIR',
                    help='file to test')

def main():
    global args
    # Parse commands from ArgumentParser
    args = parser.parse_args()

    train = pd.read_csv(args.train_file) 
    test = pd.read_csv(args.test_file) 
    
    word_list = set()
    train['comments'].str.split().apply(word_list.update)
    word_list = list(word_list)

    # Next, construct a dictionary that maps words to their GloVe vectors
    glove_dict = {}
    # There are this many words included in the file
    total_glove_num = 1355383 + 1
    # We also want to store all the embeddings to a file, which we can't do from a dict
    all_orig_embeddings = torch.zeros(total_glove_num, 200, dtype=torch.float)
    # We also want a list of all glove_words, because it is handy
    glove_words = []
    # Reading previously specified file
    with open(args.glove_file) as file:
        # For every line, the first part is the word, and the rest is the vector.
        for i, line in enumerate(file):
            entry = line.split()
            word = entry[0]
            embedding = np.array(entry[1:], dtype='float32')
            # Add word to our running list
            glove_words.append(word)
            # Add word -> embedding pair to dict
            glove_dict[word] = embedding
            # Also to our FloatTensor for all GloVe embeddings
            all_orig_embeddings[i] = torch.FloatTensor(embedding)
    print('GloVe dict constructed')

    # Now we make a list of words that appear in both the IMDB dataset and the GloVE file
    shared_words = []
    for word in word_list:
        if word in glove_dict:
            shared_words.append(word)
    print('Shared words list constructed.')
    # We write our shared_word list to a text file for easy reference
    with open(args.data_folder + 'shared_words.txt', 'w') as out_file:
        out_file.write('\n'.join(shared_words))

    # We write our glove_word list to a text file for easy reference
    with open(args.data_folder + 'glove_words.txt', 'w') as out_file:
        out_file.write('\n'.join(glove_words))

    # We save our glove_embedding for later use
    torch.save(all_orig_embeddings, args.data_folder + 'all_orig_emb.pt')

if __name__ == '__main__':
    main()