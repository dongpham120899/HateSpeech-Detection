import numpy as np
import pandas as pd
from unidecode import unidecode
from processing.glove_helper import correction_glove

def preprocess(data):
    '''
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    '''
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    # Normalize bold word
    def Unidecode(text):
      out = []
      arr = text.split()
      for word in arr:
        if word.encode()[0] == 240:
          out.append(unidecode(word))
        else:
          out.append(word)

      return " ".join(out)

    #mapping short word to normal word
    def mapping(text):
      mapping = pd.read_csv("Datasets/Mapping HateSpeech_3.csv", header=None)
      origins = mapping[0].values
      mapped = mapping[1].values
      out = []
      for word in text.split():
        checked = False
        for i, ori in enumerate(origins):
          if word.lower() == ori:
            out.append(mapped[i])
            checked = True
            break
        if checked == False:
          out.append(word)

      return " ".join(out)

    # Correction word out of vocab to highest probability word in glove dic
    def correction_Glove(text):
      out = []
      for word in text.split():
        out.append(correction_glove(word))
      
      return " ".join(out)

    

    # data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
    data = data.astype(str).apply(lambda x: mapping(x))
    data = data.astype(str).apply(lambda x: correction_Glove(x))
    data = data.astype(str).apply(lambda x: Unidecode(x))
    return data

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in (f))

def build_matrix(word_index, path, EMBEDDING_DIM, max_features):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    unknown_words = []
    for key, i in word_index.items():
        word = key
        if i >= max_features:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = key.lower()
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = key.upper()
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = key.capitalize()
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        if len(key) > 1:
            word = correction_glove(key)
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                continue
        if embedding_vector is None:
            unknown_words.append(word)
        embedding_matrix[i] = embedding_index.get('trống')
    return embedding_matrix, unknown_words