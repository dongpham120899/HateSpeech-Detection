import numpy as np
import pandas as pd 
from unidecode import unidecode
from RNN_model.glove_helper import correction_glove

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
    mapping = pd.read_csv("RNN_model/Mapping HateSpeech_3.csv", header=None)
    origins = mapping[0].values
    mapped = mapping[1].values
    def mapping(text):
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
    # data = data.astype(str).apply(lambda x: mapping(x))
    # data = data.astype(str).apply(lambda x: correction_Glove(x))
    # data = data.astype(str).apply(lambda x: Unidecode(x))
    data = mapping(data)
    data = correction_Glove(data)

    return data