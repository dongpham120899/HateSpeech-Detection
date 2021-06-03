import numpy as np

GLOVE_EMBEDDING_PATH = 'RNN_model/vectors_200d_v1.txt'

with open(GLOVE_EMBEDDING_PATH, 'r') as f:
    words_glove = {}
    for line in f:
        vals = line.rstrip().split(' ')
        words_glove[vals[0]] = float(vals[1])

def P_glove(word): 
    "Probability of `word`."
    # use inverse of rank as proxy
    # returns 0 if the word isn't in the dictionary
    return - words_glove.get(word, 0)
def correction_glove(word): 
    "Most probable spelling correction for word."
    return max(candidates_glove(word), key=P_glove)
def candidates_glove(word): 
    "Generate possible spelling corrections for word."
    return (known_glove([word]) or known_glove(edits1_glove(word)) or [word])
def known_glove(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in words_glove)
def edits1_glove(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

if __name__ == "__main__":
    print(P_glove("hello"))