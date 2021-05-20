# HateSpeech-Detection Vietnamese

This repository implements two approach to sovle HateSpeech Detection for Vietnamese

## Dependencies
* Python 3
* Pytorch (version 0.4.0)
* Transformers
* Fairseq
* Unidecode
* Torchtext
* GloVe vectors (Download glove.42B.300d.zip from https://nlp.stanford.edu/projects/glove/)

## Approach
* BERT (PhoBERT and EnviBERT)
* LSTM (GloVE Embeddings baseline and Compressing Word Embeddings follow work: https://github.com/nguyenphuhien13/Composition-Code-Learning-HateSpeech)
