# HateSpeech-Detection Vietnamese

This repository implements two approaches to sovle HateSpeech Detection tasks.

## Dependencies
* Python 3
* Pytorch (version 0.4.0)
* Transformers
* Fairseq
* Unidecode
* Torchtext
* GloVe vectors (Download glove.42B.300d.zip from https://nlp.stanford.edu/projects/glove/)

## Approach
* BERT 
  * PhoBERT (Dat Quoc Nguyen and Anh Tuan Nguyen)
  * EnviBERT (Binh Quoc Nguyen)
* LSTM
  * GloVE Embeddings baseline
  * Compressing Word Embeddings follow work: https://github.com/nguyenphuhien13/Composition-Code-Learning-HateSpeech)

## Compare results
I did a comparison of the approaches and got the result on Toxicity label as below:

|      Model       |          Accuracy     |
| ------------- | ------------- |
| Classifier with fine-tuning PhoBERT | 0.853|
| Classifier with fine-tuning EnviBERT | 0.853|
| Classifier with baseline GloVe embedding | 0.853|
| Classifier with 64x16 encoding | 0.841|

The problem of this task is out of vocab, GloVe baseline gives best results partly because we retrained GloVe embedding on the large Vietnamese training set.
