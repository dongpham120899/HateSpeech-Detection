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
  * PhoBERT (Dat Quoc Nguyen and Anh Tuan Nguyen VinAI follow work: https://github.com/VinAIResearch/PhoBERT)
  * EnviBERT (Binh Quoc Nguyen Vin Bigdata)
* LSTM
  * GloVE Embeddings baseline
  * Compressing Word Embeddings (follow work: https://github.com/nguyenphuhien13/Composition-Code-Learning-HateSpeech)

## Compare results
I did a comparison of the approaches and got the result on Toxicity label as below:

|      Model       |          F1_score (macro)     |
| ------------- | ------------- |
| Fine-tuning PhoBERT | 0.7868|
| Fine-tuning EnviBERT | 0.7878 |
| LSTM with baseline GloVe embedding | 0.8134 |
| LSTM with 64x16 encoding | 0.7664 |

In this task we met out of vocab problem, GloVe baseline gives best results partly because we retrained GloVe embedding on the large Vietnamese training set.
