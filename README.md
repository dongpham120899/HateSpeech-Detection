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
```
pip install -r requirements.txt
```
## Approach
* BERT 
  * PhoBERT (Dat Quoc Nguyen and Anh Tuan Nguyen VinAI follow work: https://github.com/VinAIResearch/PhoBERT)
  * EnviBERT (Binh Quoc Nguyen Vin Bigdata)
* LSTM
  * GloVE Embeddings baseline
  * Compressing Word Embeddings (follow work: https://github.com/nguyenphuhien13/Composition-Code-Learning-HateSpeech)
## How to use
To train model LSTM with baseline GloVe embedding, follow command:
```
python train.py -model_type="LSTM" -embedding_type="baseline" -weight_file="weights/LSTM_baseline_model.pt"
```
To train model LSTM with 64x16 encoding, follow work:
First, execute the following scripts in the repository: https://github.com/nguyenphuhien13/Composition-Code-Learning-HateSpeech .
After that, run the command:
```
python train.py -model_type="" -embedding_type="coded" -weight_file="weights/LSTM_64x16_model.pt"
```
To train model EnviBERT, follow command:
```
python train.py -model_type="enviBert" -max_sequence_length=256 -train_batch_size=24 -valid_batch_size=32 -learning_rate=5e-5 -epochs=6 -weight_file="weights/LSTM_baseline_model.pt" -weight_file="weights/enviBert_model.pt
```
To train model phoBert, follow command:
```
python train.py -model_type="phoBert" -max_sequence_length=256 -train_batch_size=24 -valid_batch_size=32 -learning_rate=5e-5 -epochs=6 -weight_file="weights/enviBert_model.pt
```
Pay attention to the selection of parameters suitable for the training model:
 * learning_rate 
 * train_batch_size
 * valid_batch_size
 * epochs

Choose the right option for training: 
 * text_type: choose text type, spoken_form_text or raw_text, default: raw text
 * data_train_type: choose data, full_data (train and test data) or only train_data, default: train data

If you need more help, run the command:
```
python train.py -h/--help
```
## Compare results
I did a comparison of the approaches and got the result on Toxicity label as below:

|      Model       |          F1_score (macro)     |
| ------------- | ------------- |
| Fine-tuning PhoBERT | 0.7868|
| Fine-tuning EnviBERT | 0.7878 |
| LSTM with baseline GloVe embedding | 0.8134 |
| LSTM with 64x16 encoding | 0.7664 |

In this task we met out of vocab problem, GloVe baseline gives best results partly because we retrained GloVe embedding on the large Vietnamese training set.
