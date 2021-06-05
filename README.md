# HateSpeech-Detection Web API
This repository implements a web api using Flask to detect hatespeech.

**Input**: Text

**Output**: Toxicity, Obscence, Threat, Identity attack-Insult, Sexual-explicit, Sedition-Politics, Spam score
## How to use
To run our Web API, follow command:
```
python app.py
```
You can send a request with the following Python code, then the system will return the result 
```
url = "http://10.124.67.54:5000/result?format=json"
requests.post(url, data={'result':'xin chao me m', 'model':'LSTM'})
```
* url: url address
* result: text input
* model: :LSTM/enviBert/phoBert
