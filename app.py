from flask import Flask, render_template, request,jsonify
import time
import json
import torch
import pandas as pd
import torch
import numpy as np
from PhoBERT_model.models import *
from PhoBERT_model.utils import *
from EnviBERT_model.tokenizer import XLMRobertaTokenizer
from EnviBERT_model.utils import predict_enviBERT
from RNN_model.models import *
from RNN_model.utils import predict_RNN
from Norm_TTS import norm_sentence
app = Flask(__name__)

list_results = []

@app.route("/")
def input():
   return render_template("index.html")

@app.route('/result', methods = ['POST'])
def result():
   t1 = time.time()
   sentence = str(request.form['result'])
   request_model = request.form['model']
   t2 = time.time()
   args = request.args.get('format', default="json")
   print("alo", args)

   print("MODEL:\t", request_model)
   if request_model == 'phoBERT':
      t3 = time.time()
      result = predict_phoBERT(sentence)
      t4 = time.time()
      print("Time Infer PhoBERT:", t4-t3)
   elif request_model == 'enviBERT':
      t5 = time.time()
      result = predict_enviBERT(sentence)
      t6 = time.time()
      print("Time Infer EnviBERT:", t6-t5)
   elif request_model == "LSTM":
      result = predict_RNN(sentence)

   print("*********************************************************")
   
   t3 = time.time()
   print("Time Predict:\t", t3-t2)    
   # output = {"Sentence":sentence,
   #          "time": round(t3-t1,4),
   #          "model": request_model,
   #          "Toxicity": round(float(result[0]), 4),
   #          "Obscence": round(float(result[1]), 4),
   #          "Threat": round(float(result[2]), 4),
   #          "Identity attack-Insult": round(float(result[3]), 4),
   #          "Sexual-explicit": round(float(result[4]), 4),
   #          "Sedition-Politics": round(float(result[5]), 4),
   #          "Spam": round(float(result[6]), 4)}
   
   # list_results.append(output)


   mapping = {1:"Obscence",2:"Threat",3:"Identity attack-Insult",4:"Sexual-explicit",5:"Sedition-Politics",6:"Spam"}
   
   output = {}
   result_list = list(result)

   if result[0] > 0.5:
      max_indice = result_list.index(max(result_list[1:]))
      output['Category'] = mapping[max_indice]
      output['Toxicity'] = 1
   else:
      output['Toxicity'] = 0
   list_results.append(output)

   if args == 'json':
      return jsonify(output)
   else:
      return render_template("result.html", results=list_results)
   

if __name__ == '__main__':
   app.run(host="0.0.0.0",debug = True)    