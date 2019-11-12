from flask import Flask, request, jsonify
import json
import requests
import os
from keras.preprocessing.sequence import pad_sequences
import re
import string
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

#chat_path = r"data/chat.json"
model_path = 'paper_cnn_gru_drop02.h5'
tokenizer_path =  "tokenizer.pickle"
#path_chatbot = r'data/datensatz_chatbot.csv' # Excel vorher als UTF-8 abspeichern
path_stopwords = r"data/stopwords_chatbot.txt"

maxlen = 27

app = Flask(__name__)
port = int(os.environ.get('PORT', 33507))
print(port)

@app.route('/', methods=['POST'])

def index():
  print(port)
  data = json.loads(request.get_data().decode('utf-8'))

  # FETCH THE CRYPTO NAME
  chat_path = data['nlp']['source']
  

  # FETCH BTC/USD/EUR PRICES
  #r = requests.get("https://min-api.cryptocompare.com/data/price?fsym="+crypto_ticker+"&tsyms=BTC,USD,EUR")

    return jsonify(
      status=200,
      replies=[{
        'type': 'text',
        'content': 'Der Sentiment %s ' % (chat_path)
      }]
    )

@app.route('/errors', methods=['POST'])
def errors():
  print(json.loads(request.get_data()))
  return jsonify(status=200)

app.run(port=port, host="0.0.0.0")
