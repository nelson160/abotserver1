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


app = Flask(__name__)
port = int(os.environ.get('PORT', 33507))
print(port)

#chat_path = r"data/chat.json"
model_path = 'paper_cnn_gru_drop02.h5'
tokenizer_path =  "tokenizer.pickle"
#path_chatbot = r'data/datensatz_chatbot.csv' # Excel vorher als UTF-8 abspeichern
path_stopwords = r"data/stopwords_chatbot.txt"

maxlen = 27


@app.route('/', methods=['POST'])
def index():
  print(port)
  data = json.loads(request.get_data().decode('utf-8'))

  # FETCH THE CRYPTO NAME
  chat_path = data['nlp']['source']
  
  loaded_model = tf.keras.models.load_model(model_path)

    with open(tokenizer_path, 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
   
    sentences_token = loaded_tokenizer.texts_to_sequences([chat_path])
    sentences_pad = pad_sequences(sentences_token, padding='post', maxlen=maxlen)
    sentences_pad
    predi = loaded_model.predict(sentences_pad)
    labels = ['negativ', 'neutral', 'positiv']
    #print(predi, labels[np.argmax(predi)])
    sentiment = labels[np.argmax(predi)]



  # FETCH BTC/USD/EUR PRICES
  #r = requests.get("https://min-api.cryptocompare.com/data/price?fsym="+crypto_ticker+"&tsyms=BTC,USD,EUR")

    return jsonify(
      status=200,
      replies=[{
        'type': 'text',
        'content': 'Der Sentiment %s ' % (sentiment)
      }]
    )

@app.route('/errors', methods=['POST'])
def errors():
  print(json.loads(request.get_data()))
  return jsonify(status=200)

app.run(port=port, host="0.0.0.0")
