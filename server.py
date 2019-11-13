from flask import Flask, request, jsonify
import json
import requests
import os

import numpy as np
import tensorflow as tf
from keras.models import load_model

# initialize our Flask application and the Keras model
app = Flask(__name__)
port = int(os.environ.get('PORT', 33507))
print(port)

def init():
    global model,graph
    # load the pre-trained Keras model
    model = load_model('paper_cnn_gru_drop02.h5')
    graph = tf.get_default_graph()

@app.route('/', methods=['POST'])
def index():
  print(port)
  data = json.loads(request.get_data().decode('utf-8'))

  # FETCH THE CRYPTO NAME
  crypto_name = data['nlp']['source']
  
        if crypto_name = 'hallo':
            prediction = 'Erkannt'


  # FETCH BTC/USD/EUR PRICES
  #r = requests.get("https://min-api.cryptocompare.com/data/price?fsym="+crypto_ticker+"&tsyms=BTC,USD,EUR")

      return jsonify(
        status=200,
        replies=[{
          'type': 'text',
          'content': 'Der Sentiment %s ' % (prediction)
        }]
      )

@app.route('/errors', methods=['POST'])
def errors():
  print(json.loads(request.get_data()))
  return jsonify(status=200)

app.run(port=port, host="0.0.0.0")
