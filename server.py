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

chat_path = r"chat.json"
model_path = 'paper_cnn_gru_drop02.h5'
tokenizer_path =  "tokenizer.pickle"
#path_chatbot = r'data/datensatz_chatbot.csv' # Excel vorher als UTF-8 abspeichern
path_stopwords = r"stopwords_chatbot.txt"

maxlen = 27


app = Flask(__name__)
port = int(os.environ.get('PORT', 33507))
print(port)

@app.route('/', methods=['POST'])

def index():
  print(port)
  data = json.loads(request.get_data().decode('utf-8'))
 # txt = "das ist blöd"

  # FETCH THE CRYPTO NAME
  #chat_path = data['nlp']['source']  

def load_json():
    with open(chat_path) as json_file:
        data = json.load(json_file)
        for p in data['chat']:
           # print('Time: ' + p['time'])
           # print('Text: ' + p['text'])
           # print('')
            txt = p['text']            
    return txt

    
def preprocess_text(sen):
    
    sentence = sen
    # Entfernt Punktuationen, behält Bindestriche ! 
    
    remove = string.punctuation
    remove = remove.replace("-", "") # keine Bindestriche entfernen
    pattern = r"[{}]".format(remove) # Pattern erstellen 
    sentence = re.sub(pattern, "", sentence) 
    
    # Entfernt Zahlen
    sentence = re.sub('[0-9]', '', sentence)

    # Entfernt einzelne Buchstaben
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Entfernt mehrere Spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    
    # Wandelt Großbuchstaben in Kleinbuchstaben um 
    sentence = sentence.lower()
    
    return sentence

def load_stopwords():
    stopwordslist = open(path_stopwords, "r", encoding = "utf-8")
    stop = stopwordslist.read()
    stop = stop.split(', ')
    #print(stop)
    return stop 

def load_models():
    loaded_model = tf.keras.models.load_model(model_path)

    with open(tokenizer_path, 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
        
    return loaded_model, loaded_tokenizer

def do_prediction(txt, loaded_model, loaded_tokenizer, stop): 
    preprocessed = preprocess_text(txt)
    without_stop = ([word for word in preprocessed.split() if word not in (stop)])
    sentences_token = loaded_tokenizer.texts_to_sequences([without_stop])
    sentences_pad = pad_sequences(sentences_token, padding='post', maxlen=maxlen)
    sentences_pad
    predi = loaded_model.predict(sentences_pad)
    labels = ['negativ', 'neutral', 'positiv']
    #print(predi, labels[np.argmax(predi)])
    sentiment = labels[np.argmax(predi)]
    return sentiment

#def save_json(txt):
 #   json_output = {
  #      'model': model_path,
   #     'text': txt,
    #    'sentiment': sentiment
    #}

    #with open('sentiment.json', 'w', encoding="utf-8") as json_file:
     #   json.dump(json_output, json_file, ensure_ascii=False)
        
        
txt = load_json()
stop = load_stopwords()
loaded_model, loaded_tokenizer = load_models()
sentiment = do_prediction(txt, loaded_model, loaded_tokenizer, stop)
save_json(txt)
print("Der Sentimentwert wurde in sentiment.json abgespeichert")
    
  

  # FETCH BTC/USD/EUR PRICES
  #r = requests.get("https://min-api.cryptocompare.com/data/price?fsym="+crypto_ticker+"&tsyms=BTC,USD,EUR")

return jsonify(
  status=200,
  replies=[{
    'type': 'text',
    'content': 'Der Sentiment %s ' % (txt)
  }]   )

@app.route('/errors', methods=['POST'])
def errors():
  print(json.loads(request.get_data()))
  return jsonify(status=200)

app.run(port=port, host="0.0.0.0")
