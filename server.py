from flask import Flask, request, jsonify
from keras.preprocessing.sequence import pad_sequences
import re
import string
import numpy as np
import tensorflow as tf
import pickle
import json
import requests
import os



app = Flask(__name__)
port = int(os.environ.get('PORT', 33507))


chat_path = r"chat.json"
model_path = 'paper_cnn_gru_drop02.h5'
tokenizer_path =  "tokenizer.pickle"
path_stopwords = r"stopwords_chatbot.txt"

maxlen = 27


def preprocess_text(sen):
    sentence = sen
    # Entfernt Punktuationen, behält Bindestriche !

    remove = string.punctuation
    remove = remove.replace("-", "")  # keine Bindestriche entfernen
    pattern = r"[{}]".format(remove)  # Pattern erstellen
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
    sentiment = labels[np.argmax(predi)]
    return sentiment


def save_json(txt):
    json_output = {
        'model': model_path,
        'text': txt,
        'sentiment': sentiment
    }

    with open('data/sentiment.json', 'w', encoding="utf-8") as json_file:
        json.dump(json_output, json_file, ensure_ascii=False)



@app.route('/', methods=['POST'])
def bot():
    data = json.loads(request.get_data())

    bot_chat = data['nlp']['source']
    print(type(bot_chat))

    stop = load_stopwords()
    loaded_model, loaded_tokenizer = load_models()
    sentiment = do_prediction(bot_chat, loaded_model, loaded_tokenizer, stop)
    output_Text = sentiment
    print(bot_chat)

    return jsonify(
        status=200,
        replies=[{
            'type': 'text',
            'content': output_Text
        }]
    )




if __name__ == '__main__':
	app.run(host="0.0.0.0", port=environ.get('PORT'))