from flask import Flask, request, jsonify
import json
import requests
import os

app = Flask(__name__)
port = int(os.environ.get('PORT', 33507))
print(port)

@app.route('/', methods=['POST'])
def index():
  print(port)
  data = json.loads(request.get_data().decode('utf-8'))

  # FETCH THE CRYPTO NAME
  crypto_name = data['conversation']['memory']
  crypto_ticker = crypto_name.upper()

  # FETCH BTC/USD/EUR PRICES
  #r = requests.get("https://min-api.cryptocompare.com/data/price?fsym="+crypto_ticker+"&tsyms=BTC,USD,EUR")

  return jsonify(
    status=200,
    replies=[{
      'type': 'text',
      'content': 'Der Sentiment %s ' % (crypto_ticker)
    }]
  )

@app.route('/errors', methods=['POST'])
def errors():
  print(json.loads(request.get_data()))
  return jsonify(status=200)

app.run(port=port, host="0.0.0.0")
