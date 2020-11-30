import flask
from flask import Flask, jsonify, request
import json
from data_input import data_in
import numpy as np
import tensorflow as tf
from tensorflow import keras
path = 'C:/Users/Xkfal/Documents/nyc_realestate_proj/FlaskAPI/models'

app = Flask(__name__)


def load_models():
    model = tf.keras.models.load_model(path + '/model')
    return model

@app.route('/predict', methods=['GET'])  
def predict():
    # stub input features
    x = np.array(data_in).reshape(1,-1)
    # load model
    model = load_models()
    print('CHECK HERE', x.shape)
    prediction = model.predict(x)[0][0]
    response = json.dumps({'response': np.expm1(prediction).item()})
    return response, 200

if __name__ == '__main__':
    application.run(debug=True)
