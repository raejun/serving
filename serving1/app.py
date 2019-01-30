import sys
import json
import numpy as np
from sklearn import datasets
import tensorflow as tf
from keras.models import load_model
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

global model
global graph
global target_names
model = load_model('iris_model.h5') # 저장된 모델 로딩
graph = tf.get_default_graph()
# species 이름 로딩 예) ['setosa' 'versicolor' 'virginica']
target_names = datasets.load_iris().target_names

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/checkform')
def checkform():
    return render_template('checkform.html')

@app.route('/check', methods=['POST'])
def check():
    sepal_length = request.form['sepal_length']
    sepal_width = request.form['sepal_width']
    petal_length = request.form['petal_length']
    petal_width = request.form['petal_width']
    features = [sepal_length, sepal_width, petal_length, petal_width]
    features = np.reshape(features, (1, 4))
    with graph.as_default():
        Y_pred = model.predict_classes(features)
    print({'species': target_names[Y_pred[0]]})
    return jsonify( {'species': target_names[Y_pred[0]]} )

if __name__ == '__main__':
    app.run(debug=True) # 로컬 테스트용 (http://localhost:5000)
    #app.run(host='0.0.0.0', debug=True) # 외부 접속용
