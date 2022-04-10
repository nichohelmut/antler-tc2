import pickle

import pandas as pd
from flask import Flask, render_template, request

from ml_model.helper import simple_hot_encoding

app = Flask(__name__)
model = pickle.load(open('model2.pkl', 'rb'))


@app.route("/")
def hello():
    return render_template('index1.html')


def get_value_zips():
    df = pd.read_csv('ml_model/dataset/thelook2.csv')
    _, gender_zip, usage_zip, brands_zip = simple_hot_encoding(df)
    return gender_zip, usage_zip, brands_zip


@app.route("/predict", methods=['POST'])
def predict():
    gender_zip, usage_zip, brands_zip = get_value_zips()
    item = request.form['item']
    brand = int(request.form['brand'])
    usage = int(request.form['usage'])
    gender = int(request.form['gender'])
    size = int(request.form['size'])
    sale_until = int(request.form['sale_until'])
    prediction = model.predict([[brand, usage, gender, size, sale_until]])
    print(prediction)
    output = prediction[0]
    return render_template('index1.html',
                           prediction_text=f'Awesome! You could offer your {usage} {brand} shoe (category {item}) for ${round(output)}, if you sell within the next {sale_until} days!')


if __name__ == "__main__":
    app.run()
