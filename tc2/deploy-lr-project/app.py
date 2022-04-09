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
    print(gender_zip,usage_zip,brands_zip)
    item = request.form['item']
    print(item)
    brand = int(request.form['brand'])
    print(brand)
    usage = int(request.form['usage'])
    print(usage)
    gender = int(request.form['gender'])
    print(usage)
    model_item = request.form['model']
    print(usage)
    size = int(request.form['size'])
    print(size)
    sale_until = int(request.form['sale_until'])
    print(size)
    prediction = model.predict([[brand, usage, gender, size, sale_until]])
    output = round(abs(prediction[0]), 2)
    return render_template('index1.html',
                           prediction_text=f'A house with {item} rooms per dwelling and {brand} and located {usage} km to {size} employment {sale_until}centers has a value of ${output}K')


if __name__ == "__main__":
    app.run()
