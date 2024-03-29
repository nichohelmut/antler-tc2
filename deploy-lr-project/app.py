import os
import pickle

import pandas as pd
from flask import Flask, render_template, request
from google.cloud import storage

from helper import simple_hot_encoding

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "footy-dev-343115-1ab77dbb8af4.json"
app = Flask(__name__)

storage_client = storage.Client()

bucket = storage_client.bucket('pricing_model_nhu')
blob = bucket.blob('model2.pkl')
pickle_in = blob.download_as_string()
model = pickle.loads(pickle_in)


@app.route("/")
def hello():
    gender_zip, usage_zip, brands_zip = get_value_zips()
    return render_template('index1.html',
                           gender=gender_zip, usage=usage_zip, brands=brands_zip,
                           scroll="")


def get_value_zips():
    df = pd.read_csv('dataset/thelook2.csv')
    _, gender_zip, usage_zip, brands_zip = simple_hot_encoding(df)
    return gender_zip, usage_zip, brands_zip


def inverse_dicts(usage_zip: dict, brands_zip: dict):
    usage_map = {v: k for k, v in usage_zip.items()}
    brands_map = {v: k for k, v in brands_zip.items()}
    return usage_map, brands_map


@app.route("/predict", methods=['POST'])
def predict():
    gender_zip, usage_zip, brands_zip = get_value_zips()
    usage_map, brands_map = inverse_dicts(usage_zip, brands_zip)
    item = request.form['item']
    brand = int(request.form['brand'])
    usage = int(request.form['usage'])
    gender = int(request.form['gender'])
    size = int(request.form['size'])
    sale_until = int(request.form['sale_until'])
    prediction = model.predict([[brand, usage, gender, size, sale_until]])
    output = prediction[0]
    return render_template('index1.html',
                           prediction_text=f'Awesome! You could offer your {usage_map[usage]} {brands_map[brand]} shoe (category {item}) for ${round(output * 1.2)}, if you sell within the next {sale_until} days!',
                           gender=gender_zip, usage=usage_zip, brands=brands_zip, scroll="prediction")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
