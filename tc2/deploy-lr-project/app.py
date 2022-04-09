from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('model2.pkl', 'rb'))


@app.route("/")
def hello():
    return render_template('index1.html')


def translate_item(item):
    pass


def translate_item(item):
    pass


def translate_item(item):
    pass


@app.route("/predict", methods=['POST'])
def predict():
    item = int(request.form['item'])
    print(item)
    brand = int(request.form['brand'])
    print(brand)
    usage = int(request.form['usage'])
    print(usage)
    size = int(request.form['size'])
    print(size)
    prediction = model.predict([[item, brand, usage, size]])
    output = round(abs(prediction[0]), 2)
    return render_template('index1.html',
                           prediction_text=f'A house with {item} rooms per dwelling and {brand} and located {usage} km to {size} employment centers has a value of ${output}K')


if __name__ == "__main__":
    app.run()
